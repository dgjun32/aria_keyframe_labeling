#!/usr/bin/env python3
"""Real-time VLM runtime visualization for asynchronous keyframe simulation.

This script is intentionally separate from eval_vlm_baseline_asynchronous.py.
It visualizes the scenario the user actually wants to inspect:

1. The left panel plays the source video continuously in real time.
2. VLM requests run in the background without stopping playback.
3. The right panel shows pending requests, measured latency, returned actions,
   and keyframe / queue state as responses arrive.
4. The combined visualization is saved as an MP4, with original audio muxed in
   when available.

The scheduling rule matches the async evaluator semantics:
after a request starts, the next request may start after
max(dispatch_hop_sec, latency), so long latencies delay the next dispatch
and the newest available context window is used when the worker becomes free
again.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import json
import os
import subprocess
import tempfile
import textwrap
import time
from typing import Any

import cv2
import numpy as np
from google import genai

from eval_vlm_baseline import (
    DEFAULT_GEMINI_VIDEO_FPS_HINT,
    DEFAULT_THINKING_BUDGET,
    resolve_gemini_api_key,
    resolve_openai_api_key,
)
from eval_vlm_baseline_asynchronous import (
    AUDIO_MODE_NONE,
    AUDIO_MODE_VOICED,
    AUDIO_MODE_WHISPER,
    DEFAULT_MODEL,
    MODEL_CHOICES,
    PROMPT_INIT,
    PROMPT_UPDATE,
    KeyframeQueue,
    KeyframeQueueElement,
    apply_queue_update,
    burn_captions_on_video,
    call_gemini_with_timing,
    cut_video_segment,
    cut_audio_segment,
    extract_cost,
    extract_frame,
    get_video_meta,
    list_episode_names,
    merge_video_audio,
    parse_init_response,
    parse_resolution,
    parse_update_response,
    resolve_episode_files,
    whisper_transcribe_segment,
)


LEFT_VIDEO_WIDTH = 720
RIGHT_PANEL_WIDTH = 820
WINDOW_NAME = "VLM Runtime Visualization"
MIN_CANVAS_HEIGHT = 1080
DEFAULT_CONTEXT_WINDOW_SEC = 5.0
DEFAULT_DISPATCH_HOP_SEC = 5.0
BG_COLOR = (22, 24, 28)
CARD_COLOR = (34, 37, 43)
CARD_BORDER = (70, 74, 82)
TEXT_MAIN = (235, 235, 235)
TEXT_SUB = (180, 184, 192)
TEXT_DIM = (130, 136, 145)
TEXT_GOOD = (120, 220, 140)
TEXT_INFO = (120, 190, 255)
TEXT_WARN = (255, 190, 110)
TEXT_BAD = (255, 120, 120)


@dataclasses.dataclass
class PendingRequest:
    kind: str  # "init" or "update"
    step: int
    dispatch_time: float
    clip_start_sec: float
    clip_end_sec: float
    segment_path: str
    queue_revision_at_dispatch: int
    queue_snapshot: list[dict]
    queue_prompt_text: str
    future: Future


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    lines = textwrap.wrap(text, width=width) or [text]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _truncate(lines[-1], max(1, width))
    return lines


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = TEXT_MAIN,
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_card(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_BORDER, 1)
    _draw_text(canvas, title, x + 12, y + 22, scale=0.54, color=TEXT_MAIN)
    cv2.line(canvas, (x, y + 30), (x + w, y + 30), CARD_BORDER, 1)


def _get_stage_time(stage_times: dict[str, Any] | None, key: str) -> float:
    if not stage_times:
        return 0.0
    return float(stage_times.get(key) or 0.0)


def _format_stage_breakdown(stage_times: dict[str, Any] | None, total_worker_sec: float) -> list[str]:
    if not stage_times:
        return []

    gemini_sec = _get_stage_time(stage_times, "gemini_api_sec")
    whisper_sec = _get_stage_time(stage_times, "whisper_sec")
    clip_cut_sec = _get_stage_time(stage_times, "clip_cut_sec")
    caption_burn_sec = _get_stage_time(stage_times, "caption_burn_sec")
    audio_mux_sec = _get_stage_time(stage_times, "audio_mux_sec")
    save_input_sec = _get_stage_time(stage_times, "save_input_sec")
    prompt_build_sec = _get_stage_time(stage_times, "prompt_build_sec")
    parse_sec = _get_stage_time(stage_times, "parse_sec")
    non_gemini_sec = max(0.0, float(total_worker_sec or 0.0) - gemini_sec)

    return [
        (
            f"worker total {float(total_worker_sec or 0.0):.1f}s"
            f"  |  gemini {gemini_sec:.1f}s"
            f"  |  non-gemini {non_gemini_sec:.1f}s"
        ),
        (
            f"cut {clip_cut_sec:.1f}s  |  whisper {whisper_sec:.1f}s"
            f"  |  caption {caption_burn_sec:.1f}s  |  audio {audio_mux_sec:.1f}s"
        ),
        (
            f"save {save_input_sec:.1f}s  |  prompt {prompt_build_sec:.1f}s"
            f"  |  parse {parse_sec:.1f}s"
        ),
    ]


def _event_mutates_queue(event: dict[str, Any]) -> bool:
    return event.get("parsed_action") in {"Initialize", "Append", "Replace", "Remove"}


def _render_element_rows(
    canvas: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    items: list[dict],
    frame_cache: dict[int, np.ndarray],
    video_path: str,
    max_rows: int | None,
    available_height: int | None = None,
    thumb_size: tuple[int, int] = (62, 62),
) -> int:
    """Render queue/keyframe rows and return consumed height."""
    shown = items if max_rows is None else items[:max_rows]

    if not shown:
        _draw_text(canvas, "(none)", x, y + 18, scale=0.44, color=TEXT_DIM)
        return 26

    if available_height is not None and available_height > 0:
        row_h = max(34, min(76, available_height // max(1, len(shown))))
    else:
        row_h = 76
    thumb_w = min(thumb_size[0], max(24, row_h - 12))
    thumb_h = thumb_w
    desc_lines = 2 if row_h >= 68 else (1 if row_h >= 52 else 0)

    for idx, item in enumerate(shown):
        row_y = y + idx * row_h
        frame_idx = int(item.get("keyframe_frame_idx", 0))
        if frame_idx not in frame_cache:
            thumb = extract_frame(video_path, frame_idx, size=(thumb_w, thumb_h))
            if thumb is None:
                thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
            frame_cache[frame_idx] = thumb
        thumb = frame_cache[frame_idx]
        if thumb.shape[0] != thumb_h or thumb.shape[1] != thumb_w:
            thumb = cv2.resize(thumb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[row_y : row_y + thumb_h, x : x + thumb_w] = thumb

        instr = str(item.get("subtask_instruction", ""))
        desc = _truncate(item.get("keyframe_description", ""), 80)
        kf_t = float(item.get("keyframe_time_sec", 0.0))
        color = TEXT_GOOD if instr.startswith("pick up") else TEXT_INFO
        _draw_text(canvas, f"[{idx}] {instr}", x + thumb_w + 12, row_y + min(20, row_h - 12), scale=0.42 if row_h >= 52 else 0.36, color=color)
        _draw_text(canvas, f"keyframe @ {kf_t:.1f}s", x + thumb_w + 12, row_y + min(39, row_h - 4), scale=0.36 if row_h >= 52 else 0.33, color=TEXT_SUB)
        if desc_lines > 0:
            desc_y = row_y + (56 if row_h >= 68 else row_h - 2)
            for line_i, line in enumerate(_wrap(desc, 58, desc_lines)):
                _draw_text(canvas, line, x + thumb_w + 12, desc_y + line_i * 14, scale=0.33, color=TEXT_DIM)

    return max(26, len(shown) * row_h)


def _mux_audio_keep_video_duration(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux audio without shortening the rendered video if it outlasts the audio."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            output_path,
        ],
        check=True,
    )


def _request_worker(
    *,
    api_key: str,
    model: str,
    video_path: str,
    audio_path: str | None,
    audio_mode: str,
    embed_audio: bool,
    clip_start_sec: float,
    clip_end_sec: float,
    step: int,
    kind: str,
    queue_prompt_text: str,
    video_fps_hint: int,
    target_resolution: tuple[int, int] | None,
    segment_path: str,
) -> dict[str, Any]:
    """Build the real VLM input clip, send the request, and return raw outputs."""
    worker_t0 = time.perf_counter()
    client = genai.Client(api_key=api_key)
    _, fps = get_video_meta(video_path)
    stage_times: dict[str, float] = {
        "clip_cut_sec": 0.0,
        "whisper_sec": 0.0,
        "caption_burn_sec": 0.0,
        "audio_mux_sec": 0.0,
        "prompt_build_sec": 0.0,
        "gemini_api_sec": 0.0,
        "parse_sec": 0.0,
        "save_input_sec": 0.0,
    }

    t0 = time.perf_counter()
    clip_bytes = cut_video_segment(
        video_path,
        clip_start_sec,
        clip_end_sec,
        target_resolution=target_resolution,
    )
    stage_times["clip_cut_sec"] = round(time.perf_counter() - t0, 4)

    vlm_bytes = clip_bytes
    transcript_section = ""
    transcript_data = {"text": "", "words": []}
    has_audio = False

    if audio_mode == AUDIO_MODE_WHISPER and audio_path:
        t0 = time.perf_counter()
        transcript_data = whisper_transcribe_segment(audio_path, clip_start_sec, clip_end_sec)
        stage_times["whisper_sec"] = round(time.perf_counter() - t0, 4)
        if transcript_data["text"]:
            t0 = time.perf_counter()
            vlm_bytes = burn_captions_on_video(
                vlm_bytes,
                transcript_data["words"],
                clip_start_sec,
                fps,
            )
            stage_times["caption_burn_sec"] = round(time.perf_counter() - t0, 4)
            transcript_section = (
                f"\nA speech transcript was detected in this clip:\n"
                f"  \"{transcript_data['text']}\"\n"
                f"Use this transcript to understand the human's spoken instruction.\n"
            )

    if (embed_audio or audio_mode == AUDIO_MODE_VOICED) and audio_path:
        t0 = time.perf_counter()
        audio_seg_path = cut_audio_segment(audio_path, clip_start_sec, clip_end_sec)
        try:
            vlm_bytes = merge_video_audio(vlm_bytes, audio_seg_path)
            has_audio = True
        finally:
            os.unlink(audio_seg_path)
        stage_times["audio_mux_sec"] = round(time.perf_counter() - t0, 4)
        if audio_mode == AUDIO_MODE_VOICED:
            transcript_section += (
                "\nThis video clip contains an audio track with the human's spoken "
                "instruction. Listen carefully to understand what the human is saying.\n"
            )
        else:
            transcript_section += (
                "\nThe original audio track is included in the video clip in addition "
                "to any transcript text shown above.\n"
            )

    t0 = time.perf_counter()
    with open(segment_path, "wb") as f:
        f.write(vlm_bytes)
    stage_times["save_input_sec"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    if kind == "init":
        prompt = PROMPT_INIT.format(
            clip_duration=clip_end_sec - clip_start_sec,
            fps=fps,
            transcript_section=transcript_section,
        )
    else:
        prompt = PROMPT_UPDATE.format(
            clip_duration=clip_end_sec - clip_start_sec,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            fps=fps,
            queue_status=queue_prompt_text,
            transcript_section=transcript_section,
        )
    stage_times["prompt_build_sec"] = round(time.perf_counter() - t0, 4)

    response, raw_text, latency = call_gemini_with_timing(
        client,
        model,
        vlm_bytes,
        prompt,
        video_fps=video_fps_hint,
        has_audio=has_audio,
        thinking_budget=DEFAULT_THINKING_BUDGET,
    )
    stage_times["gemini_api_sec"] = round(latency, 4)
    cost = extract_cost(response, model)

    t0 = time.perf_counter()
    parsed = parse_init_response(raw_text) if kind == "init" else parse_update_response(raw_text)
    stage_times["parse_sec"] = round(time.perf_counter() - t0, 4)
    total_worker_sec = round(time.perf_counter() - worker_t0, 4)

    return {
        "step": step,
        "kind": kind,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "segment_path": segment_path,
        "request_size_kb": round(len(vlm_bytes) / 1024.0, 2),
        "has_audio": has_audio,
        "transcript_data": transcript_data,
        "prompt_preview": prompt[:600],
        "raw_response": raw_text,
        "latency": latency,
        "cost": cost,
        "parsed": parsed,
        "stage_times": stage_times,
        "total_worker_sec": total_worker_sec,
    }


def _process_completed_request(
    *,
    pending: PendingRequest,
    result: dict[str, Any] | None,
    current_runtime_time: float,
    current_queue_revision: int,
    queue: KeyframeQueue,
) -> dict[str, Any]:
    """Apply a completed response to the queue and return a timeline event."""
    queue_before = pending.queue_snapshot
    timeline_event: dict[str, Any] = {
        "event_type": "response",
        "kind": pending.kind,
        "step": pending.step,
        "dispatch_time": round(pending.dispatch_time, 3),
        "response_visible_time": round(current_runtime_time, 3),
        "clip_start_sec": round(pending.clip_start_sec, 3),
        "clip_end_sec": round(pending.clip_end_sec, 3),
        "segment_path": pending.segment_path,
        "queue_revision_at_dispatch": pending.queue_revision_at_dispatch,
        "queue_revision_before_apply": current_queue_revision,
        "stale_queue_snapshot": pending.queue_revision_at_dispatch != current_queue_revision,
        "queue_before": queue_before,
    }

    if result is None:
        timeline_event.update(
            {
                "parsed_action": "WorkerFailure",
                "parsed_action_details": None,
                "latency": None,
                "cost": None,
                "raw_response": "",
                "transcript_text": "",
                "last_keyframes": [],
                "queue_after": queue.snapshot(),
            }
        )
        return timeline_event

    timeline_event["latency"] = result["latency"]
    timeline_event["cost"] = result["cost"]
    timeline_event["raw_response"] = result["raw_response"]
    timeline_event["transcript_text"] = result["transcript_data"].get("text", "")
    timeline_event["request_size_kb"] = result["request_size_kb"]
    timeline_event["has_audio"] = result["has_audio"]
    timeline_event["stage_times"] = result.get("stage_times", {})
    timeline_event["total_worker_sec"] = result.get("total_worker_sec")

    parsed = result["parsed"]
    if pending.kind == "init":
        if parsed is None:
            timeline_event["parsed_action"] = "InitFailure"
            timeline_event["parsed_action_details"] = None
            timeline_event["last_keyframes"] = []
        else:
            elements = [KeyframeQueueElement.from_dict(item) for item in parsed["action_queue"]]
            queue.initialize(elements)
            timeline_event["parsed_action"] = "Initialize"
            timeline_event["parsed_action_details"] = {
                "reasoning": parsed.get("reasoning", ""),
                "human_instruction_summary": parsed.get("human_instruction_summary", ""),
                "num_elements": len(elements),
            }
            timeline_event["last_keyframes"] = queue.snapshot()
    else:
        if parsed is None:
            timeline_event["parsed_action"] = "ParseFailure"
            timeline_event["parsed_action_details"] = None
            timeline_event["last_keyframes"] = []
        else:
            action = apply_queue_update(queue, parsed)
            timeline_event["parsed_action"] = action
            timeline_event["parsed_action_details"] = {
                "reasoning": parsed.get("reasoning", ""),
                "action": action,
                "target_idx": parsed.get("target_idx"),
                "new_elements": parsed.get("new_elements", []),
            }
            timeline_event["last_keyframes"] = parsed.get("new_elements", [])

    timeline_event["queue_after"] = queue.snapshot()
    return timeline_event


def _render_right_panel(
    *,
    panel_height: int,
    runtime_time: float,
    video_time: float,
    video_duration: float,
    context_window_sec: float,
    dispatch_hop_sec: float,
    fully_async: bool,
    episode: str,
    model: str,
    pending_requests: list[PendingRequest],
    max_inflight: int,
    init_dispatched: bool,
    updates_enabled: bool,
    next_dispatch_time: float,
    queue: KeyframeQueue,
    last_response: dict[str, Any] | None,
    recent_events: list[dict[str, Any]],
    frame_cache: dict[int, np.ndarray],
    video_path: str,
) -> np.ndarray:
    height = max(520, panel_height)
    canvas = np.full((height, RIGHT_PANEL_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    card_x = 14
    card_w = RIGHT_PANEL_WIDTH - 28
    gap = 10
    request_y = 64
    request_h = 146
    response_y = request_y + request_h + gap
    response_h = 214
    recent_y = response_y + response_h + gap
    recent_h = 92
    latest_y = recent_y + recent_h + gap
    latest_h = 124
    queue_y = latest_y + latest_h + gap
    queue_h = max(120, height - queue_y - 14)

    _draw_text(canvas, f"Episode: {episode}", 18, 26, scale=0.58, color=TEXT_MAIN)
    _draw_text(
        canvas,
        (
            f"Model: {model}  |  video t={video_time:.1f}/{video_duration:.1f}s"
            f"  |  runtime t={runtime_time:.1f}s"
        ),
        18,
        50,
        scale=0.43,
        color=TEXT_SUB,
    )

    _draw_card(canvas, card_x, request_y, card_w, request_h, "Request Status")
    if not init_dispatched:
        if video_time < context_window_sec:
            status = f"Waiting for first {context_window_sec:.0f}s context before init dispatch."
        elif video_time >= video_duration:
            status = "Video playback ended. Waiting only for any final pending response."
        else:
            status = f"Init dispatch eligible now  |  ctx={context_window_sec:.1f}s"
        _draw_text(canvas, status, 28, request_y + 38, scale=0.46, color=TEXT_INFO)
    else:
        inflight = len(pending_requests)
        if not updates_enabled:
            status = (
                f"Init sent. Waiting for init response  |  inflight={inflight}/{max_inflight}"
            )
        elif video_time >= video_duration and inflight == 0:
            status = "Video playback ended. Waiting only for any final pending response."
        else:
            mode_text = "Fully async" if fully_async else "Sequential async"
            status = (
                f"{mode_text}  |  inflight={inflight}/{max_inflight}"
                f"  |  next target={next_dispatch_time:.1f}s"
                f"  |  hop={dispatch_hop_sec:.1f}s  ctx={context_window_sec:.1f}s"
            )
        _draw_text(canvas, status, 28, request_y + 38, scale=0.46, color=TEXT_INFO)
        shown_pending = sorted(pending_requests, key=lambda req: req.dispatch_time)[:3]
        for idx, req in enumerate(shown_pending):
            elapsed = max(0.0, runtime_time - req.dispatch_time)
            line_y = request_y + 62 + idx * 24
            _draw_text(
                canvas,
                f"[{idx}] {req.kind}:{req.step}  clip={req.clip_start_sec:.1f}-{req.clip_end_sec:.1f}s  elapsed={elapsed:.1f}s",
                28,
                line_y,
                scale=0.39,
                color=TEXT_WARN,
            )
        if inflight > len(shown_pending):
            _draw_text(
                canvas,
                f"... {inflight - len(shown_pending)} more pending requests",
                28,
                request_y + 136,
                scale=0.36,
                color=TEXT_DIM,
            )

    _draw_card(canvas, card_x, response_y, card_w, response_h, "Last Response")
    if last_response is None:
        _draw_text(canvas, "No VLM response yet.", 28, response_y + 40, scale=0.48, color=TEXT_DIM)
    else:
        action = last_response.get("parsed_action", "N/A")
        color = TEXT_GOOD if action in {"Initialize", "Append", "Replace", "Remove", "NoOp"} else TEXT_BAD
        stale_text = "  |  STALE SNAPSHOT" if last_response.get("stale_queue_snapshot") else ""
        _draw_text(
            canvas,
            (
                f"{action}  |  arrived @ {last_response.get('response_visible_time', 0.0):.1f}s"
                f"  |  gemini {float(last_response.get('latency') or 0.0):.1f}s"
                f"{stale_text}"
            ),
            28,
            response_y + 38,
            scale=0.47,
            color=color,
        )
        _draw_text(
            canvas,
            f"clip={float(last_response.get('clip_start_sec', 0.0)):.1f}-{float(last_response.get('clip_end_sec', 0.0)):.1f}s  cost=${float((last_response.get('cost') or {}).get('cost_usd', 0.0)):.4f}",
            28,
            response_y + 60,
            scale=0.41,
            color=TEXT_SUB,
        )
        stage_times = last_response.get("stage_times") or {}
        total_worker_sec = float(last_response.get("total_worker_sec") or 0.0)
        for idx, line in enumerate(_format_stage_breakdown(stage_times, total_worker_sec)):
            _draw_text(
                canvas,
                line,
                28,
                response_y + 82 + idx * 18,
                scale=0.37,
                color=TEXT_SUB if idx == 0 else TEXT_DIM,
            )
        details = last_response.get("parsed_action_details") or {}
        summary = details.get("human_instruction_summary", "")
        summary_y = response_y + 138
        if summary:
            for idx, line in enumerate(_wrap(summary, 78, 2)):
                _draw_text(canvas, f"Summary: {line}" if idx == 0 else f"         {line}", 28, summary_y + idx * 18, scale=0.38, color=TEXT_SUB)
        reasoning = details.get("reasoning", "")
        base_y = summary_y + (36 if summary else 0)
        for idx, line in enumerate(_wrap(reasoning, 82, 3)):
            _draw_text(canvas, f"Reasoning: {line}" if idx == 0 else f"           {line}", 28, base_y + idx * 18, scale=0.36, color=TEXT_DIM)

    _draw_card(canvas, card_x, recent_y, card_w, recent_h, "Recent Actions")
    if not recent_events:
        _draw_text(canvas, "(no completed responses yet)", 28, recent_y + 40, scale=0.44, color=TEXT_DIM)
    else:
        for idx, event in enumerate(recent_events[-2:]):
            y = recent_y + 38 + idx * 24
            action = event.get("parsed_action", "N/A")
            line = (
                f"[step {event.get('step', '?')}] {action}  "
                f"t={float(event.get('response_visible_time', 0.0)):.1f}s  "
                f"gemini={float(event.get('latency') or 0.0):.1f}s"
                f"{'  stale' if event.get('stale_queue_snapshot') else ''}"
            )
            color = TEXT_GOOD if action in {"Initialize", "Append", "Replace", "Remove", "NoOp"} else TEXT_BAD
            _draw_text(canvas, line, 28, y, scale=0.4, color=color)

    _draw_card(canvas, card_x, latest_y, card_w, latest_h, "Latest Keyframe Results")
    latest_items = (last_response or {}).get("last_keyframes", [])
    _render_element_rows(
        canvas,
        x=28,
        y=latest_y + 40,
        width=RIGHT_PANEL_WIDTH - 56,
        items=latest_items,
        frame_cache=frame_cache,
        video_path=video_path,
        max_rows=1 if latest_h < 130 else 2,
        available_height=latest_h - 48,
    )

    _draw_card(canvas, card_x, queue_y, card_w, queue_h, "Current Queue")
    _render_element_rows(
        canvas,
        x=28,
        y=queue_y + 40,
        width=RIGHT_PANEL_WIDTH - 56,
        items=queue.snapshot(),
        frame_cache=frame_cache,
        video_path=video_path,
        max_rows=None,
        available_height=queue_h - 48,
    )
    return canvas


def _compose_frame(
    *,
    source_frame: np.ndarray,
    runtime_time: float,
    video_time: float,
    video_duration: float,
    context_window_sec: float,
    dispatch_hop_sec: float,
    fully_async: bool,
    episode: str,
    model: str,
    pending_requests: list[PendingRequest],
    max_inflight: int,
    init_dispatched: bool,
    updates_enabled: bool,
    next_dispatch_time: float,
    queue: KeyframeQueue,
    last_response: dict[str, Any] | None,
    recent_events: list[dict[str, Any]],
    frame_cache: dict[int, np.ndarray],
    video_path: str,
    left_height: int,
    canvas_height: int,
) -> np.ndarray:
    left_frame = cv2.resize(
        source_frame,
        (LEFT_VIDEO_WIDTH, left_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    left_canvas = np.full((canvas_height, LEFT_VIDEO_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    top = max(0, (canvas_height - left_height) // 2)
    left_canvas[top : top + left_height, :] = left_frame
    right_panel = _render_right_panel(
        panel_height=canvas_height,
        runtime_time=runtime_time,
        video_time=video_time,
        video_duration=video_duration,
        context_window_sec=context_window_sec,
        dispatch_hop_sec=dispatch_hop_sec,
        fully_async=fully_async,
        episode=episode,
        model=model,
        pending_requests=pending_requests,
        max_inflight=max_inflight,
        init_dispatched=init_dispatched,
        updates_enabled=updates_enabled,
        next_dispatch_time=next_dispatch_time,
        queue=queue,
        last_response=last_response,
        recent_events=recent_events,
        frame_cache=frame_cache,
        video_path=video_path,
    )
    return np.hstack([left_canvas, right_panel])


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize real-time asynchronous VLM behavior with continuous playback",
    )
    parser.add_argument("--episode", required=True)
    parser.add_argument("--dataset-dir", default="./process")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_CHOICES)
    parser.add_argument("--output-dir", default="./eval_results_async_runtime_viz")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_GEMINI_VIDEO_FPS_HINT)
    parser.add_argument(
        "--segment-duration-sec",
        "--context-window-sec",
        dest="context_window_sec",
        type=float,
        default=DEFAULT_CONTEXT_WINDOW_SEC,
        help="Length of the video context window sent to the VLM.",
    )
    parser.add_argument(
        "--dispatch-hop-sec",
        type=float,
        default=DEFAULT_DISPATCH_HOP_SEC,
        help="Minimum target interval between dispatches before latency is considered.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=8,
        help="Maximum number of concurrent in-flight VLM requests when --fully-async is enabled.",
    )
    parser.add_argument(
        "--fully-async",
        action="store_true",
        help="Enable overlapping in-flight VLM requests after init. Default is sequential async.",
    )
    parser.add_argument("--target-resolution", default="256")
    parser.add_argument("--use-gaze-video", action="store_true")
    parser.add_argument(
        "--embed-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Embed the sidecar audio into clips. Enabled by default and uses native audio mode.",
    )
    parser.add_argument("--show-window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end-hold-sec", type=float, default=2.0, help="How long to keep the final state visible after the last response")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--verbose", action="store_true")

    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--whisper", action="store_true")
    audio_group.add_argument("--voiced-video", action="store_true")

    args = parser.parse_args()

    api_key = resolve_gemini_api_key(args.api_key)
    if not api_key:
        print("ERROR: No Gemini API key found.")
        return

    files = resolve_episode_files(args.dataset_dir, args.episode)
    video_path = files["rgb_with_gaze"] if args.use_gaze_video else files["rgb"]
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        print("Available episodes:")
        for name in list_episode_names(args.dataset_dir):
            print(f"  {name}")
        return

    if args.whisper:
        audio_mode = AUDIO_MODE_WHISPER
        openai_api_key = resolve_openai_api_key()
        if not openai_api_key:
            print("ERROR: --whisper requires OPENAI_API_KEY.")
            return
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif args.voiced_video or args.embed_audio:
        audio_mode = AUDIO_MODE_VOICED
    else:
        audio_mode = AUDIO_MODE_NONE

    needs_audio_file = args.embed_audio or (audio_mode != AUDIO_MODE_NONE)
    audio_path = files["audio"] if needs_audio_file else None
    if audio_path and not os.path.exists(audio_path):
        print(f"WARNING: Audio file not found: {audio_path}, continuing without audio")
        audio_path = None
        audio_mode = AUDIO_MODE_NONE

    target_resolution = parse_resolution(args.target_resolution)
    context_window_sec = max(0.5, float(args.context_window_sec))
    dispatch_hop_sec = max(0.1, float(args.dispatch_hop_sec))
    fully_async = bool(args.fully_async)
    max_inflight = max(1, int(args.max_inflight if fully_async else 1))

    ep_output_dir = os.path.join(args.output_dir, args.episode)
    os.makedirs(ep_output_dir, exist_ok=True)
    inputs_dir = os.path.join(ep_output_dir, "vlm_inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    timeline_path = os.path.join(ep_output_dir, "runtime_timeline.json")
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video.close()
    output_video_path = os.path.join(ep_output_dir, "runtime_visualization.mp4")

    total_frames, fps = get_video_meta(video_path)
    video_duration = total_frames / fps
    init_dispatch_threshold = min(context_window_sec, video_duration)

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left_height = int(orig_h * (LEFT_VIDEO_WIDTH / max(orig_w, 1)))
    canvas_height = max(left_height, MIN_CANVAS_HEIGHT)
    combined_w = LEFT_VIDEO_WIDTH + RIGHT_PANEL_WIDTH
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video.name, fourcc, fps, (combined_w, canvas_height))

    queue = KeyframeQueue()
    queue_revision = 0
    frame_cache: dict[int, np.ndarray] = {}
    events: list[dict[str, Any]] = []
    recent_events: list[dict[str, Any]] = []
    last_response: dict[str, Any] | None = None
    pending_requests: list[PendingRequest] = []
    init_dispatched = False
    updates_enabled = False
    next_dispatch_time = init_dispatch_threshold
    next_update_step = 1
    latest_frame: np.ndarray | None = None
    show_window = args.show_window

    if show_window:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"[WARN] Could not open preview window: {exc}")
            show_window = False

    print(f"\n{'='*72}")
    print(f"Runtime Visualization — {args.episode}")
    print(
        f"video={video_duration:.1f}s @ {fps:.1f}fps  model={args.model}  "
        f"audio_mode={audio_mode}  context={context_window_sec:.1f}s  "
        f"hop={dispatch_hop_sec:.1f}s  fully_async={fully_async}  max_inflight={max_inflight}"
    )
    print(f"output={ep_output_dir}")
    print(f"{'='*72}")

    def dispatch_request(kind: str, step: int, dispatch_time: float) -> PendingRequest:
        clip_end_sec = min(dispatch_time, video_duration)
        clip_start_sec = max(0.0, clip_end_sec - context_window_sec)
        segment_path = os.path.join(
            inputs_dir,
            f"step_{step:03d}_{kind}_{clip_start_sec:.1f}s_{clip_end_sec:.1f}s.mp4",
        )
        queue_snapshot = queue.snapshot()
        queue_prompt_text = queue.format_for_prompt()
        future = executor.submit(
            _request_worker,
            api_key=api_key,
            model=args.model,
            video_path=video_path,
            audio_path=audio_path,
            audio_mode=audio_mode,
            embed_audio=args.embed_audio,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            step=step,
            kind=kind,
            queue_prompt_text=queue_prompt_text,
            video_fps_hint=args.video_fps,
            target_resolution=target_resolution,
            segment_path=segment_path,
        )
        dispatch_event = {
            "event_type": "dispatch",
            "kind": kind,
            "step": step,
            "dispatch_time": round(dispatch_time, 3),
            "clip_start_sec": round(clip_start_sec, 3),
            "clip_end_sec": round(clip_end_sec, 3),
            "segment_path": segment_path,
            "queue_snapshot": queue_snapshot,
        }
        events.append(dispatch_event)
        print(
            f"[dispatch] {kind.upper()} step={step}  "
            f"t={dispatch_time:.1f}s  clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s"
        )
        return PendingRequest(
            kind=kind,
            step=step,
            dispatch_time=dispatch_time,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            segment_path=segment_path,
            queue_revision_at_dispatch=queue_revision,
            queue_snapshot=queue_snapshot,
            queue_prompt_text=queue_prompt_text,
            future=future,
        )

    def handle_completed_request(pending: PendingRequest, runtime_time: float) -> None:
        nonlocal last_response, recent_events, queue_revision, updates_enabled, next_dispatch_time

        try:
            result = pending.future.result()
        except Exception as exc:
            print(f"[ERROR] worker failed: {exc}")
            result = None

        response_event = _process_completed_request(
            pending=pending,
            result=result,
            current_runtime_time=runtime_time,
            current_queue_revision=queue_revision,
            queue=queue,
        )
        events.append(response_event)
        recent_events.append(response_event)
        recent_events = recent_events[-8:]
        last_response = response_event

        latency = float(response_event.get("latency") or 0.0)
        total_worker_sec = float(response_event.get("total_worker_sec") or 0.0)
        stage_times = response_event.get("stage_times") or {}
        stale = bool(response_event.get("stale_queue_snapshot"))
        if _event_mutates_queue(response_event):
            queue_revision += 1
        if pending.kind == "init":
            updates_enabled = True
            if fully_async:
                next_dispatch_time = max(runtime_time, init_dispatch_threshold + dispatch_hop_sec)
            else:
                next_dispatch_time = pending.dispatch_time + max(dispatch_hop_sec, latency)
        elif not fully_async:
            next_dispatch_time = pending.dispatch_time + max(dispatch_hop_sec, latency)

        stale_suffix = "  stale=yes" if stale else ""
        print(
            f"[response] {response_event.get('parsed_action')}  "
            f"step={pending.step}  arrived={runtime_time:.1f}s  "
            f"gemini={latency:.1f}s  total={total_worker_sec:.1f}s{stale_suffix}"
        )
        print(
            "           "
            f"cut={_get_stage_time(stage_times, 'clip_cut_sec'):.1f}s  "
            f"whisper={_get_stage_time(stage_times, 'whisper_sec'):.1f}s  "
            f"caption={_get_stage_time(stage_times, 'caption_burn_sec'):.1f}s  "
            f"audio={_get_stage_time(stage_times, 'audio_mux_sec'):.1f}s  "
            f"parse={_get_stage_time(stage_times, 'parse_sec'):.1f}s"
        )
        if args.verbose and response_event.get("parsed_action_details"):
            details = response_event["parsed_action_details"]
            reasoning = _truncate(details.get("reasoning", ""), 180)
            if reasoning:
                print(f"           reasoning: {reasoning}")

    def poll_completed_requests(runtime_time: float) -> None:
        nonlocal pending_requests

        completed: list[PendingRequest] = []
        remaining: list[PendingRequest] = []
        for req in pending_requests:
            if req.future.done():
                completed.append(req)
            else:
                remaining.append(req)
        pending_requests = remaining
        for req in sorted(completed, key=lambda item: item.dispatch_time):
            handle_completed_request(req, runtime_time)

    executor = ThreadPoolExecutor(max_workers=max_inflight)
    start_wall = time.monotonic()
    frame_interval = 1.0 / max(fps, 1.0)
    next_tick = start_wall
    current_frame_idx = -1

    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)

            runtime_time = time.monotonic() - start_wall
            if runtime_time >= video_duration:
                break

            desired_frame_idx = min(total_frames - 1, int(runtime_time * fps))
            if desired_frame_idx > current_frame_idx:
                while current_frame_idx + 1 < desired_frame_idx:
                    if not cap.grab():
                        break
                    current_frame_idx += 1
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_idx = desired_frame_idx
                latest_frame = frame.copy()
            elif latest_frame is not None:
                frame = latest_frame
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_idx = max(0, desired_frame_idx)
                latest_frame = frame.copy()

            video_time = min(current_frame_idx / max(fps, 1.0), video_duration)

            poll_completed_requests(runtime_time)

            if not init_dispatched and runtime_time >= init_dispatch_threshold and len(pending_requests) < max_inflight:
                pending_requests.append(dispatch_request("init", 0, runtime_time))
                init_dispatched = True
            elif updates_enabled and runtime_time < video_duration and runtime_time >= next_dispatch_time and len(pending_requests) < max_inflight:
                pending_requests.append(dispatch_request("update", next_update_step, runtime_time))
                next_update_step += 1
                if fully_async:
                    next_dispatch_time = runtime_time + dispatch_hop_sec

            combined = _compose_frame(
                source_frame=frame,
                runtime_time=runtime_time,
                video_time=video_time,
                video_duration=video_duration,
                context_window_sec=context_window_sec,
                dispatch_hop_sec=dispatch_hop_sec,
                fully_async=fully_async,
                episode=args.episode,
                model=args.model,
                pending_requests=pending_requests,
                max_inflight=max_inflight,
                init_dispatched=init_dispatched,
                updates_enabled=updates_enabled,
                next_dispatch_time=next_dispatch_time,
                queue=queue,
                last_response=last_response,
                recent_events=recent_events,
                frame_cache=frame_cache,
                video_path=video_path,
                left_height=left_height,
                canvas_height=canvas_height,
            )
            writer.write(combined)

            if show_window:
                try:
                    cv2.imshow(WINDOW_NAME, combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        show_window = False
                        cv2.destroyWindow(WINDOW_NAME)
                except cv2.error as exc:
                    print(f"[WARN] live preview disabled: {exc}")
                    show_window = False
            next_tick += frame_interval

        if latest_frame is None:
            print("ERROR: Failed to read any video frame.")
            return

        post_deadline = time.monotonic() + max(0.0, args.end_hold_sec)
        while pending_requests or time.monotonic() < post_deadline:
            runtime_time = time.monotonic() - start_wall
            video_time = video_duration

            before_pending = len(pending_requests)
            poll_completed_requests(runtime_time)
            if len(pending_requests) < before_pending:
                post_deadline = time.monotonic() + max(0.0, args.end_hold_sec)

            combined = _compose_frame(
                source_frame=latest_frame,
                runtime_time=runtime_time,
                video_time=video_time,
                video_duration=video_duration,
                context_window_sec=context_window_sec,
                dispatch_hop_sec=dispatch_hop_sec,
                fully_async=fully_async,
                episode=args.episode,
                model=args.model,
                pending_requests=pending_requests,
                max_inflight=max_inflight,
                init_dispatched=init_dispatched,
                updates_enabled=updates_enabled,
                next_dispatch_time=next_dispatch_time,
                queue=queue,
                last_response=last_response,
                recent_events=recent_events,
                frame_cache=frame_cache,
                video_path=video_path,
                left_height=left_height,
                canvas_height=canvas_height,
            )
            writer.write(combined)
            if show_window:
                try:
                    cv2.imshow(WINDOW_NAME, combined)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        show_window = False
                        cv2.destroyWindow(WINDOW_NAME)
                except cv2.error:
                    show_window = False
            time.sleep(max(0.0, 1.0 / max(fps, 1.0)))
    finally:
        cap.release()
        writer.release()
        executor.shutdown(wait=True)
        if show_window:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass

    final_video_path = output_video_path
    if audio_path and os.path.exists(audio_path):
        _mux_audio_keep_video_duration(temp_video.name, audio_path, output_video_path)
        os.unlink(temp_video.name)
    else:
        os.replace(temp_video.name, output_video_path)

    summary = {
        "episode": args.episode,
        "model": args.model,
        "video_duration_sec": round(video_duration, 3),
        "context_window_sec": round(context_window_sec, 3),
        "dispatch_hop_sec": round(dispatch_hop_sec, 3),
        "max_inflight": max_inflight,
        "fully_async": fully_async,
        "audio_mode": audio_mode,
        "embed_audio": args.embed_audio,
        "show_window": args.show_window,
        "events": events,
        "final_queue": queue.snapshot(),
        "output_video": final_video_path,
    }
    _save_json(timeline_path, summary)

    print(f"\nSaved runtime visualization: {output_video_path}")
    print(f"Saved runtime timeline: {timeline_path}")
    print(f"Saved exact VLM input clips under: {inputs_dir}")


if __name__ == "__main__":
    main()
