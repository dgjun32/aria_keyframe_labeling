#!/usr/bin/env python3
"""Simple text-first VLM simulator with editable prompt templates.

This tool intentionally avoids the heavy video visualization UI.
It lets the user:
1. Pick an episode and timing parameters.
2. Edit the init/update prompt templates directly.
3. Run a sequential async simulation over the video timeline.
4. Inspect dispatch times, response arrival times, and raw text responses.
"""

from __future__ import annotations

import json
import os
import queue as queue_mod
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import scrolledtext

import cv2
from google import genai
from PIL import Image, ImageTk

from eval_vlm_baseline import (
    DEFAULT_GEMINI_VIDEO_FPS_HINT,
    resolve_gemini_api_key,
)
from eval_vlm_baseline_asynchronous import (
    AUDIO_MODE_VOICED,
    DEFAULT_MODEL,
    MODEL_CHOICES,
    PROMPT_INIT,
    PROMPT_UPDATE,
    KeyframeQueue,
    KeyframeQueueElement,
    apply_queue_update,
    call_gemini_with_timing,
    cut_audio_segment,
    cut_video_segment,
    extract_cost,
    get_video_meta,
    list_episode_names,
    merge_video_audio,
    parse_init_response,
    parse_resolution,
    parse_update_response,
    resolve_episode_files,
)


DEFAULT_CONTEXT_WINDOW_SEC = 5.0
DEFAULT_DISPATCH_HOP_SEC = 5.0
DEFAULT_MAX_INFLIGHT = 4
RESPONSE_MODE_FREEFORM = "freeform"
RESPONSE_MODE_QUEUE = "queue"
RESPONSE_MODE_CHOICES = (RESPONSE_MODE_FREEFORM, RESPONSE_MODE_QUEUE)
EXECUTION_MODE_SEQUENTIAL = "sequential"
EXECUTION_MODE_ASYNC = "async"
EXECUTION_MODE_CHOICES = (EXECUTION_MODE_SEQUENTIAL, EXECUTION_MODE_ASYNC)
FREEFORM_QUEUE_DISABLED = "(freeform mode: queue disabled)"
POLL_MS = 120
RUN_OUTPUT_ROOT = "./simple_vlm_text_runs"
TIMELINE_ROW_HEIGHT = 46
TIMELINE_TOP_PAD = 28
TIMELINE_BOTTOM_PAD = 26
TIMELINE_LEFT_PAD = 126
TIMELINE_RIGHT_PAD = 20
TIMELINE_CURSOR_REFRESH_SEC = 0.12
PLAYBACK_ROW_MIN_HEIGHT = 340


@dataclass
class SimulationConfig:
    dataset_dir: str
    episode: str
    model: str
    response_mode: str
    execution_mode: str
    max_inflight: int
    context_window_sec: float
    dispatch_hop_sec: float
    video_fps_hint: int
    target_resolution: tuple[int, int] | None
    use_gaze_video: bool
    embed_audio: bool
    precompute_init: bool
    init_prompt_template: str
    update_prompt_template: str


def _format_queue_snapshot(queue: KeyframeQueue) -> str:
    if len(queue) == 0:
        return "(queue is empty)"
    lines = []
    for idx in range(len(queue)):
        elem = queue[idx]
        lines.append(
            f"[{idx}] {elem.subtask_instruction} | "
            f"{elem.keyframe_description} | t={elem.keyframe_time_sec:.1f}s"
        )
    return "\n".join(lines)


def _format_prompt(template: str, *, clip_start_sec: float, clip_end_sec: float, fps: float, queue_status: str) -> str:
    values = {
        "clip_duration": clip_end_sec - clip_start_sec,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "fps": fps,
        "queue_status": queue_status,
        "transcript_section": "",
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Prompt template has an unknown or missing placeholder: {{{missing}}}. "
            "Allowed placeholders are {clip_duration}, {clip_start_sec}, {clip_end_sec}, "
            "{fps}, {queue_status}, {transcript_section}."
        ) from exc


def _make_request_id(kind: str, step: int) -> str:
    return f"{kind}:{step}"


def _compact_path(path: str | None, *, max_len: int = 64) -> str:
    if not path:
        return "(none)"
    norm = os.path.abspath(path)
    if len(norm) <= max_len:
        return norm
    base = os.path.basename(norm)
    parent = os.path.basename(os.path.dirname(norm))
    compact = os.path.join("...", parent, base) if parent else os.path.join("...", base)
    if len(compact) <= max_len:
        return compact
    if len(base) + 4 <= max_len:
        return os.path.join("...", base)
    keep = max(8, max_len - 7)
    return "..." + base[-keep:]


def _truncate_text(text: str, *, max_len: int = 84) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _summarize_freeform_response(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return "Freeform response (empty)"

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            timestamp = first.get("timestamp")
            description = first.get("description")
            if isinstance(description, str) and description.strip():
                prefix = f"{len(payload)} events"
                if timestamp:
                    prefix += f" from {timestamp}"
                return _truncate_text(f"{prefix}: {description}")
        return _truncate_text(f"Freeform list response ({len(payload)} items)")

    if isinstance(payload, dict):
        for key in ("summary", "description", "reasoning", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate_text(value)
        return _truncate_text(f"Freeform JSON response with keys: {', '.join(payload.keys())}")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped not in {"[", "]", "{", "}"}:
            return _truncate_text(stripped)
    return "Freeform response"


def _run_request(
    *,
    client: genai.Client,
    config: SimulationConfig,
    video_path: str,
    audio_path: str | None,
    clip_start_sec: float,
    clip_end_sec: float,
    kind: str,
    queue_status: str,
    save_clip_path: str | None = None,
) -> dict[str, Any]:
    worker_t0 = time.perf_counter()
    _, fps = get_video_meta(video_path)

    stage_times: dict[str, float] = {
        "clip_cut_sec": 0.0,
        "audio_mux_sec": 0.0,
        "save_clip_sec": 0.0,
        "prompt_build_sec": 0.0,
        "gemini_api_sec": 0.0,
        "parse_sec": 0.0,
    }

    t0 = time.perf_counter()
    video_bytes = cut_video_segment(
        video_path,
        clip_start_sec,
        clip_end_sec,
        target_resolution=config.target_resolution,
    )
    stage_times["clip_cut_sec"] = round(time.perf_counter() - t0, 4)

    has_audio = False
    if config.embed_audio and audio_path:
        t0 = time.perf_counter()
        audio_seg_path = cut_audio_segment(audio_path, clip_start_sec, clip_end_sec)
        try:
            video_bytes = merge_video_audio(video_bytes, audio_seg_path)
            has_audio = True
        finally:
            os.unlink(audio_seg_path)
        stage_times["audio_mux_sec"] = round(time.perf_counter() - t0, 4)

    if save_clip_path:
        t0 = time.perf_counter()
        os.makedirs(os.path.dirname(save_clip_path) or ".", exist_ok=True)
        with open(save_clip_path, "wb") as f:
            f.write(video_bytes)
        stage_times["save_clip_sec"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    template = config.init_prompt_template if kind == "init" else config.update_prompt_template
    prompt = _format_prompt(
        template,
        clip_start_sec=clip_start_sec,
        clip_end_sec=clip_end_sec,
        fps=fps,
        queue_status=queue_status,
    )
    stage_times["prompt_build_sec"] = round(time.perf_counter() - t0, 4)

    response, raw_text, latency = call_gemini_with_timing(
        client,
        config.model,
        video_bytes,
        prompt,
        video_fps=config.video_fps_hint,
        has_audio=has_audio,
        response_mime_type="application/json" if config.response_mode == RESPONSE_MODE_QUEUE else "text/plain",
    )
    stage_times["gemini_api_sec"] = round(latency, 4)

    t0 = time.perf_counter()
    if config.response_mode == RESPONSE_MODE_QUEUE:
        parsed = parse_init_response(raw_text) if kind == "init" else parse_update_response(raw_text)
    else:
        parsed = None
    stage_times["parse_sec"] = round(time.perf_counter() - t0, 4)

    return {
        "kind": kind,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "response_mode": config.response_mode,
        "prompt": prompt,
        "raw_response": raw_text,
        "parsed": parsed,
        "cost": extract_cost(response, config.model),
        "latency": latency,
        "stage_times": stage_times,
        "total_worker_sec": round(time.perf_counter() - worker_t0, 4),
        "request_size_kb": round(len(video_bytes) / 1024.0, 2),
        "has_audio": has_audio,
        "audio_mode": AUDIO_MODE_VOICED if has_audio else "none",
        "clip_path": save_clip_path,
    }


class TextSimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simple VLM Text Simulator")
        self.root.geometry("1720x1040")

        self.event_queue: queue_mod.Queue[dict[str, Any]] = queue_mod.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None

        self.status_var = tk.StringVar(value="Idle")
        self.running_var = tk.StringVar(value="Not running")
        self.selected_clip_var = tk.StringVar(value="Clip: (none)")
        self.selected_source_var = tk.StringVar(value="Source: (none)")
        self.source_playback_var = tk.StringVar(value="Source playback: idle")
        self.clip_playback_var = tk.StringVar(value="Clip playback: idle")

        self.dataset_dir_var = tk.StringVar(value="./process")
        self.episode_var = tk.StringVar(value="")
        self.episode_names: list[str] = []
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.response_mode_var = tk.StringVar(value=RESPONSE_MODE_FREEFORM)
        self.execution_mode_var = tk.StringVar(value=EXECUTION_MODE_SEQUENTIAL)
        self.max_inflight_var = tk.StringVar(value=str(DEFAULT_MAX_INFLIGHT))
        self.context_window_var = tk.StringVar(value=str(DEFAULT_CONTEXT_WINDOW_SEC))
        self.dispatch_hop_var = tk.StringVar(value=str(DEFAULT_DISPATCH_HOP_SEC))
        self.video_fps_var = tk.StringVar(value=str(DEFAULT_GEMINI_VIDEO_FPS_HINT))
        self.target_resolution_var = tk.StringVar(value="256")
        self.use_gaze_var = tk.BooleanVar(value=False)
        self.embed_audio_var = tk.BooleanVar(value=True)
        self.precompute_init_var = tk.BooleanVar(value=True)
        self.current_run_dir: str | None = None
        self.history_payloads: list[dict[str, Any]] = []
        self.history_index_by_request_id: dict[str, int] = {}
        self.context_menu: tk.Menu | None = None
        self.source_video_job: str | None = None
        self.source_video_cap: cv2.VideoCapture | None = None
        self.source_video_path: str | None = None
        self.source_video_start_wall: float | None = None
        self.source_video_fps: float = 0.0
        self.source_video_total_frames: int = 0
        self.source_video_current_idx: int = -1
        self.source_video_photo_ref: ImageTk.PhotoImage | None = None
        self.source_video_duration_sec: float = 0.0
        self.clip_video_job: str | None = None
        self.clip_video_cap: cv2.VideoCapture | None = None
        self.clip_video_path: str | None = None
        self.clip_video_fps: float = 0.0
        self.clip_video_photo_ref: ImageTk.PhotoImage | None = None
        self.timeline_video_duration_sec: float = 0.0
        self.timeline_current_runtime_sec: float = 0.0
        self.timeline_last_cursor_sec: float = -1.0
        self.timeline_rows: list[dict[str, Any]] = []
        self.timeline_index_by_request_id: dict[str, int] = {}
        self.timeline_selected_request_id: str | None = None
        self.timeline_detail_var = tk.StringVar(
            value="Timeline: blue=input clip, orange=dispatch, green=arrival, red dashed=video end"
        )

        self._build_ui()
        self._install_clipboard_shortcuts()
        self._load_default_episode()
        self.root.after(POLL_MS, self._poll_events)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for idx in range(10):
            controls.columnconfigure(idx, weight=1 if idx in (1, 3, 5, 7) else 0)

        ttk.Label(controls, text="Dataset Dir").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.dataset_dir_var).grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(controls, text="Episode").grid(row=0, column=2, sticky="w")
        self.episode_combo = ttk.Combobox(
            controls,
            textvariable=self.episode_var,
            values=self.episode_names,
            state="readonly",
        )
        self.episode_combo.grid(row=0, column=3, sticky="ew", padx=(6, 12))
        ttk.Button(controls, text="Refresh Episodes", command=self._load_default_episode).grid(row=0, column=4, padx=(0, 12))
        ttk.Checkbutton(controls, text="Use Gaze Video", variable=self.use_gaze_var).grid(row=0, column=5, sticky="w")
        ttk.Checkbutton(controls, text="Include Audio", variable=self.embed_audio_var).grid(row=0, column=6, sticky="w")
        ttk.Label(controls, text="Response Mode").grid(row=0, column=7, sticky="e")
        ttk.Combobox(
            controls,
            textvariable=self.response_mode_var,
            values=RESPONSE_MODE_CHOICES,
            state="readonly",
            width=12,
        ).grid(row=0, column=8, columnspan=2, sticky="w", padx=(6, 0))

        ttk.Label(controls, text="Model").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(controls, textvariable=self.model_var, values=MODEL_CHOICES, state="readonly").grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(10, 0))
        ttk.Label(controls, text="Context Window").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.context_window_var, width=8).grid(row=1, column=3, sticky="w", padx=(6, 12), pady=(10, 0))
        ttk.Label(controls, text="Dispatch Hop").grid(row=1, column=4, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.dispatch_hop_var, width=8).grid(row=1, column=5, sticky="w", padx=(6, 12), pady=(10, 0))
        ttk.Label(controls, text="Video FPS Hint").grid(row=1, column=6, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.video_fps_var, width=8).grid(row=1, column=7, sticky="w", padx=(6, 12), pady=(10, 0))
        ttk.Label(controls, text="Resolution").grid(row=1, column=8, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.target_resolution_var, width=8).grid(row=1, column=9, sticky="w", padx=(6, 0), pady=(10, 0))

        actions = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(actions, text="Start", command=self.start).pack(side="left")
        ttk.Button(actions, text="Stop", command=self.stop).pack(side="left", padx=8)
        ttk.Label(actions, text="Execution").pack(side="left", padx=(14, 4))
        ttk.Combobox(
            actions,
            textvariable=self.execution_mode_var,
            values=EXECUTION_MODE_CHOICES,
            state="readonly",
            width=11,
        ).pack(side="left")
        ttk.Checkbutton(
            actions,
            text="Precompute Init",
            variable=self.precompute_init_var,
        ).pack(side="left", padx=(12, 0))
        ttk.Label(actions, text="Max Inflight").pack(side="left", padx=(12, 4))
        ttk.Entry(actions, textvariable=self.max_inflight_var, width=4).pack(side="left")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=14)
        ttk.Label(actions, textvariable=self.running_var).pack(side="left", padx=14)

        hint = ttk.Label(
            self.root,
            padding=(10, 0, 10, 6),
            text=(
                "Prompt placeholders: {clip_duration}, {clip_start_sec}, {clip_end_sec}, "
                "{fps}, {queue_status}, {transcript_section}"
            ),
        )
        hint.grid(row=2, column=0, sticky="w")

        prompt_frame = ttk.Frame(self.root, padding=10)
        prompt_frame.grid(row=3, column=0, sticky="nsew")
        prompt_frame.columnconfigure(0, weight=1)
        prompt_frame.columnconfigure(1, weight=1)
        prompt_frame.rowconfigure(0, weight=1)

        init_box = ttk.LabelFrame(prompt_frame, text="Init Prompt Template")
        init_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        init_box.rowconfigure(0, weight=1)
        init_box.columnconfigure(0, weight=1)
        self.init_prompt_text = scrolledtext.ScrolledText(init_box, wrap="word", font=("Menlo", 11))
        self.init_prompt_text.grid(row=0, column=0, sticky="nsew")
        self.init_prompt_text.insert("1.0", PROMPT_INIT)

        update_box = ttk.LabelFrame(prompt_frame, text="Update Prompt Template")
        update_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        update_box.rowconfigure(0, weight=1)
        update_box.columnconfigure(0, weight=1)
        self.update_prompt_text = scrolledtext.ScrolledText(update_box, wrap="word", font=("Menlo", 11))
        self.update_prompt_text.grid(row=0, column=0, sticky="nsew")
        self.update_prompt_text.insert("1.0", PROMPT_UPDATE)

        output_frame = ttk.Frame(self.root, padding=10)
        output_frame.grid(row=4, column=0, sticky="nsew")
        output_frame.columnconfigure(0, weight=5, minsize=920)
        output_frame.columnconfigure(1, weight=4, minsize=720)
        output_frame.rowconfigure(0, weight=1)

        detail_tabs = ttk.Notebook(output_frame)
        detail_tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        latest_input_box = ttk.Frame(detail_tabs)
        latest_input_box.columnconfigure(0, weight=1)
        latest_input_box.rowconfigure(0, weight=0, minsize=PLAYBACK_ROW_MIN_HEIGHT)
        latest_input_box.rowconfigure(1, weight=1)
        detail_tabs.add(latest_input_box, text="Latest Input")

        playback_row = ttk.Frame(latest_input_box)
        playback_row.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        playback_row.columnconfigure(0, weight=1)
        playback_row.columnconfigure(1, weight=1)
        playback_row.rowconfigure(0, weight=1)

        source_box = ttk.LabelFrame(playback_row, text="Source Video Playback")
        source_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        source_box.columnconfigure(0, weight=1)
        source_box.rowconfigure(0, weight=1)
        self.source_video_label = tk.Label(
            source_box,
            text="(source playback idle)",
            bg="black",
            fg="white",
            width=56,
            height=16,
            relief="sunken",
        )
        self.source_video_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(source_box, textvariable=self.source_playback_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

        clip_box = ttk.LabelFrame(playback_row, text="Selected Input Clip Playback")
        clip_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        clip_box.columnconfigure(0, weight=1)
        clip_box.rowconfigure(0, weight=1)
        self.clip_video_label = tk.Label(
            clip_box,
            text="(select a history item)",
            bg="black",
            fg="white",
            width=56,
            height=16,
            relief="sunken",
        )
        self.clip_video_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(clip_box, textvariable=self.clip_playback_var).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.latest_input_text = scrolledtext.ScrolledText(latest_input_box, wrap="word", font=("Menlo", 11))
        self.latest_input_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        latest_response_box = ttk.Frame(detail_tabs)
        latest_response_box.columnconfigure(0, weight=1)
        latest_response_box.rowconfigure(0, weight=1)
        detail_tabs.add(latest_response_box, text="Latest Response")
        self.latest_response_text = scrolledtext.ScrolledText(latest_response_box, wrap="word", font=("Menlo", 11))
        self.latest_response_text.grid(row=0, column=0, sticky="nsew")

        side_panel = ttk.Frame(output_frame)
        side_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(0, weight=1)
        side_panel.rowconfigure(1, weight=2)

        side_tabs = ttk.Notebook(side_panel)
        side_tabs.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        history_tab = ttk.Frame(side_tabs)
        history_tab.columnconfigure(0, weight=1)
        history_tab.rowconfigure(0, weight=1)
        side_tabs.add(history_tab, text="Responses")

        history_box = ttk.LabelFrame(history_tab, text="Response History")
        history_box.grid(row=0, column=0, sticky="nsew")
        history_box.columnconfigure(0, weight=1)
        history_box.rowconfigure(0, weight=1)
        self.history_listbox = tk.Listbox(history_box, font=("Menlo", 10), exportselection=False)
        self.history_listbox.grid(row=0, column=0, sticky="nsew")
        self.history_listbox.bind("<<ListboxSelect>>", self._on_history_select)

        history_actions = ttk.Frame(history_tab, padding=(0, 6, 0, 6))
        history_actions.grid(row=1, column=0, sticky="ew")
        history_actions.columnconfigure(0, weight=0)
        history_actions.columnconfigure(1, weight=0)
        history_actions.columnconfigure(2, weight=1)
        ttk.Button(history_actions, text="Open Selected Clip", command=self._open_selected_clip).grid(row=0, column=0, sticky="w")
        ttk.Button(history_actions, text="Open Source Video", command=self._open_selected_source).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            history_actions,
            textvariable=self.selected_clip_var,
            anchor="w",
            justify="left",
            wraplength=520,
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(
            history_actions,
            textvariable=self.selected_source_var,
            anchor="w",
            justify="left",
            wraplength=520,
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        log_tab = ttk.Frame(side_tabs)
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        side_tabs.add(log_tab, text="Timeline Log")

        log_box = ttk.LabelFrame(log_tab, text="Timeline Log")
        log_box.grid(row=0, column=0, sticky="nsew")
        log_box.rowconfigure(0, weight=1)
        log_box.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_box, wrap="word", font=("Menlo", 11))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        timeline_box = ttk.LabelFrame(side_panel, text="Request Timeline")
        timeline_box.grid(row=1, column=0, sticky="nsew")
        timeline_box.columnconfigure(0, weight=1)
        timeline_box.rowconfigure(0, weight=1)
        timeline_canvas_frame = ttk.Frame(timeline_box)
        timeline_canvas_frame.grid(row=0, column=0, sticky="nsew")
        timeline_canvas_frame.columnconfigure(0, weight=1)
        timeline_canvas_frame.rowconfigure(0, weight=1)
        timeline_canvas_frame.rowconfigure(1, weight=0)
        self.timeline_canvas = tk.Canvas(
            timeline_canvas_frame,
            background="#111111",
            highlightthickness=0,
            height=380,
        )
        self.timeline_canvas.grid(row=0, column=0, sticky="nsew")
        self.timeline_canvas.bind("<Configure>", self._on_timeline_configure)
        self.timeline_canvas.bind("<MouseWheel>", self._on_timeline_mousewheel)
        self.timeline_canvas.bind("<Shift-MouseWheel>", self._on_timeline_shift_mousewheel)
        self.timeline_canvas.bind("<Button-4>", self._on_timeline_scroll_up)
        self.timeline_canvas.bind("<Button-5>", self._on_timeline_scroll_down)
        self.timeline_canvas.bind("<Shift-Button-4>", self._on_timeline_scroll_left)
        self.timeline_canvas.bind("<Shift-Button-5>", self._on_timeline_scroll_right)
        self.timeline_vscroll = ttk.Scrollbar(timeline_canvas_frame, orient="vertical", command=self.timeline_canvas.yview)
        self.timeline_vscroll.grid(row=0, column=1, sticky="ns")
        self.timeline_hscroll = ttk.Scrollbar(timeline_canvas_frame, orient="horizontal", command=self.timeline_canvas.xview)
        self.timeline_hscroll.grid(row=1, column=0, sticky="ew")
        self.timeline_canvas.configure(yscrollcommand=self.timeline_vscroll.set, xscrollcommand=self.timeline_hscroll.set)
        ttk.Label(
            timeline_box,
            textvariable=self.timeline_detail_var,
            padding=(2, 6, 2, 0),
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="ew")

    def _install_clipboard_shortcuts(self) -> None:
        text_classes = ("Text",)
        entry_classes = ("Entry", "TEntry", "TCombobox")

        for class_name in text_classes:
            self.root.bind_class(class_name, "<Command-c>", lambda e: self._event_generate_and_break(e, "<<Copy>>"))
            self.root.bind_class(class_name, "<Command-v>", lambda e: self._event_generate_and_break(e, "<<Paste>>"))
            self.root.bind_class(class_name, "<Command-x>", lambda e: self._event_generate_and_break(e, "<<Cut>>"))
            self.root.bind_class(class_name, "<Command-a>", self._select_all_text)
            self.root.bind_class(class_name, "<Control-c>", lambda e: self._event_generate_and_break(e, "<<Copy>>"))
            self.root.bind_class(class_name, "<Control-v>", lambda e: self._event_generate_and_break(e, "<<Paste>>"))
            self.root.bind_class(class_name, "<Control-x>", lambda e: self._event_generate_and_break(e, "<<Cut>>"))
            self.root.bind_class(class_name, "<Control-a>", self._select_all_text)

        for class_name in entry_classes:
            self.root.bind_class(class_name, "<Command-c>", lambda e: self._event_generate_and_break(e, "<<Copy>>"))
            self.root.bind_class(class_name, "<Command-v>", lambda e: self._event_generate_and_break(e, "<<Paste>>"))
            self.root.bind_class(class_name, "<Command-x>", lambda e: self._event_generate_and_break(e, "<<Cut>>"))
            self.root.bind_class(class_name, "<Command-a>", self._select_all_entry)
            self.root.bind_class(class_name, "<Control-c>", lambda e: self._event_generate_and_break(e, "<<Copy>>"))
            self.root.bind_class(class_name, "<Control-v>", lambda e: self._event_generate_and_break(e, "<<Paste>>"))
            self.root.bind_class(class_name, "<Control-x>", lambda e: self._event_generate_and_break(e, "<<Cut>>"))
            self.root.bind_class(class_name, "<Control-a>", self._select_all_entry)

        for class_name in text_classes + entry_classes:
            self.root.bind_class(class_name, "<Button-2>", self._show_context_menu)
            self.root.bind_class(class_name, "<Button-3>", self._show_context_menu)
            self.root.bind_class(class_name, "<Control-Button-1>", self._show_context_menu)

        global_shortcuts = [
            ("<Command-a>", self._handle_select_all_shortcut),
            ("<Command-A>", self._handle_select_all_shortcut),
            ("<Command-KeyPress-a>", self._handle_select_all_shortcut),
            ("<Command-KeyPress-A>", self._handle_select_all_shortcut),
            ("<Control-a>", self._handle_select_all_shortcut),
            ("<Control-A>", self._handle_select_all_shortcut),
            ("<Control-KeyPress-a>", self._handle_select_all_shortcut),
            ("<Control-KeyPress-A>", self._handle_select_all_shortcut),
            ("<Command-c>", self._handle_copy_shortcut),
            ("<Command-C>", self._handle_copy_shortcut),
            ("<Command-KeyPress-c>", self._handle_copy_shortcut),
            ("<Command-KeyPress-C>", self._handle_copy_shortcut),
            ("<Command-v>", self._handle_paste_shortcut),
            ("<Command-V>", self._handle_paste_shortcut),
            ("<Command-KeyPress-v>", self._handle_paste_shortcut),
            ("<Command-KeyPress-V>", self._handle_paste_shortcut),
            ("<Command-x>", self._handle_cut_shortcut),
            ("<Command-X>", self._handle_cut_shortcut),
            ("<Command-KeyPress-x>", self._handle_cut_shortcut),
            ("<Command-KeyPress-X>", self._handle_cut_shortcut),
        ]
        for sequence, handler in global_shortcuts:
            self.root.bind_all(sequence, handler, add="+")

    @staticmethod
    def _event_generate_and_break(event: Any, sequence: str) -> str:
        event.widget.event_generate(sequence)
        return "break"

    @staticmethod
    def _widget_supports_text_ops(widget: Any) -> bool:
        try:
            return widget.winfo_class() in {"Text", "Entry", "TEntry", "TCombobox"}
        except Exception:
            return False

    def _handle_copy_shortcut(self, event: Any) -> str | None:
        if self._widget_supports_text_ops(event.widget):
            return self._event_generate_and_break(event, "<<Copy>>")
        return None

    def _handle_paste_shortcut(self, event: Any) -> str | None:
        if self._widget_supports_text_ops(event.widget):
            return self._event_generate_and_break(event, "<<Paste>>")
        return None

    def _handle_cut_shortcut(self, event: Any) -> str | None:
        if self._widget_supports_text_ops(event.widget):
            return self._event_generate_and_break(event, "<<Cut>>")
        return None

    def _handle_select_all_shortcut(self, event: Any) -> str | None:
        if not self._widget_supports_text_ops(event.widget):
            return None
        if isinstance(event.widget, tk.Text):
            return self._select_all_text(event)
        return self._select_all_entry(event)

    @staticmethod
    def _select_all_text(event: Any) -> str:
        widget = event.widget
        widget.tag_add("sel", "1.0", "end-1c")
        widget.mark_set("insert", "1.0")
        widget.see("insert")
        return "break"

    @staticmethod
    def _select_all_entry(event: Any) -> str:
        widget = event.widget
        widget.selection_range(0, "end")
        widget.icursor("end")
        return "break"

    def _show_context_menu(self, event: Any) -> str:
        widget = event.widget
        try:
            widget.focus_set()
        except Exception:
            pass

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_separator()

        if isinstance(widget, tk.Text):
            menu.add_command(
                label="Select All",
                command=lambda: (
                    widget.tag_add("sel", "1.0", "end-1c"),
                    widget.mark_set("insert", "1.0"),
                    widget.see("insert"),
                ),
            )
        else:
            menu.add_command(
                label="Select All",
                command=lambda: (
                    widget.selection_range(0, "end"),
                    widget.icursor("end"),
                ),
            )

        self.context_menu = menu
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    @staticmethod
    def _frame_to_photo(frame: Any, size: tuple[int, int]) -> ImageTk.PhotoImage:
        h, w = frame.shape[:2]
        target_w, target_h = size
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        resized = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)

    def _stop_source_video_playback(self) -> None:
        if self.source_video_job is not None:
            self.root.after_cancel(self.source_video_job)
            self.source_video_job = None
        if self.source_video_cap is not None:
            self.source_video_cap.release()
            self.source_video_cap = None
        self.source_video_label.configure(image="", text="(source playback idle)")
        self.source_video_photo_ref = None
        self.source_video_path = None
        self.source_video_start_wall = None
        self.source_video_current_idx = -1
        self.source_playback_var.set("Source playback: idle")
        self._update_timeline_cursor(0.0)

    def _start_source_video_playback(self, video_path: str, duration_sec: float) -> None:
        self._stop_source_video_playback()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.source_video_label.configure(text="(failed to open source video)")
            return
        self.source_video_cap = cap
        self.source_video_path = video_path
        self.source_video_duration_sec = duration_sec
        self.source_video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.source_video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_video_start_wall = time.monotonic()
        self.source_video_current_idx = -1
        self.source_playback_var.set(f"Source playback: {os.path.basename(video_path)}")
        self._set_timeline_video_duration(duration_sec)
        self._tick_source_video_playback()

    def _tick_source_video_playback(self) -> None:
        if self.source_video_cap is None or self.source_video_start_wall is None:
            return
        try:
            elapsed = max(0.0, time.monotonic() - self.source_video_start_wall)
            target_idx = min(
                max(0, self.source_video_total_frames - 1),
                int(elapsed * max(self.source_video_fps, 1.0)),
            )
            if target_idx < 0:
                target_idx = 0
            self.source_video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = self.source_video_cap.read()
            if ret:
                self.source_video_current_idx = target_idx
                photo = self._frame_to_photo(frame, (500, 300))
                self.source_video_photo_ref = photo
                self.source_video_label.configure(image=photo, text="")
                self.source_playback_var.set(
                    f"Source playback: {os.path.basename(self.source_video_path or '')}  "
                    f"t={elapsed:.1f}/{self.source_video_duration_sec:.1f}s"
                )
                self._update_timeline_cursor(min(elapsed, self.source_video_duration_sec))
            else:
                self.source_playback_var.set("Source playback: ended")
                self._update_timeline_cursor(self.source_video_duration_sec)
                return

            if elapsed < self.source_video_duration_sec:
                delay_ms = max(15, int(1000.0 / max(self.source_video_fps, 1.0)))
                self.source_video_job = self.root.after(delay_ms, self._tick_source_video_playback)
            else:
                self.source_playback_var.set(
                    f"Source playback: ended at {self.source_video_duration_sec:.1f}s"
                )
                self._update_timeline_cursor(self.source_video_duration_sec)
                self.source_video_job = None
        except Exception as exc:
            self.source_playback_var.set(f"Source playback error: {exc}")
            self.source_video_job = None

    def _stop_clip_video_playback(self) -> None:
        if self.clip_video_job is not None:
            self.root.after_cancel(self.clip_video_job)
            self.clip_video_job = None
        if self.clip_video_cap is not None:
            self.clip_video_cap.release()
            self.clip_video_cap = None
        self.clip_video_label.configure(image="", text="(select a history item)")
        self.clip_video_photo_ref = None
        self.clip_video_path = None
        self.clip_playback_var.set("Clip playback: idle")

    def _start_clip_video_playback(self, clip_path: str | None) -> None:
        self._stop_clip_video_playback()
        if not clip_path or not os.path.exists(clip_path):
            self.clip_video_label.configure(text="(saved input clip not found)")
            self.clip_playback_var.set("Clip playback: saved input clip not found")
            return
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            self.clip_video_label.configure(text="(failed to open saved input clip)")
            self.clip_playback_var.set("Clip playback: failed to open saved input clip")
            return
        self.clip_video_cap = cap
        self.clip_video_path = clip_path
        self.clip_video_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.clip_playback_var.set(f"Clip playback: {os.path.basename(clip_path)}")
        self._tick_clip_video_playback()

    def _tick_clip_video_playback(self) -> None:
        if self.clip_video_cap is None:
            return
        try:
            ret, frame = self.clip_video_cap.read()
            if not ret:
                self.clip_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.clip_video_cap.read()
                if not ret:
                    self.clip_playback_var.set("Clip playback: unable to decode clip")
                    return
            photo = self._frame_to_photo(frame, (500, 300))
            self.clip_video_photo_ref = photo
            self.clip_video_label.configure(image=photo, text="")
            self.clip_playback_var.set(
                f"Clip playback: {os.path.basename(self.clip_video_path or '')}"
            )
            delay_ms = max(20, int(1000.0 / max(self.clip_video_fps, 1.0)))
            self.clip_video_job = self.root.after(delay_ms, self._tick_clip_video_playback)
        except Exception as exc:
            self.clip_playback_var.set(f"Clip playback error: {exc}")
            self.clip_video_job = None

    def _reset_timeline(self) -> None:
        self.timeline_video_duration_sec = 0.0
        self.timeline_current_runtime_sec = 0.0
        self.timeline_last_cursor_sec = -1.0
        self.timeline_rows = []
        self.timeline_index_by_request_id = {}
        self.timeline_selected_request_id = None
        self.timeline_detail_var.set(
            "Timeline: blue=input clip, orange=dispatch, green=arrival, red dashed=video end"
        )
        self.timeline_canvas.delete("all")
        self.timeline_canvas.configure(scrollregion=(0, 0, 1, 1))

    def _on_timeline_configure(self, _event: Any) -> None:
        self._draw_timeline()

    @staticmethod
    def _wheel_units_from_event(event: Any) -> int:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return 0
        if sys.platform == "darwin":
            return -1 if delta > 0 else 1
        steps = int(delta / 120) if delta else 0
        if steps == 0:
            return -1 if delta > 0 else 1
        return -steps

    def _on_timeline_mousewheel(self, event: Any) -> str:
        units = self._wheel_units_from_event(event)
        if units:
            self.timeline_canvas.yview_scroll(units, "units")
        return "break"

    def _on_timeline_shift_mousewheel(self, event: Any) -> str:
        units = self._wheel_units_from_event(event)
        if units:
            self.timeline_canvas.xview_scroll(units, "units")
        return "break"

    def _on_timeline_scroll_up(self, _event: Any) -> str:
        self.timeline_canvas.yview_scroll(-1, "units")
        return "break"

    def _on_timeline_scroll_down(self, _event: Any) -> str:
        self.timeline_canvas.yview_scroll(1, "units")
        return "break"

    def _on_timeline_scroll_left(self, _event: Any) -> str:
        self.timeline_canvas.xview_scroll(-1, "units")
        return "break"

    def _on_timeline_scroll_right(self, _event: Any) -> str:
        self.timeline_canvas.xview_scroll(1, "units")
        return "break"

    def _set_timeline_video_duration(self, duration_sec: float) -> None:
        self.timeline_video_duration_sec = max(0.0, float(duration_sec))
        self._draw_timeline()

    def _update_timeline_cursor(self, runtime_sec: float) -> None:
        runtime_sec = max(0.0, float(runtime_sec))
        self.timeline_current_runtime_sec = runtime_sec
        if self.timeline_last_cursor_sec < 0 or abs(runtime_sec - self.timeline_last_cursor_sec) >= TIMELINE_CURSOR_REFRESH_SEC:
            self.timeline_last_cursor_sec = runtime_sec
            self._draw_timeline()

    def _handle_dispatch(self, payload: dict[str, Any]) -> None:
        request_id = payload["request_id"]
        row = {
            "request_id": request_id,
            "kind": payload["kind"],
            "step": payload["step"],
            "dispatch_time": float(payload["dispatch_time"]),
            "clip_start_sec": float(payload["clip_start_sec"]),
            "clip_end_sec": float(payload["clip_end_sec"]),
            "arrival_time": None,
            "action_text": "(pending)",
            "latency": None,
            "status": "pending",
        }
        idx = self.timeline_index_by_request_id.get(request_id)
        if idx is None:
            self.timeline_index_by_request_id[request_id] = len(self.timeline_rows)
            self.timeline_rows.append(row)
        else:
            self.timeline_rows[idx].update(row)

        self.timeline_selected_request_id = request_id
        self.timeline_detail_var.set(
            f"Pending [{payload['step']:02d}] {payload['kind'].upper()}  "
            f"clip={payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s  "
            f"dispatch={payload['dispatch_time']:.1f}s"
        )
        self._draw_timeline()

    def _update_timeline_response(self, payload: dict[str, Any]) -> None:
        request_id = payload["request_id"]
        idx = self.timeline_index_by_request_id.get(request_id)
        if idx is None:
            self.timeline_index_by_request_id[request_id] = len(self.timeline_rows)
            self.timeline_rows.append(
                {
                    "request_id": request_id,
                    "kind": payload["kind"],
                    "step": payload["step"],
                    "dispatch_time": float(payload["dispatch_time"]),
                    "clip_start_sec": float(payload["clip_start_sec"]),
                    "clip_end_sec": float(payload["clip_end_sec"]),
                    "arrival_time": float(payload["arrival_time"]),
                    "action_text": payload["action_text"],
                    "latency": float(payload["latency"]),
                    "status": "done",
                }
            )
        else:
            self.timeline_rows[idx].update(
                {
                    "arrival_time": float(payload["arrival_time"]),
                    "action_text": payload["action_text"],
                    "latency": float(payload["latency"]),
                    "status": "done",
                }
            )
        self.timeline_selected_request_id = request_id
        self.timeline_detail_var.set(self._format_timeline_detail(payload))
        self._draw_timeline()

    @staticmethod
    def _timeline_tick_step(axis_end: float) -> float:
        if axis_end <= 10:
            return 1.0
        if axis_end <= 30:
            return 2.0
        if axis_end <= 60:
            return 5.0
        if axis_end <= 120:
            return 10.0
        return 20.0

    @staticmethod
    def _format_timeline_detail(payload: dict[str, Any]) -> str:
        arrival = payload.get("arrival_time")
        response_mode = payload.get("response_mode", RESPONSE_MODE_QUEUE)
        delivery_mode = payload.get("delivery_mode", "realtime")
        if arrival is None:
            return (
                f"Pending [{payload['step']:02d}] {payload['kind'].upper()}  "
                f"clip={payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s  "
                f"dispatch={payload['dispatch_time']:.1f}s"
            )
        label = "summary" if response_mode == RESPONSE_MODE_FREEFORM else "action"
        return (
            f"[{payload['step']:02d}] {payload['kind'].upper()}  "
            f"clip={payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s  "
            f"dispatch={payload['dispatch_time']:.1f}s  "
            f"arrival={arrival:.1f}s  "
            f"wait={arrival - payload['dispatch_time']:.1f}s  "
            f"delivery={delivery_mode}  "
            f"{label}={payload.get('action_text', '(pending)')}"
        )

    def _select_timeline_request(self, request_id: str) -> None:
        self.timeline_selected_request_id = request_id
        history_idx = self.history_index_by_request_id.get(request_id)
        if history_idx is not None:
            self.history_listbox.selection_clear(0, "end")
            self.history_listbox.selection_set(history_idx)
            self.history_listbox.see(history_idx)
            self._render_payload(self.history_payloads[history_idx])
            self.status_var.set(
                f"Viewing history item {history_idx + 1}/{len(self.history_payloads)}  "
                f"(dispatch={self.history_payloads[history_idx]['dispatch_time']:.1f}s)"
            )
        else:
            idx = self.timeline_index_by_request_id.get(request_id)
            if idx is not None:
                self.timeline_detail_var.set(self._format_timeline_detail(self.timeline_rows[idx]))
                self.status_var.set(
                    f"Pending request [{self.timeline_rows[idx]['step']:02d}] "
                    f"{self.timeline_rows[idx]['kind'].upper()}"
                )
        self._draw_timeline()

    def _draw_timeline(self) -> None:
        canvas = self.timeline_canvas
        canvas.delete("all")

        canvas_w = max(640, int(canvas.winfo_width() or 640))
        rows = self.timeline_rows
        content_h = TIMELINE_TOP_PAD + TIMELINE_BOTTOM_PAD + max(1, len(rows)) * TIMELINE_ROW_HEIGHT
        axis_end = max(
            self.timeline_video_duration_sec,
            self.timeline_current_runtime_sec,
            max(
                [
                    max(
                        float(row.get("arrival_time") or 0.0),
                        float(row.get("dispatch_time") or 0.0),
                        float(row.get("clip_end_sec") or 0.0),
                    )
                    for row in rows
                ]
                or [1.0]
            ),
            1.0,
        )
        x0 = TIMELINE_LEFT_PAD
        content_w = canvas_w
        x1 = max(x0 + 40, content_w - TIMELINE_RIGHT_PAD)
        axis_w = max(1.0, float(x1 - x0))
        axis_y = TIMELINE_TOP_PAD - 8
        canvas.configure(scrollregion=(0, 0, content_w, content_h))

        def time_to_x(time_sec: float) -> float:
            return x0 + (max(0.0, min(axis_end, float(time_sec))) / axis_end) * axis_w

        canvas.create_rectangle(0, 0, content_w, content_h, fill="#111111", outline="")
        video_end_x = time_to_x(self.timeline_video_duration_sec) if self.timeline_video_duration_sec > 0 else x0
        canvas.create_rectangle(x0, 6, video_end_x, content_h - TIMELINE_BOTTOM_PAD + 10, fill="#171f28", outline="")
        if video_end_x < x1:
            canvas.create_rectangle(video_end_x, 6, x1, content_h - TIMELINE_BOTTOM_PAD + 10, fill="#251919", outline="")
        canvas.create_line(x0, axis_y, x1, axis_y, fill="#cfcfcf", width=1)

        tick_step = self._timeline_tick_step(axis_end)
        tick = 0.0
        while tick <= axis_end + 1e-6:
            x = time_to_x(tick)
            canvas.create_line(x, axis_y - 4, x, content_h - TIMELINE_BOTTOM_PAD + 4, fill="#2e2e2e", width=1)
            canvas.create_text(x, 8, text=f"{tick:.0f}s", fill="#bdbdbd", anchor="n", font=("Menlo", 9))
            tick += tick_step

        if self.timeline_video_duration_sec > 0:
            vx = time_to_x(self.timeline_video_duration_sec)
            canvas.create_line(vx, 4, vx, content_h - TIMELINE_BOTTOM_PAD + 8, fill="#ff6b6b", dash=(6, 4), width=2)
            canvas.create_text(vx + 4, content_h - 8, text="video end", fill="#ff8f8f", anchor="sw", font=("Menlo", 9))

        if not rows:
            canvas.create_text(
                (x0 + x1) / 2.0,
                TIMELINE_TOP_PAD + 20,
                text="No requests yet. Start the simulator to plot clip / dispatch / arrival.",
                fill="#cfcfcf",
                font=("Menlo", 10),
            )

        for idx, row in enumerate(rows):
            tag = f"row_{idx}"
            y_top = TIMELINE_TOP_PAD + idx * TIMELINE_ROW_HEIGHT
            y_mid = y_top + TIMELINE_ROW_HEIGHT / 2.0
            y_bottom = y_top + TIMELINE_ROW_HEIGHT - 4
            is_selected = row["request_id"] == self.timeline_selected_request_id
            if is_selected:
                canvas.create_rectangle(
                    2,
                    y_top - 2,
                    content_w - 2,
                    y_bottom + 2,
                    fill="#263646",
                    outline="#6db4ff",
                    width=1,
                    tags=(tag,),
                )

            label = f"[{row['step']:02d}] {row['kind'].upper()}"
            canvas.create_text(8, y_mid - 9, text=label, fill="#f1f1f1", anchor="w", font=("Menlo", 10, "bold"), tags=(tag,))
            canvas.create_text(
                8,
                y_mid + 9,
                text=row.get("action_text") or "(pending)",
                fill="#9cd67d" if row.get("status") == "done" else "#e4c27a",
                anchor="w",
                font=("Menlo", 9),
                tags=(tag,),
            )

            clip_x0 = time_to_x(float(row["clip_start_sec"]))
            clip_x1 = time_to_x(float(row["clip_end_sec"]))
            dispatch_x = time_to_x(float(row["dispatch_time"]))
            arrival_time = row.get("arrival_time")
            wait_x1 = time_to_x(float(arrival_time) if arrival_time is not None else self.timeline_current_runtime_sec)

            canvas.create_rectangle(
                clip_x0,
                y_mid - 10,
                max(clip_x1, clip_x0 + 2),
                y_mid + 10,
                fill="#3b82f6",
                outline="#93c5fd",
                tags=(tag,),
            )
            canvas.create_line(dispatch_x, y_mid - 16, dispatch_x, y_mid + 16, fill="#f59e0b", width=2, tags=(tag,))
            canvas.create_line(
                dispatch_x,
                y_mid + 15,
                wait_x1,
                y_mid + 15,
                fill="#f59e0b" if arrival_time is None else "#34d399",
                width=3,
                tags=(tag,),
            )
            canvas.create_oval(dispatch_x - 4, y_mid + 11, dispatch_x + 4, y_mid + 19, fill="#f59e0b", outline="", tags=(tag,))
            if arrival_time is not None:
                arrival_x = time_to_x(float(arrival_time))
                canvas.create_line(arrival_x, y_mid - 16, arrival_x, y_mid + 16, fill="#34d399", width=2, tags=(tag,))
                canvas.create_oval(arrival_x - 4, y_mid + 11, arrival_x + 4, y_mid + 19, fill="#34d399", outline="", tags=(tag,))
                canvas.create_text(
                    min(x1 - 4, arrival_x + 6),
                    y_mid - 18,
                    text=f"{arrival_time - row['dispatch_time']:.1f}s",
                    fill="#c8facc",
                    anchor="w",
                    font=("Menlo", 9),
                    tags=(tag,),
                )
            else:
                canvas.create_text(
                    min(x1 - 4, wait_x1 + 6),
                    y_mid - 18,
                    text="pending",
                    fill="#ffd18a",
                    anchor="w",
                    font=("Menlo", 9),
                    tags=(tag,),
                )

            canvas.create_text(
                min(x1 - 4, clip_x0 + 6),
                y_mid - 23,
                text=f"{row['clip_start_sec']:.1f}-{row['clip_end_sec']:.1f}s",
                fill="#b7dcff",
                anchor="w",
                font=("Menlo", 9),
                tags=(tag,),
            )
            canvas.tag_bind(tag, "<Button-1>", lambda _event, rid=row["request_id"]: self._select_timeline_request(rid))

        if self.timeline_current_runtime_sec > 0:
            cursor_x = time_to_x(self.timeline_current_runtime_sec)
            canvas.create_line(cursor_x, 0, cursor_x, content_h, fill="#f8fafc", width=2)
            canvas.create_text(
                min(x1 - 4, cursor_x + 6),
                content_h - TIMELINE_BOTTOM_PAD + 10,
                text=f"now {self.timeline_current_runtime_sec:.1f}s",
                fill="#f8fafc",
                anchor="w",
                font=("Menlo", 9, "bold"),
            )

    def _load_default_episode(self) -> None:
        dataset_dir = self.dataset_dir_var.get().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            self.episode_names = []
            self.episode_combo.configure(values=self.episode_names)
            self.episode_var.set("")
            return
        names = list_episode_names(dataset_dir)
        self.episode_names = names
        self.episode_combo.configure(values=self.episode_names)

        current = self.episode_var.get().strip()
        if current in self.episode_names:
            self.episode_var.set(current)
        elif self.episode_names:
            self.episode_var.set(self.episode_names[0])
        else:
            self.episode_var.set("")

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")

    def _set_latest_response(self, text: str) -> None:
        self.latest_response_text.delete("1.0", "end")
        self.latest_response_text.insert("1.0", text.rstrip() + "\n")
        self.latest_response_text.see("1.0")

    def _set_latest_input(self, text: str) -> None:
        self.latest_input_text.delete("1.0", "end")
        self.latest_input_text.insert("1.0", text.rstrip() + "\n")
        self.latest_input_text.see("1.0")

    def _get_selected_history_index(self) -> int | None:
        sel = self.history_listbox.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if 0 <= idx < len(self.history_payloads):
            return idx
        return None

    def _format_history_label(self, payload: dict[str, Any]) -> str:
        summary = payload["action_text"]
        if len(summary) > 54:
            summary = summary[:51].rstrip() + "..."
        delivery_marker = " [prefetch]" if payload.get("delivery_mode") == "precomputed" else ""
        return (
            f"[{payload['step']:02d}] {payload['kind'].upper()}  "
            f"d={payload['dispatch_time']:.1f}s  a={payload['arrival_time']:.1f}s  "
            f"{summary}{delivery_marker}"
        )

    def _render_payload(self, payload: dict[str, Any]) -> None:
        self.timeline_selected_request_id = payload.get("request_id")
        self.timeline_detail_var.set(self._format_timeline_detail(payload))
        response_mode = payload.get("response_mode", RESPONSE_MODE_QUEUE)
        latest_input = (
            f"[{payload['kind'].upper()} step={payload['step']}]\n"
            f"mode:     {response_mode}\n"
            f"delivery: {payload.get('delivery_mode', 'realtime')}\n"
            f"dispatch: {payload['dispatch_time']:.1f}s\n"
            f"arrival:  {payload['arrival_time']:.1f}s\n"
            f"wall wait: {payload['arrival_time'] - payload['dispatch_time']:.1f}s\n"
            f"clip:     {payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s\n"
            f"source:   {payload['video_path']}\n"
            f"audio:    {payload['audio_mode']}\n"
            f"size:     {payload['request_size_kb']:.1f}KB\n"
            f"cost:     ${payload['cost_usd']:.4f}\n"
            f"clipfile: {payload.get('clip_path') or '(not saved)'}\n"
            f"stages:   {payload['stage_summary']}\n\n"
        )
        if response_mode == RESPONSE_MODE_QUEUE:
            latest_input += f"Queue Before:\n{payload['queue_before_text']}\n\n"
        latest_input += f"Prompt Sent:\n{payload['prompt']}"
        self._set_latest_input(latest_input)
        self._start_clip_video_playback(payload.get("clip_path"))

        latest_response = (
            f"[{payload['kind'].upper()} step={payload['step']}]\n"
            f"mode:     {response_mode}\n"
            f"delivery: {payload.get('delivery_mode', 'realtime')}\n"
            f"dispatch: {payload['dispatch_time']:.1f}s\n"
            f"arrival:  {payload['arrival_time']:.1f}s\n"
            f"wall wait: {payload['arrival_time'] - payload['dispatch_time']:.1f}s\n"
            f"clip:     {payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s\n"
        )
        if response_mode == RESPONSE_MODE_QUEUE:
            latest_response += f"action:   {payload['action_text']}\n"
        else:
            latest_response += f"summary:  {payload['action_text']}\n"
        latest_response += (
            f"gemini:   {payload['latency']:.1f}s\n"
            f"worker:   {payload['total_worker_sec']:.1f}s\n"
            f"cost:     ${payload['cost_usd']:.4f}\n"
            f"stages:   {payload['stage_summary']}\n\n"
        )
        if response_mode == RESPONSE_MODE_QUEUE:
            latest_response += f"Queue After:\n{payload['queue_after_text']}\n\n"
        latest_response += f"Raw Response:\n{payload['raw_response']}"
        self._set_latest_response(latest_response)
        clip_path = payload.get("clip_path") or "(not saved)"
        self.selected_clip_var.set(f"Clip: {_compact_path(clip_path)}")
        self.selected_source_var.set(f"Source: {_compact_path(payload.get('video_path'))}")
        self._draw_timeline()

    def _on_history_select(self, _event: Any) -> None:
        idx = self._get_selected_history_index()
        if idx is None:
            return
        payload = self.history_payloads[idx]
        self._render_payload(payload)
        self.status_var.set(
            f"Viewing history item {idx + 1}/{len(self.history_payloads)}  "
            f"(dispatch={payload['dispatch_time']:.1f}s)"
        )

    def _open_selected_clip(self) -> None:
        idx = self._get_selected_history_index()
        if idx is None:
            messagebox.showinfo("Open Clip", "Select a history item first.")
            return
        clip_path = self.history_payloads[idx].get("clip_path")
        if not clip_path or not os.path.exists(clip_path):
            messagebox.showerror("Open Clip", "Saved clip file was not found.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", clip_path])
            elif os.name == "nt":
                os.startfile(clip_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", clip_path])
        except Exception as exc:
            messagebox.showerror("Open Clip", str(exc))

    def _open_selected_source(self) -> None:
        idx = self._get_selected_history_index()
        if idx is None:
            messagebox.showinfo("Open Source Video", "Select a history item first.")
            return
        video_path = self.history_payloads[idx].get("video_path")
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Open Source Video", "Source video file was not found.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", video_path])
            elif os.name == "nt":
                os.startfile(video_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", video_path])
        except Exception as exc:
            messagebox.showerror("Open Source Video", str(exc))

    def _request_worker_emit_response(
        self,
        *,
        api_key: str,
        config: SimulationConfig,
        video_path: str,
        audio_path: str | None,
        start_wall: float,
        request_id: str,
        kind: str,
        step: int,
        dispatch_time: float,
        clip_start_sec: float,
        clip_end_sec: float,
        queue_status: str,
        queue_before_text: str,
        clip_path: str | None,
    ) -> None:
        request_t0 = time.perf_counter()
        try:
            client = genai.Client(api_key=api_key)
            result = _run_request(
                client=client,
                config=config,
                video_path=video_path,
                audio_path=audio_path,
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                kind=kind,
                queue_status=queue_status,
                save_clip_path=clip_path,
            )
            arrival_time = time.monotonic() - start_wall
            parsed = result["parsed"]
            if config.response_mode == RESPONSE_MODE_QUEUE:
                action_text = "ParseFailure"
                queue_after_text = queue_before_text
                if kind == "init" and parsed is not None:
                    elements = [KeyframeQueueElement.from_dict(item) for item in parsed["action_queue"]]
                    action_text = f"Initialize ({len(elements)} elements)"
                    temp_queue = KeyframeQueue()
                    temp_queue.initialize(elements)
                    queue_after_text = _format_queue_snapshot(temp_queue)
                elif kind == "update" and parsed is not None:
                    action_text = parsed.get("action", "ParseFailure")
                    queue_after_text = "(async queue mode not supported)"
            else:
                action_text = _summarize_freeform_response(result["raw_response"])
                queue_after_text = FREEFORM_QUEUE_DISABLED

            stage = result["stage_times"]
            payload = {
                "request_id": request_id,
                "kind": kind,
                "step": step,
                "response_mode": config.response_mode,
                "delivery_mode": "realtime",
                "dispatch_time": round(dispatch_time, 3),
                "arrival_time": round(arrival_time, 3),
                "clip_start_sec": round(clip_start_sec, 3),
                "clip_end_sec": round(clip_end_sec, 3),
                "action_text": action_text,
                "latency": result["latency"],
                "total_worker_sec": result["total_worker_sec"],
                "request_size_kb": result["request_size_kb"],
                "cost_usd": float((result["cost"] or {}).get("cost_usd", 0.0)),
                "prompt": result["prompt"],
                "raw_response": result["raw_response"],
                "clip_path": result.get("clip_path"),
                "video_path": video_path,
                "audio_mode": result["audio_mode"],
                "queue_before_text": queue_before_text,
                "queue_after_text": queue_after_text,
                "stage_summary": (
                    f"cut={stage['clip_cut_sec']:.1f}s  "
                    f"audio={stage['audio_mux_sec']:.1f}s  "
                    f"prompt={stage['prompt_build_sec']:.1f}s  "
                    f"gemini={stage['gemini_api_sec']:.1f}s  "
                    f"parse={stage['parse_sec']:.1f}s"
                ),
            }
        except Exception as exc:
            arrival_time = time.monotonic() - start_wall
            payload = {
                "request_id": request_id,
                "kind": kind,
                "step": step,
                "response_mode": config.response_mode,
                "delivery_mode": "realtime",
                "dispatch_time": round(dispatch_time, 3),
                "arrival_time": round(arrival_time, 3),
                "clip_start_sec": round(clip_start_sec, 3),
                "clip_end_sec": round(clip_end_sec, 3),
                "action_text": "WorkerError",
                "latency": 0.0,
                "total_worker_sec": round(time.perf_counter() - request_t0, 4),
                "request_size_kb": 0.0,
                "cost_usd": 0.0,
                "prompt": "",
                "raw_response": str(exc),
                "clip_path": clip_path,
                "video_path": video_path,
                "audio_mode": AUDIO_MODE_VOICED if audio_path else "none",
                "queue_before_text": queue_before_text,
                "queue_after_text": FREEFORM_QUEUE_DISABLED if config.response_mode == RESPONSE_MODE_FREEFORM else queue_before_text,
                "stage_summary": "cut=0.0s  audio=0.0s  prompt=0.0s  gemini=0.0s  parse=0.0s",
            }
        self._emit("response", payload=payload)

    def _read_config(self) -> SimulationConfig:
        dataset_dir = self.dataset_dir_var.get().strip()
        episode = self.episode_var.get().strip()
        if not dataset_dir:
            raise ValueError("Dataset directory is empty.")
        if not episode:
            raise ValueError("Episode is empty.")

        return SimulationConfig(
            dataset_dir=dataset_dir,
            episode=episode,
            model=self.model_var.get().strip(),
            response_mode=self.response_mode_var.get().strip() or RESPONSE_MODE_FREEFORM,
            execution_mode=self.execution_mode_var.get().strip() or EXECUTION_MODE_SEQUENTIAL,
            max_inflight=max(1, int(self.max_inflight_var.get().strip())),
            context_window_sec=max(0.5, float(self.context_window_var.get().strip())),
            dispatch_hop_sec=max(0.1, float(self.dispatch_hop_var.get().strip())),
            video_fps_hint=max(1, int(self.video_fps_var.get().strip())),
            target_resolution=parse_resolution(self.target_resolution_var.get().strip()),
            use_gaze_video=bool(self.use_gaze_var.get()),
            embed_audio=bool(self.embed_audio_var.get()),
            precompute_init=bool(self.precompute_init_var.get()),
            init_prompt_template=self.init_prompt_text.get("1.0", "end").strip(),
            update_prompt_template=self.update_prompt_text.get("1.0", "end").strip(),
        )

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Simulator", "A simulation is already running.")
            return

        try:
            config = self._read_config()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc))
            return

        if config.execution_mode == EXECUTION_MODE_ASYNC and config.response_mode != RESPONSE_MODE_FREEFORM:
            messagebox.showerror(
                "Invalid Config",
                "Async execution currently supports freeform response mode only.",
            )
            return

        self.log_text.delete("1.0", "end")
        self.latest_input_text.delete("1.0", "end")
        self.latest_response_text.delete("1.0", "end")
        self.history_listbox.delete(0, "end")
        self.history_payloads = []
        self.history_index_by_request_id = {}
        self.selected_clip_var.set("Clip: (none)")
        self.selected_source_var.set("Source: (none)")
        self.source_playback_var.set("Source playback: idle")
        self.clip_playback_var.set("Clip playback: idle")
        self.status_var.set("Starting")
        self.running_var.set("Running")
        self.stop_event.clear()
        self._reset_timeline()
        self._stop_source_video_playback()
        self._stop_clip_video_playback()
        safe_episode = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in config.episode)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.abspath(os.path.join(RUN_OUTPUT_ROOT, f"{run_stamp}_{safe_episode}"))
        os.makedirs(os.path.join(self.current_run_dir, "clips"), exist_ok=True)
        self.worker_thread = threading.Thread(target=self._simulation_worker, args=(config,), daemon=True)
        self.worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.execution_mode_var.get().strip() == EXECUTION_MODE_ASYNC:
            self.status_var.set("Stopping new dispatches; waiting for in-flight requests")
        else:
            self.status_var.set("Stopping after current request")

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.event_queue.put({"type": event_type, **payload})

    def _simulation_worker(self, config: SimulationConfig) -> None:
        try:
            api_key = resolve_gemini_api_key("")
            if not api_key:
                raise RuntimeError("No Gemini API key found.")

            files = resolve_episode_files(config.dataset_dir, config.episode)
            video_path = files["rgb_with_gaze"] if config.use_gaze_video else files["rgb"]
            if not os.path.exists(video_path):
                raise RuntimeError(f"Video not found: {video_path}")

            audio_path = files["audio"] if config.embed_audio else None
            if audio_path and not os.path.exists(audio_path):
                self._emit("log", text=f"[warn] Audio file not found: {audio_path}. Continuing without audio.")
                audio_path = None

            client = genai.Client(api_key=api_key)
            total_frames, fps = get_video_meta(video_path)
            video_duration = total_frames / max(fps, 1.0)
            init_dispatch_threshold = min(config.context_window_sec, video_duration)

            queue = KeyframeQueue()
            precomputed_init_result: dict[str, Any] | None = None

            self._emit(
                "log",
                text=(
                    f"[start] episode={config.episode}  model={config.model}  mode={config.response_mode}  exec={config.execution_mode}  "
                    f"video={video_duration:.1f}s  ctx={config.context_window_sec:.1f}s  "
                    f"hop={config.dispatch_hop_sec:.1f}s  audio={bool(audio_path)}  "
                    f"max_inflight={config.max_inflight}  precompute_init={config.precompute_init}"
                ),
            )
            if self.current_run_dir:
                self._emit("log", text=f"[output] run_dir={self.current_run_dir}")

            if config.precompute_init and init_dispatch_threshold > 1e-6:
                init_clip_path = None
                if self.current_run_dir:
                    init_clip_path = os.path.join(
                        self.current_run_dir,
                        "clips",
                        f"step_{0:03d}_init_{0.0:.1f}s_{init_dispatch_threshold:.1f}s.mp4",
                    )
                init_queue_status = (
                    _format_queue_snapshot(queue)
                    if config.response_mode == RESPONSE_MODE_QUEUE
                    else FREEFORM_QUEUE_DISABLED
                )
                self._emit(
                    "status",
                    text=f"Precomputing init clip 0.0-{init_dispatch_threshold:.1f}s before playback",
                )
                self._emit(
                    "log",
                    text=f"[precompute] INIT clip=0.0-{init_dispatch_threshold:.1f}s",
                )
                try:
                    precomputed_init_result = _run_request(
                        client=client,
                        config=config,
                        video_path=video_path,
                        audio_path=audio_path,
                        clip_start_sec=0.0,
                        clip_end_sec=init_dispatch_threshold,
                        kind="init",
                        queue_status=init_queue_status,
                        save_clip_path=init_clip_path,
                    )
                    self._emit(
                        "log",
                        text=(
                            f"[precompute] INIT ready  "
                            f"gemini={precomputed_init_result['latency']:.1f}s  "
                            f"total={precomputed_init_result['total_worker_sec']:.1f}s"
                        ),
                    )
                except Exception as exc:
                    precomputed_init_result = None
                    self._emit(
                        "log",
                        text=f"[warn] Precompute init failed; falling back to runtime init. {exc}",
                    )

            self._emit("source_video", video_path=video_path, duration_sec=video_duration)

            start_wall = time.monotonic()
            next_dispatch_time = init_dispatch_threshold
            next_update_step = 1
            init_dispatched = False
            last_dispatched_clip_end_sec = -1.0

            if config.execution_mode == EXECUTION_MODE_ASYNC:
                active_threads: dict[str, threading.Thread] = {}
                while not self.stop_event.is_set():
                    active_threads = {
                        req_id: thread
                        for req_id, thread in active_threads.items()
                        if thread.is_alive()
                    }

                    runtime_time = time.monotonic() - start_wall
                    effective_runtime_time = min(runtime_time, video_duration)
                    at_video_end = effective_runtime_time >= (video_duration - 1e-6)

                    if at_video_end and last_dispatched_clip_end_sec >= (video_duration - 1e-6):
                        break

                    if len(active_threads) >= config.max_inflight:
                        self._emit(
                            "status",
                            text=(
                                f"Async mode waiting for slot  "
                                f"({len(active_threads)}/{config.max_inflight} in flight)"
                            ),
                        )
                        time.sleep(0.05)
                        continue

                    if runtime_time < next_dispatch_time and not at_video_end:
                        self._emit(
                            "status",
                            text=(
                                f"Async mode waiting for next dispatch at t={next_dispatch_time:.1f}s  "
                                f"(runtime={runtime_time:.1f}s, inflight={len(active_threads)})"
                            ),
                        )
                        time.sleep(min(0.05, next_dispatch_time - runtime_time))
                        continue

                    if not init_dispatched and precomputed_init_result is not None:
                        dispatch_time = init_dispatch_threshold
                        clip_start_sec = 0.0
                        clip_end_sec = init_dispatch_threshold
                        request_id = _make_request_id("init", 0)
                        queue_before_text = (
                            _format_queue_snapshot(queue)
                            if config.response_mode == RESPONSE_MODE_QUEUE
                            else FREEFORM_QUEUE_DISABLED
                        )
                        self._emit(
                            "log",
                            text=(
                                f"[dispatch] INIT step=0  t={dispatch_time:.1f}s  "
                                f"clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s  "
                                f"inflight=0  precomputed=1"
                            ),
                        )
                        self._emit(
                            "status",
                            text=f"Delivering precomputed init at t={dispatch_time:.1f}s",
                        )
                        self._emit(
                            "dispatch",
                            payload={
                                "request_id": request_id,
                                "kind": "init",
                                "step": 0,
                                "dispatch_time": round(dispatch_time, 3),
                                "clip_start_sec": round(clip_start_sec, 3),
                                "clip_end_sec": round(clip_end_sec, 3),
                            },
                        )

                        parsed = precomputed_init_result["parsed"]
                        if config.response_mode == RESPONSE_MODE_QUEUE:
                            action_text = "ParseFailure"
                            queue_after_text = _format_queue_snapshot(queue)
                            if parsed is not None:
                                elements = [KeyframeQueueElement.from_dict(item) for item in parsed["action_queue"]]
                                queue.initialize(elements)
                                action_text = f"Initialize ({len(elements)} elements)"
                                queue_after_text = _format_queue_snapshot(queue)
                        else:
                            action_text = _summarize_freeform_response(precomputed_init_result["raw_response"])
                            queue_after_text = FREEFORM_QUEUE_DISABLED

                        stage = precomputed_init_result["stage_times"]
                        self._emit(
                            "response",
                            payload={
                                "request_id": request_id,
                                "kind": "init",
                                "step": 0,
                                "response_mode": config.response_mode,
                                "delivery_mode": "precomputed",
                                "dispatch_time": round(dispatch_time, 3),
                                "arrival_time": round(dispatch_time, 3),
                                "clip_start_sec": round(clip_start_sec, 3),
                                "clip_end_sec": round(clip_end_sec, 3),
                                "action_text": action_text,
                                "latency": precomputed_init_result["latency"],
                                "total_worker_sec": precomputed_init_result["total_worker_sec"],
                                "request_size_kb": precomputed_init_result["request_size_kb"],
                                "cost_usd": float((precomputed_init_result["cost"] or {}).get("cost_usd", 0.0)),
                                "prompt": precomputed_init_result["prompt"],
                                "raw_response": precomputed_init_result["raw_response"],
                                "clip_path": precomputed_init_result.get("clip_path"),
                                "video_path": video_path,
                                "audio_mode": precomputed_init_result["audio_mode"],
                                "queue_before_text": queue_before_text,
                                "queue_after_text": queue_after_text,
                                "stage_summary": (
                                    f"cut={stage['clip_cut_sec']:.1f}s  "
                                    f"audio={stage['audio_mux_sec']:.1f}s  "
                                    f"prompt={stage['prompt_build_sec']:.1f}s  "
                                    f"gemini={stage['gemini_api_sec']:.1f}s  "
                                    f"parse={stage['parse_sec']:.1f}s"
                                ),
                            },
                        )

                        last_dispatched_clip_end_sec = clip_end_sec
                        next_dispatch_time = dispatch_time + config.dispatch_hop_sec
                        init_dispatched = True
                        if clip_end_sec >= (video_duration - 1e-6):
                            break
                        continue

                    kind = "init" if not init_dispatched else "update"
                    step = 0 if kind == "init" else next_update_step
                    request_id = _make_request_id(kind, step)
                    clip_end_sec = effective_runtime_time
                    if clip_end_sec <= last_dispatched_clip_end_sec + 1e-6:
                        if at_video_end:
                            break
                        time.sleep(0.05)
                        continue
                    clip_start_sec = max(0.0, clip_end_sec - config.context_window_sec)
                    queue_status = FREEFORM_QUEUE_DISABLED
                    queue_before_text = FREEFORM_QUEUE_DISABLED

                    self._emit(
                        "log",
                        text=(
                            f"[dispatch] {kind.upper()} step={step}  "
                            f"t={runtime_time:.1f}s  clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s  "
                            f"inflight={len(active_threads) + 1}"
                        ),
                    )
                    self._emit(
                        "status",
                        text=(
                            f"Dispatching async {kind} step={step}  "
                            f"clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s"
                        ),
                    )
                    self._emit(
                        "dispatch",
                        payload={
                            "request_id": request_id,
                            "kind": kind,
                            "step": step,
                            "dispatch_time": round(runtime_time, 3),
                            "clip_start_sec": round(clip_start_sec, 3),
                            "clip_end_sec": round(clip_end_sec, 3),
                        },
                    )

                    clip_path = None
                    if self.current_run_dir:
                        clip_path = os.path.join(
                            self.current_run_dir,
                            "clips",
                            f"step_{step:03d}_{kind}_{clip_start_sec:.1f}s_{clip_end_sec:.1f}s.mp4",
                        )

                    thread = threading.Thread(
                        target=self._request_worker_emit_response,
                        kwargs={
                            "api_key": api_key,
                            "config": config,
                            "video_path": video_path,
                            "audio_path": audio_path,
                            "start_wall": start_wall,
                            "request_id": request_id,
                            "kind": kind,
                            "step": step,
                            "dispatch_time": runtime_time,
                            "clip_start_sec": clip_start_sec,
                            "clip_end_sec": clip_end_sec,
                            "queue_status": queue_status,
                            "queue_before_text": queue_before_text,
                            "clip_path": clip_path,
                        },
                        daemon=True,
                    )
                    active_threads[request_id] = thread
                    thread.start()

                    last_dispatched_clip_end_sec = clip_end_sec
                    next_dispatch_time = runtime_time + config.dispatch_hop_sec
                    init_dispatched = True
                    if kind == "update":
                        next_update_step += 1

                    if clip_end_sec >= (video_duration - 1e-6):
                        break

                while active_threads:
                    active_threads = {
                        req_id: thread
                        for req_id, thread in active_threads.items()
                        if thread.is_alive()
                    }
                    if not active_threads:
                        break
                    self._emit(
                        "status",
                        text=f"Waiting for {len(active_threads)} in-flight async request(s) to finish",
                    )
                    time.sleep(0.05)

                self._emit("finished", text="Simulation finished.")
                return

            while not self.stop_event.is_set():
                runtime_time = time.monotonic() - start_wall
                effective_runtime_time = min(runtime_time, video_duration)
                at_video_end = effective_runtime_time >= (video_duration - 1e-6)

                if at_video_end and last_dispatched_clip_end_sec >= (video_duration - 1e-6):
                    break

                if runtime_time < next_dispatch_time and not at_video_end:
                    self._emit(
                        "status",
                        text=f"Waiting for next dispatch at t={next_dispatch_time:.1f}s (runtime={runtime_time:.1f}s)",
                    )
                    time.sleep(min(0.1, next_dispatch_time - runtime_time))
                    continue

                if not init_dispatched and precomputed_init_result is not None:
                    dispatch_time = init_dispatch_threshold
                    clip_start_sec = 0.0
                    clip_end_sec = init_dispatch_threshold
                    request_id = _make_request_id("init", 0)
                    if config.response_mode == RESPONSE_MODE_QUEUE:
                        queue_before_text = _format_queue_snapshot(queue)
                    else:
                        queue_before_text = FREEFORM_QUEUE_DISABLED

                    self._emit(
                        "log",
                        text=(
                            f"[dispatch] INIT step=0  "
                            f"t={dispatch_time:.1f}s  clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s  "
                            f"precomputed=1"
                        ),
                    )
                    self._emit(
                        "status",
                        text=f"Delivering precomputed init at t={dispatch_time:.1f}s",
                    )
                    self._emit(
                        "dispatch",
                        payload={
                            "request_id": request_id,
                            "kind": "init",
                            "step": 0,
                            "dispatch_time": round(dispatch_time, 3),
                            "clip_start_sec": round(clip_start_sec, 3),
                            "clip_end_sec": round(clip_end_sec, 3),
                        },
                    )

                    parsed = precomputed_init_result["parsed"]
                    if config.response_mode == RESPONSE_MODE_QUEUE:
                        action_text = "ParseFailure"
                        queue_after_text = _format_queue_snapshot(queue)
                        if parsed is not None:
                            elements = [KeyframeQueueElement.from_dict(item) for item in parsed["action_queue"]]
                            queue.initialize(elements)
                            action_text = f"Initialize ({len(elements)} elements)"
                            queue_after_text = _format_queue_snapshot(queue)
                    else:
                        action_text = _summarize_freeform_response(precomputed_init_result["raw_response"])
                        queue_after_text = FREEFORM_QUEUE_DISABLED

                    stage = precomputed_init_result["stage_times"]
                    self._emit(
                        "response",
                        payload={
                            "request_id": request_id,
                            "kind": "init",
                            "step": 0,
                            "response_mode": config.response_mode,
                            "delivery_mode": "precomputed",
                            "dispatch_time": round(dispatch_time, 3),
                            "arrival_time": round(dispatch_time, 3),
                            "clip_start_sec": round(clip_start_sec, 3),
                            "clip_end_sec": round(clip_end_sec, 3),
                            "action_text": action_text,
                            "latency": precomputed_init_result["latency"],
                            "total_worker_sec": precomputed_init_result["total_worker_sec"],
                            "request_size_kb": precomputed_init_result["request_size_kb"],
                            "cost_usd": float((precomputed_init_result["cost"] or {}).get("cost_usd", 0.0)),
                            "prompt": precomputed_init_result["prompt"],
                            "raw_response": precomputed_init_result["raw_response"],
                            "clip_path": precomputed_init_result.get("clip_path"),
                            "video_path": video_path,
                            "audio_mode": precomputed_init_result["audio_mode"],
                            "queue_before_text": queue_before_text,
                            "queue_after_text": queue_after_text,
                            "stage_summary": (
                                f"cut={stage['clip_cut_sec']:.1f}s  "
                                f"audio={stage['audio_mux_sec']:.1f}s  "
                                f"prompt={stage['prompt_build_sec']:.1f}s  "
                                f"gemini={stage['gemini_api_sec']:.1f}s  "
                                f"parse={stage['parse_sec']:.1f}s"
                            ),
                        },
                    )

                    last_dispatched_clip_end_sec = clip_end_sec
                    next_dispatch_time = dispatch_time + config.dispatch_hop_sec
                    init_dispatched = True
                    if clip_end_sec >= (video_duration - 1e-6):
                        break
                    continue

                kind = "init" if not init_dispatched else "update"
                step = 0 if kind == "init" else next_update_step
                request_id = _make_request_id(kind, step)
                clip_end_sec = effective_runtime_time
                if clip_end_sec <= last_dispatched_clip_end_sec + 1e-6:
                    if at_video_end:
                        break
                    time.sleep(0.05)
                    continue
                clip_start_sec = max(0.0, clip_end_sec - config.context_window_sec)
                if config.response_mode == RESPONSE_MODE_QUEUE:
                    queue_status = queue.format_for_prompt()
                    queue_before_text = _format_queue_snapshot(queue)
                else:
                    queue_status = FREEFORM_QUEUE_DISABLED
                    queue_before_text = FREEFORM_QUEUE_DISABLED

                self._emit(
                    "log",
                    text=(
                        f"[dispatch] {kind.upper()} step={step}  "
                        f"t={runtime_time:.1f}s  clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s"
                    ),
                )
                self._emit(
                    "status",
                    text=(
                        f"Running {kind} step={step}  "
                        f"clip={clip_start_sec:.1f}-{clip_end_sec:.1f}s"
                    ),
                )
                self._emit(
                    "dispatch",
                    payload={
                        "request_id": request_id,
                        "kind": kind,
                        "step": step,
                        "dispatch_time": round(runtime_time, 3),
                        "clip_start_sec": round(clip_start_sec, 3),
                        "clip_end_sec": round(clip_end_sec, 3),
                    },
                )

                clip_path = None
                if self.current_run_dir:
                    clip_path = os.path.join(
                        self.current_run_dir,
                        "clips",
                        f"step_{step:03d}_{kind}_{clip_start_sec:.1f}s_{clip_end_sec:.1f}s.mp4",
                    )

                result = _run_request(
                    client=client,
                    config=config,
                    video_path=video_path,
                    audio_path=audio_path,
                    clip_start_sec=clip_start_sec,
                    clip_end_sec=clip_end_sec,
                    kind=kind,
                    queue_status=queue_status,
                    save_clip_path=clip_path,
                )

                arrival_time = time.monotonic() - start_wall
                parsed = result["parsed"]
                if config.response_mode == RESPONSE_MODE_QUEUE:
                    action_text = "ParseFailure"
                    queue_after_text = _format_queue_snapshot(queue)
                    if kind == "init" and parsed is not None:
                        elements = [KeyframeQueueElement.from_dict(item) for item in parsed["action_queue"]]
                        queue.initialize(elements)
                        action_text = f"Initialize ({len(elements)} elements)"
                        queue_after_text = _format_queue_snapshot(queue)
                    elif kind == "update" and parsed is not None:
                        action_text = apply_queue_update(queue, parsed)
                        queue_after_text = _format_queue_snapshot(queue)
                else:
                    action_text = _summarize_freeform_response(result["raw_response"])
                    queue_after_text = FREEFORM_QUEUE_DISABLED

                stage = result["stage_times"]
                self._emit(
                    "response",
                    payload={
                        "request_id": request_id,
                        "kind": kind,
                        "step": step,
                        "response_mode": config.response_mode,
                        "delivery_mode": "realtime",
                        "dispatch_time": round(runtime_time, 3),
                        "arrival_time": round(arrival_time, 3),
                        "clip_start_sec": round(clip_start_sec, 3),
                        "clip_end_sec": round(clip_end_sec, 3),
                        "action_text": action_text,
                        "latency": result["latency"],
                        "total_worker_sec": result["total_worker_sec"],
                        "request_size_kb": result["request_size_kb"],
                        "cost_usd": float((result["cost"] or {}).get("cost_usd", 0.0)),
                        "prompt": result["prompt"],
                        "raw_response": result["raw_response"],
                        "clip_path": result.get("clip_path"),
                        "video_path": video_path,
                        "audio_mode": result["audio_mode"],
                        "queue_before_text": queue_before_text,
                        "queue_after_text": queue_after_text,
                        "stage_summary": (
                            f"cut={stage['clip_cut_sec']:.1f}s  "
                            f"audio={stage['audio_mux_sec']:.1f}s  "
                            f"prompt={stage['prompt_build_sec']:.1f}s  "
                            f"gemini={stage['gemini_api_sec']:.1f}s  "
                            f"parse={stage['parse_sec']:.1f}s"
                        ),
                    },
                )

                last_dispatched_clip_end_sec = clip_end_sec
                next_dispatch_time = runtime_time + max(config.dispatch_hop_sec, float(result["latency"] or 0.0))
                init_dispatched = True
                if kind == "update":
                    next_update_step += 1

                if clip_end_sec >= (video_duration - 1e-6):
                    break

            self._emit("finished", text="Simulation finished.")
        except Exception as exc:
            self._emit("error", text=str(exc))

    def _handle_response(self, payload: dict[str, Any]) -> None:
        label = "summary" if payload.get("response_mode") == RESPONSE_MODE_FREEFORM else "action"
        delivery_suffix = "  delivery=precomputed" if payload.get("delivery_mode") == "precomputed" else ""
        self._append_log(
            (
                f"[response] {payload['kind'].upper()} step={payload['step']}  "
                f"arrived={payload['arrival_time']:.1f}s  "
                f"dispatch={payload['dispatch_time']:.1f}s  "
                f"{label}={payload['action_text']}  "
                f"gemini={payload['latency']:.1f}s  total={payload['total_worker_sec']:.1f}s"
                f"{delivery_suffix}"
            )
        )
        self._append_log(
            f"           clip={payload['clip_start_sec']:.1f}-{payload['clip_end_sec']:.1f}s  "
            f"size={payload['request_size_kb']:.1f}KB  cost=${payload['cost_usd']:.4f}"
        )
        self._append_log(f"           {payload['stage_summary']}")
        self._update_timeline_response(payload)
        self.history_payloads.append(payload)
        self.history_index_by_request_id[payload["request_id"]] = len(self.history_payloads) - 1
        self.history_listbox.insert("end", self._format_history_label(payload))
        last_idx = len(self.history_payloads) - 1
        self.history_listbox.selection_clear(0, "end")
        self.history_listbox.selection_set(last_idx)
        self.history_listbox.see(last_idx)
        self._render_payload(payload)
        self.status_var.set(
            f"Last response at t={payload['arrival_time']:.1f}s  "
            f"(gemini={payload['latency']:.1f}s, {payload.get('delivery_mode', 'realtime')})"
        )

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event["type"]
                if kind == "log":
                    self._append_log(event["text"])
                elif kind == "status":
                    self.status_var.set(event["text"])
                elif kind == "dispatch":
                    self._handle_dispatch(event["payload"])
                elif kind == "response":
                    self._handle_response(event["payload"])
                elif kind == "source_video":
                    self.selected_source_var.set(f"Source: {_compact_path(event['video_path'])}")
                    self._set_timeline_video_duration(float(event["duration_sec"]))
                    self._start_source_video_playback(event["video_path"], float(event["duration_sec"]))
                elif kind == "finished":
                    self.running_var.set("Not running")
                    self.status_var.set("Finished")
                    self._append_log("[done] Simulation finished.")
                elif kind == "error":
                    self.running_var.set("Not running")
                    self.status_var.set("Error")
                    self._append_log(f"[error] {event['text']}")
                    messagebox.showerror("Simulation Error", event["text"])
        except queue_mod.Empty:
            pass
        finally:
            if not (self.worker_thread and self.worker_thread.is_alive()):
                self.running_var.set("Not running")
            self.root.after(POLL_MS, self._poll_events)


def main() -> None:
    root = tk.Tk()
    app = TextSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
