#!/usr/bin/env python3
"""
Per-second VLM timeline analysis for task_intervention.

For each one-second clip, the model predicts whether a control-information
event is active, the human intent, inferred robot state, and a target-object
bounding box on the center frame. Event activity is evaluated against the
existing interval labels using the sample midpoint. Bounding-box quality is
stored for later manual review and a weak gaze-containment proxy is reported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import cv2
import numpy as np
from google import genai
from google.genai import types

from eval_vlm_baseline import (
    PROMPT_CONFIG_NOTES,
    extract_cost,
    prepare_video,
    resolve_gemini_api_key,
)
from eval_vlm_multisegment import MODEL_CHOICES, load_ground_truth_multi


EVENT_TYPES = [
    "target_reference",
    "target_correction",
    "destination_reference",
    "direction_adjustment",
    "stop",
    "go",
    "priority_order",
    "confirmation",
    "other_control",
]

ROBOT_STATES = [
    "awaiting_target_selection",
    "awaiting_target_confirmation",
    "moving_to_target",
    "awaiting_destination",
    "moving_to_destination",
    "adjusting_motion",
    "paused",
    "resuming",
    "task_complete",
    "unknown",
]


PROMPT_VARIANTS = {
    "baseline": """\
You are given:
1. a 1-second first-person human-robot interaction video clip with audio and burned-in captions
2. the center frame of that clip as a separate image
3. the local transcript text spoken during this clip:
   "{clip_text}"

Determine whether this clip contains an actionable control-information event:
information from the human that should change, refine, or confirm the robot's behavior.

If the clip contains such an event:
- classify the primary event_type
- describe the human_intent briefly
- infer the robot_state
- identify the target_object
- on the center frame image, output a tight bounding box around the target object or target destination region

If there is no actionable control event in this 1-second clip, set event_active=false and bbox_xyxy_1000=null.

Allowed event_type labels:
{event_types}

Allowed robot_state labels:
{robot_states}

Return JSON:
{{
  "event_active": <true or false>,
  "event_type": "<allowed label or other_control>",
  "event_description": "<short phrase>",
  "human_intent": "<short phrase>",
  "robot_state": "<allowed label>",
  "target_object": "<short noun phrase or empty>",
  "bbox_xyxy_1000": [x1, y1, x2, y2] or null,
  "confidence": <float from 0.0 to 1.0>
}}

Important:
- bbox coordinates refer to the separate center-frame image only.
- bbox coordinates must be integers in [0, 1000].
- Use null when the target is not visible in the center frame.
""",
    "strict_control_bbox": """\
You are given:
1. a 1-second first-person human-robot interaction video clip with audio and burned-in captions
2. the center frame of that clip as a separate image
3. the local transcript text spoken during this clip:
   "{clip_text}"

Your goal is to detect only NEW or DECISIVE control-information moments.
Mark event_active=true only when this exact 1-second clip contains a human signal
that would make the robot choose, correct, stop, resume, redirect, or confirm an action.

Positive examples:
- selecting the target object
- correcting the target ("not that one", "this one")
- indicating the destination ("here")
- motion correction ("more right", "left", "closer")
- stop/go commands
- ordering/prioritizing objects ("purple first")
- brief confirmation that still changes or finalizes behavior

Negative examples:
- filler speech
- repeated words with no new control meaning
- passive commentary
- aftermath after the control decision has already been made

If event_active=true:
- choose one primary event_type
- summarize the human_intent
- infer the robot_state
- identify the target_object
- on the center frame image, output a TIGHT box around only the referenced object or destination region

If event_active=false:
- use bbox_xyxy_1000=null

Allowed event_type labels:
{event_types}

Allowed robot_state labels:
{robot_states}

Return JSON:
{{
  "event_active": <true or false>,
  "event_type": "<allowed label or other_control>",
  "event_description": "<short phrase>",
  "human_intent": "<short phrase>",
  "robot_state": "<allowed label>",
  "target_object": "<short noun phrase or empty>",
  "bbox_xyxy_1000": [x1, y1, x2, y2] or null,
  "confidence": <float from 0.0 to 1.0>
}}

Important:
- bbox coordinates refer to the separate center-frame image only.
- bbox coordinates must be integers in [0, 1000].
- Use null when the target is not visible in the center frame.
- Do not draw a large scene box. Prefer no box over a loose box.
""",
    "center_moment_strict": """\
You are given:
1. a 1-second first-person human-robot interaction video clip with audio and burned-in captions
2. the center frame of that clip as a separate image
3. the local transcript text spoken during this clip:
   "{clip_text}"

Judge ONLY the CENTER MOMENT of the clip.
That means event_active should be true only if the exact center frame / center time
is inside a decisive control-information moment.

Set event_active=false when:
- the event happened earlier in the 1-second clip but the center moment is already aftermath
- the event happens later in the clip but not yet at the center moment
- the clip contains only filler speech, passive commentary, or lingering context
- the target object is visible but the human is not actively giving control information at the center moment

Set event_active=true only when the center moment itself contains active control information
that should affect robot behavior, such as:
- selecting or indicating the target object
- correcting the target
- indicating the destination
- giving stop/go
- adjusting direction
- giving an ordering cue
- giving a short confirmation that finalizes behavior

If event_active=true:
- choose one primary event_type
- summarize the human_intent
- infer the robot_state
- identify the target_object
- on the center frame image, output a tight bbox around only the referenced object or destination region

If event_active=false:
- bbox_xyxy_1000 must be null

Allowed event_type labels:
{event_types}

Allowed robot_state labels:
{robot_states}

Return JSON:
{{
  "event_active": <true or false>,
  "event_type": "<allowed label or other_control>",
  "event_description": "<short phrase>",
  "human_intent": "<short phrase>",
  "robot_state": "<allowed label>",
  "target_object": "<short noun phrase or empty>",
  "bbox_xyxy_1000": [x1, y1, x2, y2] or null,
  "confidence": <float from 0.0 to 1.0>
}}

Important:
- event_active is about the center moment only, not the entire second.
- bbox coordinates refer to the separate center-frame image only.
- bbox coordinates must be integers in [0, 1000].
- Prefer false over a speculative positive prediction.
- Prefer null over a loose box.
""",
}


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def trim_video_clip(video_path: str, start_sec: float, end_sec: float, output_path: str) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            video_path,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
    )


def trim_audio_clip(audio_path: str, start_sec: float, end_sec: float, output_path: str) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            audio_path,
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )


def build_clip_transcript(transcript: dict, start_sec: float, end_sec: float) -> dict:
    words = transcript.get("words", []) if isinstance(transcript, dict) else []
    clip_words = []
    for word in words:
        try:
            ws = float(word["start"])
            we = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if we < start_sec or ws > end_sec:
            continue
        clip_words.append(
            {
                "word": str(word.get("word", "")),
                "start": max(0.0, ws - start_sec),
                "end": max(0.0, we - start_sec),
            }
        )

    clip_text = " ".join(word["word"] for word in clip_words).strip()
    if not clip_text:
        clip_text = str(transcript.get("text", "")).strip()
    return {
        "text": clip_text,
        "words": clip_words,
    }


def extract_frame(video_path: str, frame_idx: int) -> tuple[np.ndarray, int, int]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    h, w = frame.shape[:2]
    return frame, w, h


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("failed to encode jpeg")
    return bytes(buf)


def build_prompt(variant: str, clip_text: str) -> str:
    return PROMPT_VARIANTS[variant].format(
        clip_text=clip_text.strip() or "[no speech in this clip]",
        event_types=", ".join(EVENT_TYPES),
        robot_states=", ".join(ROBOT_STATES),
    )


def call_gemini_video_and_image(
    *,
    client: genai.Client,
    model_name: str,
    video_bytes: bytes,
    image_bytes: bytes,
    prompt: str,
    video_fps: int,
    thinking_budget: int,
):
    parts = [
        types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=video_fps),
        ),
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
        types.Part(text=prompt),
    ]
    return client.models.generate_content(
        model=model_name,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            response_mime_type="application/json",
        ),
    )


def _sanitize_event_type(value: str) -> str:
    value = str(value).strip()
    return value if value in EVENT_TYPES else "other_control"


def _sanitize_robot_state(value: str) -> str:
    value = str(value).strip()
    return value if value in ROBOT_STATES else "unknown"


def _parse_bbox(item) -> list[int] | None:
    if item is None:
        return None
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return None
    try:
        coords = [int(round(float(v))) for v in item]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = coords
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_response(response) -> dict | None:
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    event_active = bool(payload.get("event_active", False))
    bbox = _parse_bbox(payload.get("bbox_xyxy_1000"))
    if not event_active:
        bbox = None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "event_active": event_active,
        "event_type": _sanitize_event_type(payload.get("event_type", "other_control")),
        "event_description": str(payload.get("event_description", "")).strip(),
        "human_intent": str(payload.get("human_intent", "")).strip(),
        "robot_state": _sanitize_robot_state(payload.get("robot_state", "unknown")),
        "target_object": str(payload.get("target_object", "")).strip(),
        "bbox_xyxy_1000": bbox,
        "confidence": confidence,
    }


def bbox_to_pixels(bbox_xyxy_1000: list[int] | None, width: int, height: int) -> list[int] | None:
    if bbox_xyxy_1000 is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy_1000
    px = [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]
    px[0] = max(0, min(width - 1, px[0]))
    px[1] = max(0, min(height - 1, px[1]))
    px[2] = max(0, min(width - 1, px[2]))
    px[3] = max(0, min(height - 1, px[3]))
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return px


def draw_annotation(
    frame: np.ndarray,
    bbox_px: list[int] | None,
    gaze_xy: tuple[float, float] | None,
    label: str,
) -> np.ndarray:
    out = frame.copy()
    if bbox_px is not None:
        x1, y1, x2, y2 = bbox_px
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    if gaze_xy is not None:
        gx, gy = gaze_xy
        cv2.circle(out, (int(round(gx)), int(round(gy))), 5, (0, 255, 0), -1)
        cv2.circle(out, (int(round(gx)), int(round(gy))), 7, (0, 0, 0), 1)
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 8 + 12 * len(label)), 40), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_gaze_lookup(gaze_path: str) -> dict[int, tuple[float, float]]:
    if not os.path.exists(gaze_path):
        return {}
    frames = json.load(open(gaze_path)).get("frames", [])
    return {
        int(item["frame_idx"]): (float(item["gaze_x"]), float(item["gaze_y"]))
        for item in frames
        if "frame_idx" in item and "gaze_x" in item and "gaze_y" in item
    }


def process_sample(
    *,
    api_key: str,
    model: str,
    episode_dir: str,
    episode_name: str,
    transcript: dict,
    gt: dict,
    gaze_lookup: dict[int, tuple[float, float]],
    second_idx: int,
    prompt_variant: str,
    clip_duration: float,
    video_fps: int,
    thinking_budget: int,
    target_resolution: tuple[int, int] | None,
    include_audio: bool,
    caption: bool,
    audio_dir: str,
    output_dir: str,
) -> dict:
    client = genai.Client(api_key=api_key)
    video_path = os.path.join(episode_dir, "video.mp4")
    audio_path = os.path.join(audio_dir, f"{episode_name}_audio.wav")
    sample_sec = float(second_idx) + 0.5
    sample_sec = min(sample_sec, max(0.0, gt["duration_sec"] - (1.0 / gt["fps"])))
    half = clip_duration / 2.0
    clip_start = max(0.0, sample_sec - half)
    clip_end = min(gt["duration_sec"], sample_sec + half)
    if clip_end - clip_start < clip_duration and gt["duration_sec"] > clip_duration:
        if clip_start <= 0.0:
            clip_end = min(gt["duration_sec"], clip_duration)
        elif clip_end >= gt["duration_sec"]:
            clip_start = max(0.0, gt["duration_sec"] - clip_duration)
    center_sec = sample_sec
    center_frame = int(round(center_sec * gt["fps"]))
    center_frame = max(0, min(gt["total_frames"] - 1, center_frame))

    with tempfile.TemporaryDirectory(prefix=f"timeline_{episode_name}_{second_idx}_") as tmpdir:
        clip_video = os.path.join(tmpdir, "clip.mp4")
        trim_video_clip(video_path, clip_start, clip_end, clip_video)

        clip_audio = None
        if include_audio and os.path.exists(audio_path):
            clip_audio = os.path.join(tmpdir, "clip.wav")
            trim_audio_clip(audio_path, clip_start, clip_end, clip_audio)

        clip_transcript = build_clip_transcript(transcript, clip_start, clip_end)
        video_bytes = prepare_video(
            video_path=clip_video,
            caption=caption,
            gaze_annot=False,
            transcript=clip_transcript,
            gaze_data=None,
            target_resolution=target_resolution,
            include_audio=include_audio,
            audio_path=clip_audio,
        )

        center_img, width, height = extract_frame(video_path, center_frame)
        image_bytes = encode_jpeg(center_img)
        prompt = build_prompt(prompt_variant, clip_transcript.get("text", ""))

        infer_start = time.perf_counter()
        response = call_gemini_video_and_image(
            client=client,
            model_name=model,
            video_bytes=video_bytes,
            image_bytes=image_bytes,
            prompt=prompt,
            video_fps=video_fps,
            thinking_budget=thinking_budget,
        )
        infer_time = time.perf_counter() - infer_start
        parsed = parse_response(response)
        cost = extract_cost(response, model)

        gt_positive = center_frame in gt["gt_kf_set"]
        gaze_xy = gaze_lookup.get(center_frame)
        bbox_px = None
        gaze_in_bbox = None

        frame_rel_path = None
        if parsed is not None:
            bbox_px = bbox_to_pixels(parsed["bbox_xyxy_1000"], width, height)
            if bbox_px is not None and gaze_xy is not None:
                x1, y1, x2, y2 = bbox_px
                gx, gy = gaze_xy
                gaze_in_bbox = bool(x1 <= gx <= x2 and y1 <= gy <= y2)

            frame_dir = os.path.join(output_dir, "frames", episode_name)
            os.makedirs(frame_dir, exist_ok=True)
            label = (
                f"t={second_idx}s | active={int(parsed['event_active'])} | "
                f"{parsed['event_type']} | {parsed['robot_state']}"
            )
            annotated = draw_annotation(center_img, bbox_px, gaze_xy, label)
            frame_path = os.path.join(frame_dir, f"sec_{second_idx:03d}.jpg")
            cv2.imwrite(frame_path, annotated)
            frame_rel_path = os.path.relpath(frame_path, os.getcwd())

        return {
            "episode_name": episode_name,
            "second_idx": second_idx,
            "clip_start_sec": round(clip_start, 3),
            "clip_end_sec": round(clip_end, 3),
            "center_sec": round(center_sec, 3),
            "center_frame": center_frame,
            "gt_positive": gt_positive,
            "prediction": parsed,
            "bbox_px": bbox_px,
            "gaze_xy": [round(gaze_xy[0], 2), round(gaze_xy[1], 2)] if gaze_xy else None,
            "gaze_in_bbox": gaze_in_bbox,
            "frame_path": frame_rel_path,
            "cost": cost,
            "inference_time_sec": round(infer_time, 4),
            "raw_response": response.text if parsed is None else "",
        }


def aggregate_samples(samples: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    bbox_on_gt_positive = 0
    gaze_proxy_hits = 0
    gaze_proxy_total = 0
    pred_event_counter = Counter()
    robot_state_counter = Counter()
    pred_positive = 0
    gt_positive = 0
    infer_times = []
    total_cost = 0.0
    total_input = total_output = total_thinking = 0

    episode_seconds: dict[str, list[dict]] = {}
    for sample in samples:
        episode_seconds.setdefault(sample["episode_name"], []).append(sample)
        pred = sample.get("prediction")
        gt_pos = bool(sample["gt_positive"])
        pred_pos = bool(pred and pred["event_active"])

        if gt_pos:
            gt_positive += 1
        if pred_pos:
            pred_positive += 1

        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif (not pred_pos) and gt_pos:
            fn += 1
        else:
            tn += 1

        if gt_pos and pred and pred["bbox_xyxy_1000"] is not None:
            bbox_on_gt_positive += 1

        if gt_pos and sample["gaze_in_bbox"] is not None:
            gaze_proxy_total += 1
            if sample["gaze_in_bbox"]:
                gaze_proxy_hits += 1

        if pred:
            pred_event_counter[pred["event_type"]] += int(pred["event_active"])
            robot_state_counter[pred["robot_state"]] += 1

        infer_times.append(sample["inference_time_sec"])
        cost = sample.get("cost", {})
        total_cost += cost.get("cost_usd", 0.0)
        total_input += cost.get("input_tokens", 0)
        total_output += cost.get("output_tokens", 0)
        total_thinking += cost.get("thinking_tokens", 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(samples) if samples else 0.0

    per_episode = []
    for episode_name, seq in sorted(episode_seconds.items()):
        seq = sorted(seq, key=lambda item: item["second_idx"])
        pred_bins = [1 if item.get("prediction") and item["prediction"]["event_active"] else 0 for item in seq]
        gt_bins = [1 if item["gt_positive"] else 0 for item in seq]
        pred_runs = sum(1 for idx, val in enumerate(pred_bins) if val and (idx == 0 or not pred_bins[idx - 1]))
        gt_runs = sum(1 for idx, val in enumerate(gt_bins) if val and (idx == 0 or not gt_bins[idx - 1]))
        per_episode.append(
            {
                "episode_name": episode_name,
                "pred_positive_seconds": sum(pred_bins),
                "gt_positive_seconds": sum(gt_bins),
                "pred_event_runs": pred_runs,
                "gt_event_runs": gt_runs,
            }
        )

    return {
        "num_samples": len(samples),
        "gt_positive_samples": gt_positive,
        "pred_positive_samples": pred_positive,
        "event_active_precision": round(precision, 4),
        "event_active_recall": round(recall, 4),
        "event_active_f1": round(f1, 4),
        "event_active_accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "bbox_presence_on_gt_positive_rate": round(
            bbox_on_gt_positive / gt_positive, 4
        ) if gt_positive else 0.0,
        "gaze_in_bbox_rate_on_gt_positive": round(
            gaze_proxy_hits / gaze_proxy_total, 4
        ) if gaze_proxy_total else None,
        "pred_event_type_counts": dict(sorted(pred_event_counter.items())),
        "robot_state_counts": dict(sorted(robot_state_counter.items())),
        "mean_inference_time_sec": round(statistics.mean(infer_times), 4) if infer_times else 0.0,
        "median_inference_time_sec": round(statistics.median(infer_times), 4) if infer_times else 0.0,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "per_episode_timeline_counts": per_episode,
    }


def print_summary(agg: dict, model: str, prompt_variant: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Task Intervention Timeline Analysis")
    print(f"  Model: {model}  |  Prompt: {prompt_variant}  |  Samples: {agg['num_samples']}")
    print(sep)
    print("\n  Event Active:")
    print(f"    Precision: {agg['event_active_precision']*100:5.1f}%")
    print(f"    Recall:    {agg['event_active_recall']*100:5.1f}%")
    print(f"    F1:        {agg['event_active_f1']*100:5.1f}%")
    print(f"    Accuracy:  {agg['event_active_accuracy']*100:5.1f}%")
    print(f"    TP/FP/TN/FN: {agg['tp']}/{agg['fp']}/{agg['tn']}/{agg['fn']}")

    print("\n  Bounding Box Proxy:")
    print(f"    BBox on GT-positive samples: {agg['bbox_presence_on_gt_positive_rate']*100:5.1f}%")
    gaze_rate = agg["gaze_in_bbox_rate_on_gt_positive"]
    if gaze_rate is None:
        print("    Gaze-in-bbox proxy: n/a")
    else:
        print(f"    Gaze-in-bbox on GT-positive: {gaze_rate*100:5.1f}%")

    print("\n  Runtime / Cost:")
    print(f"    Mean inference time: {agg['mean_inference_time_sec']:.2f}s")
    print(f"    Median inference time: {agg['median_inference_time_sec']:.2f}s")
    print(f"    Total cost: ${agg['total_cost_usd']:.4f}")
    print(f"    Thinking tokens: {agg['total_thinking_tokens']}")

    print("\n  Predicted Event Types:")
    for key, value in agg["pred_event_type_counts"].items():
        print(f"    {key}: {value}")
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--audio-dir", default="./preproc_files")
    parser.add_argument("--episodes", default="", help="Comma-separated task_intervention episode names")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--prompt-variant", default="strict_control_bbox", choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--clip-duration", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-resolution", default="256")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional global cap for debugging")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--exp-suffix", default="")
    return parser.parse_args()


def _parse_target_resolution(value: str) -> tuple[int, int] | None:
    lowered = value.strip().lower()
    if lowered in {"none", "raw", "original"}:
        return None
    size = int(lowered)
    return (size, size)


def _resolve_episodes(dataset_dir: str, episodes_arg: str) -> list[str]:
    if episodes_arg.strip():
        return [name.strip() for name in episodes_arg.split(",") if name.strip()]
    return sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith("task_intervention") and os.path.isdir(os.path.join(dataset_dir, name))
    )


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    dataset_dir = os.path.abspath(args.dataset_dir)
    audio_dir = os.path.abspath(args.audio_dir)
    target_resolution = _parse_target_resolution(args.target_resolution)
    episodes = _resolve_episodes(dataset_dir, args.episodes)

    tag = (
        f"timeline_{args.model}_audio_{'true' if args.include_audio else 'false'}"
        f"_cap_{'true' if args.caption else 'false'}_fps{args.video_fps}"
        f"_tb{args.thinking_budget}_prompt_{args.prompt_variant}"
    )
    if args.exp_suffix:
        tag += f"_{args.exp_suffix}"

    out_dir = os.path.join("results", tag)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    samples = []
    for episode_name in episodes:
        episode_dir = os.path.join(dataset_dir, episode_name)
        gt = load_ground_truth_multi(episode_dir)
        if gt is None:
            continue
        transcript = json.load(open(os.path.join(episode_dir, "transcript.json")))
        gaze_lookup = load_gaze_lookup(os.path.join(episode_dir, "gaze.json"))
        max_second = int(gt["duration_sec"])
        for second_idx in range(max_second):
            samples.append(
                {
                    "episode_name": episode_name,
                    "episode_dir": episode_dir,
                    "transcript": transcript,
                    "gt": gt,
                    "gaze_lookup": gaze_lookup,
                    "second_idx": second_idx,
                }
            )

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(
        f"[timeline] episodes={len(episodes)} samples={len(samples)} model={args.model} "
        f"audio={'on' if args.include_audio else 'off'} caption={'on' if args.caption else 'off'} "
        f"thinking={args.thinking_budget} prompt={args.prompt_variant} clip={args.clip_duration:.2f}s "
        f"resolution={target_resolution}"
    )

    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_idx = {}
        for idx, sample in enumerate(samples):
            future = executor.submit(
                process_sample,
                api_key=api_key,
                model=args.model,
                episode_dir=sample["episode_dir"],
                episode_name=sample["episode_name"],
                transcript=sample["transcript"],
                gt=sample["gt"],
                gaze_lookup=sample["gaze_lookup"],
                second_idx=sample["second_idx"],
                prompt_variant=args.prompt_variant,
                clip_duration=args.clip_duration,
                video_fps=args.video_fps,
                thinking_budget=args.thinking_budget,
                target_resolution=target_resolution,
                include_audio=args.include_audio,
                caption=args.caption,
                audio_dir=audio_dir,
                output_dir=out_dir,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            sample = samples[idx]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "episode_name": sample["episode_name"],
                    "second_idx": sample["second_idx"],
                    "clip_start_sec": sample["second_idx"],
                    "clip_end_sec": sample["second_idx"] + 1,
                    "center_sec": sample["second_idx"] + 0.5,
                    "gt_positive": False,
                    "prediction": None,
                    "bbox_px": None,
                    "gaze_xy": None,
                    "gaze_in_bbox": None,
                    "frame_path": None,
                    "cost": {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
                    "inference_time_sec": 0.0,
                    "raw_response": f"exception: {exc}",
                }
            results[idx] = result
            pred = result.get("prediction")
            status = "1" if pred and pred["event_active"] else "0"
            print(
                f"  {result['episode_name']} sec={result['second_idx']:02d} "
                f"gt={int(result['gt_positive'])} pred={status} "
                f"type={(pred['event_type'] if pred else 'ERR')}"
            )

    agg = aggregate_samples(results)
    print_summary(agg, args.model, args.prompt_variant)

    json_path = os.path.join(out_dir, "timeline_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "aggregate": agg,
                "config": {
                    "model": args.model,
                    "video_fps": args.video_fps,
                    "thinking_budget": args.thinking_budget,
                    "caption": args.caption,
                    "include_audio": args.include_audio,
                "prompt_variant": args.prompt_variant,
                "clip_duration": args.clip_duration,
                "target_resolution": target_resolution,
            },
                "samples": results,
            },
            f,
            indent=2,
        )
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
