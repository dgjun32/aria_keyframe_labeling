#!/usr/bin/env python3
"""Analyze decisive task_intervention events and midpoint bounding boxes on a prepared MP4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time

import cv2
import numpy as np
from google import genai
from google.genai import types

from eval_vlm_baseline import call_gemini, extract_cost, resolve_gemini_api_key
from eval_vlm_multisegment import MODEL_CHOICES, evaluate_episode_multi, load_ground_truth_multi


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

INTERACTION_TYPES = [
    "pick_target",
    "place_target",
    "rejected_target",
    "other_relevant",
]

PROMPT_TEMPLATE = """\
You are analyzing a first-person human-robot interaction video recorded in a store.
The human wearer gives control information to the robot.

The spoken transcript is:
  "{transcript}"

The video already includes burned-in captions and the original audio.
Use the whole video context to identify the decisive control-information events.

A control-information event is a brief moment when the human gives information
that should change, refine, or confirm the robot's behavior.

Your task:
1. identify all distinct decisive control-information events
2. for each event, return a short [start, end] interval whose midpoint lands
   on the decisive moment itself
3. at each event midpoint, return only the few interaction-relevant objects

Important correction rule:
- If the human first indicates one object and then corrects it ("not that one",
  "this one", "that one"), the pick_target should be the object the robot
  should act on AFTER the correction.
- If useful, you may also include the rejected object as rejected_target.

For each event, return:
- event_type from: {event_types}
- short event_description
- short [start, end] interval
- midpoint_objects: a few interaction-relevant boxes visible at that midpoint

Each midpoint object must include:
- label
- bbox_xyxy_1000
- interaction_type from: {interaction_types}
- interaction_score
- confidence

Return JSON:
{{
  "reasoning": "<brief explanation>",
  "events": [
    {{
      "event_type": "<allowed label>",
      "event_description": "<short description>",
      "start": <float>,
      "end": <float>,
      "midpoint_objects": [
        {{
          "label": "<short object label>",
          "bbox_xyxy_1000": [x1, y1, x2, y2],
          "interaction_type": "<allowed interaction type>",
          "interaction_score": <float 0.0 to 1.0>,
          "confidence": <float 0.0 to 1.0>
        }}
      ]
    }}
  ]
}}

Important:
- Return between 1 and {max_events} events.
- Events must be sorted by time and non-overlapping.
- Always return positive start/end values.
- Never return -1.
- Keep intervals tight, usually 0.2-2.0 seconds.
- midpoint_objects may be empty if no clear relevant object is visible.
- bbox coordinates must be integers in [0, 1000].
- Do not return human hand, human arm, robot arm, robot gripper, or robot body as objects.
- Prefer a few clean boxes over many uncertain ones.
"""

BBOX_SYSTEM_INSTRUCTION = """\
You are an expert at object localization for robot control.
When asked to detect objects, return only tight boxes around the intended physical object or destination region.
Use the exact frame image as the coordinate reference.
If multiple similar objects are visible, use pointing direction, hand-object contact, nearby motion, and speech context to disambiguate them.
"""

BBOX_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "description": "Control-relevant detections on the full frame image.",
            "minItems": 0,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short object label such as muffin or plate.",
                    },
                    "role": {
                        "type": "string",
                        "enum": INTERACTION_TYPES,
                        "description": "Role of this object in the robot's immediate control decision.",
                    },
                    "box_2d": {
                        "type": "array",
                        "description": "Bounding box on the first image only, in [ymin, xmin, ymax, xmax] normalized to 0-1000.",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1000,
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["label", "role", "box_2d", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--transcript-path", required=True)
    parser.add_argument("--episode-dir", default="")
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite-preview",
        choices=MODEL_CHOICES,
    )
    parser.add_argument(
        "--bbox-model",
        default="gemini-2.5-flash",
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--bbox-video-fps", type=int, default=8)
    parser.add_argument("--bbox-thinking-budget", type=int, default=0)
    parser.add_argument("--bbox-clip-padding-sec", type=float, default=0.75)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--match-frame-tolerance", type=int, default=3)
    parser.add_argument("--reuse-events-from", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--exp-suffix", default="")
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def build_prompt(transcript: str, max_events: int) -> str:
    return PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        event_types=", ".join(EVENT_TYPES),
        interaction_types=", ".join(INTERACTION_TYPES),
        max_events=max_events,
    )


def read_video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 15.0
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def _parse_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_response(response, max_events: int):
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None

    parsed = []
    for item in events[:max_events]:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end < 0:
            continue
        if start > end:
            start, end = end, start
        event_type = str(item.get("event_type", "other_control")).strip()
        if event_type not in EVENT_TYPES:
            event_type = "other_control"
        objects = item.get("midpoint_objects", [])
        parsed_objects = []
        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                bbox = _parse_bbox(obj.get("bbox_xyxy_1000"))
                if bbox is None:
                    continue
                try:
                    confidence = float(obj.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                try:
                    interaction_score = float(obj.get("interaction_score", confidence))
                except (TypeError, ValueError):
                    interaction_score = confidence
                interaction_type = str(obj.get("interaction_type", "")).strip()
                if interaction_type not in INTERACTION_TYPES:
                    interaction_type = "other_relevant"
                parsed_objects.append(
                    {
                        "label": str(obj.get("label", "")).strip() or "object",
                        "bbox_xyxy_1000": bbox,
                        "confidence": max(0.0, min(1.0, confidence)),
                        "interaction_type": interaction_type,
                        "interaction_score": max(0.0, min(1.0, interaction_score)),
                    }
                )
        parsed.append(
            {
                "event_type": event_type,
                "event_description": str(item.get("event_description", "")).strip(),
                "start_sec": start,
                "end_sec": end,
                "midpoint_objects": parsed_objects,
            }
        )
    parsed.sort(key=lambda event: (event["start_sec"], event["end_sec"]))
    if not parsed:
        return None
    return {"reasoning": str(payload.get("reasoning", "")), "events": parsed}


def normalize_prediction(pred: dict, meta: dict) -> dict:
    fps = meta["fps"]
    duration_sec = meta["duration_sec"]
    total_frames = meta["total_frames"]
    normalized_events = []
    for idx, event in enumerate(pred["events"]):
        start_sec = max(0.0, min(float(event["start_sec"]), duration_sec))
        end_sec = max(0.0, min(float(event["end_sec"]), duration_sec))
        if start_sec > end_sec:
            start_sec, end_sec = end_sec, start_sec
        start_frame = max(0, min(total_frames - 1, int(round(start_sec * fps)))) if total_frames else 0
        end_frame = max(0, min(total_frames - 1, int(round(end_sec * fps)))) if total_frames else 0
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        mid_sec = (start_sec + end_sec) / 2
        mid_frame = max(0, min(total_frames - 1, int(round(mid_sec * fps)))) if total_frames else 0
        normalized_events.append(
            {
                "pred_idx": idx,
                "event_type": event["event_type"],
                "event_description": event["event_description"],
                "start_sec": round(start_sec, 4),
                "end_sec": round(end_sec, 4),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "mid_sec": round(mid_sec, 4),
                "mid_frame": mid_frame,
                "midpoint_objects": event["midpoint_objects"],
            }
        )
    return {
        "reasoning": pred.get("reasoning", ""),
        "events": normalized_events,
        "intervals": [
            {
                "pred_idx": event["pred_idx"],
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "mid_sec": event["mid_sec"],
                "mid_frame": event["mid_frame"],
            }
            for event in normalized_events
        ],
    }


def bbox_to_pixels(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]


def interaction_color(interaction_type: str) -> tuple[int, int, int]:
    return {
        "pick_target": (80, 220, 80),
        "place_target": (255, 190, 0),
        "rejected_target": (80, 80, 255),
        "other_relevant": (0, 200, 255),
    }.get(interaction_type, (0, 200, 255))


def draw_boxes(frame, event: dict, objects: list[dict]):
    out = frame.copy()
    title = (
        f"event {event['pred_idx']}: {event['event_type']} | "
        f"{event['start_sec']:.2f}-{event['end_sec']:.2f}s | mid {event['mid_sec']:.2f}s"
    )
    cv2.rectangle(out, (18, 18), (min(out.shape[1] - 18, 18 + 13 * len(title)), 58), (0, 0, 0), -1)
    cv2.putText(out, title, (26, 45), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_px"]
        color = interaction_color(obj.get("interaction_type", "other_relevant"))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        score = obj.get("interaction_score", obj.get("confidence", 0.0))
        label = f"{obj['label']} [{obj.get('interaction_type')}] {score:.2f}"
        y = max(24, y1 - 8)
        cv2.rectangle(out, (x1, y - 22), (min(out.shape[1] - 1, x1 + 11 * len(label)), y + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 3, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def select_primary_object(objects: list[dict]) -> dict | None:
    if not objects:
        return None
    priority = {
        "pick_target": 0,
        "place_target": 1,
        "rejected_target": 2,
        "other_relevant": 3,
    }
    return sorted(
        objects,
        key=lambda obj: (
            priority.get(obj.get("interaction_type", "other_relevant"), 9),
            -float(obj.get("interaction_score", obj.get("confidence", 0.0))),
            -float(obj.get("confidence", 0.0)),
        ),
    )[0]


def prune_objects_for_event(event_type: str, objects: list[dict]) -> list[dict]:
    if not objects:
        return []
    priority = {
        "pick_target": 0,
        "place_target": 1,
        "rejected_target": 2,
        "other_relevant": 3,
    }
    ranked = sorted(
        objects,
        key=lambda obj: (
            priority.get(obj.get("interaction_type", "other_relevant"), 9),
            -float(obj.get("interaction_score", obj.get("confidence", 0.0))),
            -float(obj.get("confidence", 0.0)),
        ),
    )
    by_role = {}
    for obj in ranked:
        by_role.setdefault(obj.get("interaction_type", "other_relevant"), []).append(obj)

    if event_type == "destination_reference":
        return by_role.get("place_target", ranked[:1])[:1]
    if event_type == "target_correction":
        kept = []
        if by_role.get("pick_target"):
            kept.append(by_role["pick_target"][0])
        if by_role.get("rejected_target"):
            kept.append(by_role["rejected_target"][0])
        if not kept:
            kept = ranked[:2]
        return kept[:2]
    if by_role.get("pick_target"):
        return by_role["pick_target"][:1]
    return ranked[:1]


def _safe_name(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "run"


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def trim_video_clip_with_audio(video_path: str, start_sec: float, end_sec: float, output_path: str) -> None:
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
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )


def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("failed to encode frame jpeg")
    return bytes(buf)


def build_local_transcript(transcript: dict, start_sec: float, end_sec: float) -> str:
    words = transcript.get("words", []) if isinstance(transcript, dict) else []
    picked = []
    for word in words:
        try:
            ws = float(word["start"])
            we = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if we < start_sec or ws > end_sec:
            continue
        txt = str(word.get("word", "")).strip()
        if txt:
            picked.append(txt)
    return " ".join(picked).strip()


def build_bbox_prompt(
    *,
    event: dict,
    transcript_text: str,
    local_transcript: str,
    event_reasoning: str,
) -> str:
    event_type = event["event_type"]
    if event_type == "target_correction":
        role_instructions = (
            "Return the final object the robot should pick as pick_target. "
            "If the previously indicated object is also clearly visible, include it as rejected_target."
        )
    elif event_type == "destination_reference":
        role_instructions = (
            "Return only the destination region or receptacle as place_target. "
            "Do not return the carried object unless it is itself the destination."
        )
    else:
        role_instructions = (
            "Return the single main object the robot should act on now as pick_target, "
            "unless a different role is clearly required."
        )

    return f"""\
You are given:
1. The FIRST image: the exact midpoint frame of a decisive control event. All coordinates must refer to this FIRST image only.
2. A short event clip from the same moment, with audio and burned-in captions. Use the clip only to resolve pointing, correction, and control context.

Whole-video transcript:
  "{transcript_text}"

Whole-video event reasoning:
  "{event_reasoning or '[none]'}"

This event from the whole-video pass is:
- event_type: {event_type}
- event_description: {event['event_description'] or '[none]'}
- event_interval: {event['start_sec']:.2f}-{event['end_sec']:.2f} seconds
- event_midpoint: {event['mid_sec']:.2f} seconds

Local speech near this event:
  "{local_transcript or '[no local speech extracted]'}"

Your job is to localize the control-relevant object(s) on the FIRST image only.

Localization rules:
- {role_instructions}
- Use pointing direction, hand proximity, motion in the clip, and caption timing to choose the correct object.
- If multiple similar pastries or products are visible, choose the one specifically indicated at this moment, not a nearby similar item.
- Use tight boxes around the physical object only.
- Never return the human hand, human arm, robot arm, robot gripper, or robot body.
- Do not return a large scene box.
- box_2d must be [ymin, xmin, ymax, xmax] normalized to 0-1000 on the FIRST image only.
- If the true target is not visible in the FIRST image, return an empty list.

Return JSON that matches the provided schema.
"""


def parse_box_2d(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None
    return [xmin, ymin, xmax, ymax]


def parse_bbox_response(response):
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    objects = payload.get("objects")
    if not isinstance(objects, list):
        return None
    parsed_objects = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = parse_box_2d(obj.get("box_2d"))
        if bbox is None:
            continue
        role = str(obj.get("role", "")).strip()
        if role not in INTERACTION_TYPES:
            role = "other_relevant"
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        parsed_objects.append(
            {
                "label": str(obj.get("label", "")).strip() or "object",
                "bbox_xyxy_1000": bbox,
                "confidence": max(0.0, min(1.0, confidence)),
                "interaction_type": role,
                "interaction_score": max(0.0, min(1.0, confidence)),
            }
        )
    return parsed_objects


def call_bbox_model(
    *,
    client: genai.Client,
    model_name: str,
    image_bytes: bytes,
    clip_bytes: bytes,
    prompt: str,
    video_fps: int,
    thinking_budget: int,
):
    parts = [
        types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
        types.Part(
            inline_data=types.Blob(data=clip_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=video_fps),
        ),
        types.Part(text=prompt),
    ]
    return client.models.generate_content(
        model=model_name,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=BBOX_SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            response_mime_type="application/json",
            response_json_schema=BBOX_RESPONSE_SCHEMA,
        ),
    )


def refine_midpoint_boxes(
    *,
    client: genai.Client,
    args: argparse.Namespace,
    video_path: str,
    transcript: dict,
    prediction: dict,
    meta: dict,
):
    bbox_costs = []
    bbox_times = []
    for event in prediction["events"]:
        clip_start = max(0.0, event["start_sec"] - args.bbox_clip_padding_sec)
        clip_end = min(meta["duration_sec"], event["end_sec"] + args.bbox_clip_padding_sec)
        if clip_end <= clip_start:
            clip_end = min(meta["duration_sec"], clip_start + 1.0)

        frame = extract_frame(video_path, event["mid_frame"])
        image_bytes = encode_jpeg(frame)
        local_transcript = build_local_transcript(transcript, clip_start, clip_end)
        prompt = build_bbox_prompt(
            event=event,
            transcript_text=str(transcript.get("text", "")).strip(),
            local_transcript=local_transcript,
            event_reasoning=prediction.get("reasoning", ""),
        )

        with tempfile.TemporaryDirectory(prefix=f"bbox_event_{event['pred_idx']}_") as tmpdir:
            clip_path = Path(tmpdir) / "event_clip.mp4"
            trim_video_clip_with_audio(video_path, clip_start, clip_end, str(clip_path))
            clip_bytes = clip_path.read_bytes()

        infer_start = time.perf_counter()
        response = call_bbox_model(
            client=client,
            model_name=args.bbox_model,
            image_bytes=image_bytes,
            clip_bytes=clip_bytes,
            prompt=prompt,
            video_fps=args.bbox_video_fps,
            thinking_budget=args.bbox_thinking_budget,
        )
        infer_time = time.perf_counter() - infer_start
        parsed = parse_bbox_response(response)
        cost = extract_cost(response, args.bbox_model)
        bbox_costs.append(cost)
        bbox_times.append(infer_time)

        event["coarse_midpoint_objects"] = event["midpoint_objects"]
        event["bbox_refinement"] = {
            "clip_start_sec": round(clip_start, 4),
            "clip_end_sec": round(clip_end, 4),
            "local_transcript": local_transcript,
            "model": args.bbox_model,
            "inference_time_sec": round(infer_time, 4),
            "cost": cost,
            "raw_response": response.text,
        }
        if parsed:
            event["midpoint_objects"] = prune_objects_for_event(event["event_type"], parsed)
            event["bbox_source"] = "refined_image_clip"
        else:
            event["bbox_source"] = "coarse_video_event_fallback"
    return {
        "num_bbox_calls": len(bbox_costs),
        "total_cost_usd": round(sum(item.get("cost_usd", 0.0) for item in bbox_costs), 6),
        "mean_inference_time_sec": round(sum(bbox_times) / len(bbox_times), 4) if bbox_times else 0.0,
        "calls": bbox_costs,
    }


def save_midpoint_frames(video_path: str, prediction: dict, out_dir: Path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    for event in prediction["events"]:
        frame_idx = event["mid_frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        objects = []
        for obj in event["midpoint_objects"]:
            row = dict(obj)
            row["bbox_px"] = bbox_to_pixels(row["bbox_xyxy_1000"], width, height)
            objects.append(row)
        vis = draw_boxes(frame, event, objects)
        frame_path = out_dir / f"event_{event['pred_idx']:02d}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), vis)
        rows.append(
            {
                "pred_idx": event["pred_idx"],
                "event_type": event["event_type"],
                "event_description": event["event_description"],
                "frame_idx": frame_idx,
                "mid_sec": event["mid_sec"],
                "frame_path": str(frame_path),
                "objects": objects,
            }
        )
    cap.release()
    return rows


def _draw_text_block(canvas, lines: list[str], x: int, y: int, line_height: int = 30):
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + idx * line_height),
            FONT,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )


def build_contact_sheet(frame_rows: list[dict], out_path: Path):
    if not frame_rows:
        return
    thumb_w = 420
    thumb_h = 420
    text_w = 900
    pad = 24
    row_h = thumb_h + pad
    canvas_h = pad + len(frame_rows) * row_h
    canvas_w = pad * 3 + thumb_w + text_w
    canvas = np.full((canvas_h, canvas_w, 3), 248, dtype=np.uint8)

    for idx, row in enumerate(frame_rows):
        y0 = pad + idx * row_h
        img = cv2.imread(row["frame_path"])
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y0 : y0 + thumb_h, pad : pad + thumb_w] = thumb

        primary = select_primary_object(row["objects"])
        lines = [
            f"event {row['pred_idx']} | {row['event_type']} | mid={row['mid_sec']:.2f}s | frame={row['frame_idx']}",
            row["event_description"] or "[no description]",
        ]
        if primary is not None:
            lines.append(
                "primary: "
                f"{primary['label']} ({primary.get('interaction_type')}) "
                f"bbox1000={primary['bbox_xyxy_1000']}"
            )
        else:
            lines.append("primary: [none]")
        lines.append(f"objects: {len(row['objects'])}")
        for obj in row["objects"][:4]:
            lines.append(
                f"- {obj['label']} | {obj.get('interaction_type')} | "
                f"bbox1000={obj['bbox_xyxy_1000']} | score={obj.get('interaction_score', obj['confidence']):.2f}"
            )
        _draw_text_block(canvas, lines, pad * 2 + thumb_w, y0 + 36)

    cv2.imwrite(str(out_path), canvas)


def build_timeline_image(
    prediction: dict,
    duration_sec: float,
    out_path: Path,
    transcript_text: str,
    gt: dict | None = None,
):
    width = 1800
    height = 420 if gt else 320
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    left = 140
    right = 60
    line_w = width - left - right
    pred_y = 250 if gt else 200
    gt_y = 130

    title = "Task Intervention Event Timeline"
    subtitle = f"duration={duration_sec:.2f}s | predicted_events={len(prediction['events'])}"
    cv2.putText(canvas, title, (left, 42), FONT, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (left, 76), FONT, 0.7, (60, 60, 60), 2, cv2.LINE_AA)
    wrapped = textwrap.wrap(transcript_text, width=95)[:2]
    for idx, line in enumerate(wrapped):
        cv2.putText(canvas, line, (left, 108 + 28 * idx), FONT, 0.65, (90, 90, 90), 1, cv2.LINE_AA)

    def x_of(time_sec: float) -> int:
        if duration_sec <= 0:
            return left
        return left + int(round((max(0.0, min(duration_sec, time_sec)) / duration_sec) * line_w))

    for sec in range(int(math.floor(duration_sec)) + 1):
        x = x_of(float(sec))
        cv2.line(canvas, (x, pred_y - 70), (x, pred_y + 80), (220, 220, 220), 1)
        label_y = pred_y + 110 if gt else pred_y + 90
        cv2.putText(canvas, str(sec), (x - 8, label_y), FONT, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    if gt:
        cv2.putText(canvas, "GT", (40, gt_y + 10), FONT, 0.8, (40, 120, 40), 2, cv2.LINE_AA)
        cv2.line(canvas, (left, gt_y), (left + line_w, gt_y), (180, 180, 180), 2)
        for seg in gt["segments"]:
            x1 = x_of(seg["start_sec"])
            x2 = x_of(seg["end_sec"])
            cv2.rectangle(canvas, (x1, gt_y - 16), (max(x1 + 4, x2), gt_y + 16), (100, 190, 100), -1)
            cv2.circle(canvas, (x_of(seg["mid_sec"]), gt_y), 6, (30, 120, 30), -1)
            label = f"gt{seg['segment_idx']} {seg['start_sec']:.2f}-{seg['end_sec']:.2f}s"
            cv2.putText(canvas, label, (x1, gt_y - 24), FONT, 0.48, (40, 100, 40), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Pred", (28, pred_y + 10), FONT, 0.8, (10, 110, 180), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, pred_y), (left + line_w, pred_y), (180, 180, 180), 2)
    for event in prediction["events"]:
        x1 = x_of(event["start_sec"])
        x2 = x_of(event["end_sec"])
        cv2.rectangle(canvas, (x1, pred_y - 20), (max(x1 + 4, x2), pred_y + 20), (0, 180, 255), -1)
        cv2.circle(canvas, (x_of(event["mid_sec"]), pred_y), 6, (0, 90, 180), -1)
        label = f"e{event['pred_idx']} {event['event_type']} @ {event['mid_sec']:.2f}s"
        ty = pred_y - 36 if event["pred_idx"] % 2 == 0 else pred_y + 52
        cv2.putText(canvas, label, (x1, ty), FONT, 0.48, (20, 70, 120), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def attach_match_info(prediction: dict, matched_details: list[dict]):
    by_pred_idx = {item["pred_idx"]: item for item in matched_details}
    for event in prediction["events"]:
        event["matched_gt"] = by_pred_idx.get(event["pred_idx"])


def write_csv(prediction: dict, out_path: Path):
    fieldnames = [
        "pred_idx",
        "event_type",
        "event_description",
        "start_sec",
        "end_sec",
        "mid_sec",
        "mid_frame",
        "primary_label",
        "primary_interaction_type",
        "primary_bbox_xyxy_1000",
        "primary_bbox_px",
        "primary_interaction_score",
        "primary_confidence",
        "matched_gt_idx",
        "matched_iou",
        "matched_midpoint_error_sec",
        "frame_path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in prediction["events"]:
            primary = select_primary_object(event["midpoint_objects"])
            matched = event.get("matched_gt") or {}
            writer.writerow(
                {
                    "pred_idx": event["pred_idx"],
                    "event_type": event["event_type"],
                    "event_description": event["event_description"],
                    "start_sec": event["start_sec"],
                    "end_sec": event["end_sec"],
                    "mid_sec": event["mid_sec"],
                    "mid_frame": event["mid_frame"],
                    "primary_label": primary["label"] if primary else "",
                    "primary_interaction_type": primary.get("interaction_type") if primary else "",
                    "primary_bbox_xyxy_1000": json.dumps(primary["bbox_xyxy_1000"]) if primary else "",
                    "primary_bbox_px": json.dumps(primary.get("bbox_px")) if primary else "",
                    "primary_interaction_score": primary.get("interaction_score") if primary else "",
                    "primary_confidence": primary.get("confidence") if primary else "",
                    "matched_gt_idx": matched.get("gt_idx", ""),
                    "matched_iou": matched.get("iou", ""),
                    "matched_midpoint_error_sec": matched.get("midpoint_error_sec", ""),
                    "frame_path": event.get("midpoint_frame_path", ""),
                }
            )


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    video_stem = Path(args.video_path).stem
    tag = f"task_intervention_event_bboxes_{video_stem}_{args.model}_bbox_{args.bbox_model}"
    if args.exp_suffix:
        tag += f"_{_safe_name(args.exp_suffix)}"
    return Path("results") / tag


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    transcript = json.loads(Path(args.transcript_path).read_text())
    transcript_text = str(transcript.get("text", "")).strip()
    if not transcript_text:
        raise ValueError(f"missing transcript text in {args.transcript_path}")

    meta = read_video_meta(args.video_path)
    with open(args.video_path, "rb") as f:
        video_bytes = f.read()

    out_dir = build_output_dir(args)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    reused_event_source = args.reuse_events_from.strip() or None
    if reused_event_source:
        reused_payload = json.loads(Path(reused_event_source).read_text())
        if "prediction" in reused_payload:
            prediction = reused_payload["prediction"]
        else:
            prediction = reused_payload
        if not isinstance(prediction, dict) or not isinstance(prediction.get("events"), list):
            raise ValueError(f"invalid reused prediction payload: {reused_event_source}")
        response = None
        infer_time = 0.0
        cost = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cost_usd": 0.0}
    else:
        prompt = build_prompt(transcript_text, args.max_events)
        infer_start = time.perf_counter()
        response = call_gemini(
            client,
            args.model,
            video_bytes,
            prompt,
            video_fps=args.video_fps,
            thinking_budget=args.thinking_budget,
        )
        infer_time = time.perf_counter() - infer_start
        cost = extract_cost(response, args.model)
        pred = parse_response(response, args.max_events)
        if pred is None:
            raise RuntimeError(f"failed to parse model response:\n{response.text}")
        prediction = normalize_prediction(pred, meta)
    bbox_refine_summary = refine_midpoint_boxes(
        client=client,
        args=args,
        video_path=args.video_path,
        transcript=transcript,
        prediction=prediction,
        meta=meta,
    )

    gt = None
    metrics = None
    matched_details = []
    if args.episode_dir:
        gt = load_ground_truth_multi(args.episode_dir)
        if gt is not None:
            metrics, matched_details = evaluate_episode_multi(
                prediction,
                gt,
                match_frame_tolerance=args.match_frame_tolerance,
            )
            attach_match_info(prediction, matched_details)

    frame_rows = save_midpoint_frames(args.video_path, prediction, frame_dir)
    frame_row_map = {row["pred_idx"]: row for row in frame_rows}
    for event in prediction["events"]:
        frame_row = frame_row_map.get(event["pred_idx"])
        if frame_row is None:
            continue
        event["midpoint_frame_path"] = frame_row["frame_path"]
        for obj in event["midpoint_objects"]:
            for saved_obj in frame_row["objects"]:
                if obj["label"] == saved_obj["label"] and obj["bbox_xyxy_1000"] == saved_obj["bbox_xyxy_1000"]:
                    obj["bbox_px"] = saved_obj["bbox_px"]
                    break

    contact_sheet_path = out_dir / "contact_sheet.jpg"
    build_contact_sheet(frame_rows, contact_sheet_path)

    timeline_path = out_dir / "timeline.png"
    build_timeline_image(prediction, meta["duration_sec"], timeline_path, transcript_text, gt=gt)

    csv_path = out_dir / "events.csv"
    write_csv(prediction, csv_path)

    primary_rows = []
    for event in prediction["events"]:
        primary = select_primary_object(event["midpoint_objects"])
        primary_rows.append(
            {
                "pred_idx": event["pred_idx"],
                "event_type": event["event_type"],
                "event_description": event["event_description"],
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "mid_sec": event["mid_sec"],
                "mid_frame": event["mid_frame"],
                "primary_object": primary,
                "frame_path": event.get("midpoint_frame_path"),
                "matched_gt": event.get("matched_gt"),
            }
        )

    result = {
        "video_path": args.video_path,
        "transcript_path": args.transcript_path,
        "episode_dir": args.episode_dir or None,
        "model": args.model,
        "bbox_model": args.bbox_model,
        "video_fps_for_model": args.video_fps,
        "bbox_video_fps": args.bbox_video_fps,
        "thinking_budget": args.thinking_budget,
        "bbox_thinking_budget": args.bbox_thinking_budget,
        "max_events": args.max_events,
        "reuse_events_from": reused_event_source,
        "duration_sec": round(meta["duration_sec"], 4),
        "video_meta": meta,
        "prediction": prediction,
        "primary_event_rows": primary_rows,
        "bbox_refinement_summary": bbox_refine_summary,
        "metrics": metrics,
        "matched_details": matched_details,
        "cost": cost,
        "inference_time_sec": round(infer_time, 4),
        "contact_sheet_path": str(contact_sheet_path),
        "timeline_path": str(timeline_path),
        "events_csv_path": str(csv_path),
        "raw_response": response.text if response is not None else None,
    }
    json_path = out_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2))

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {timeline_path}")
    print(f"[saved] {contact_sheet_path}")
    print(f"[info] events={len(prediction['events'])} infer_time={infer_time:.2f}s cost_usd={cost.get('cost_usd', 0.0):.6f}")
    if metrics is not None:
        print(
            "[metrics] "
            f"pred={metrics['pred_interval_count']} gt={metrics['gt_interval_count']} "
            f"mid_hit={metrics['pred_midpoint_hit_rate']:.4f} "
            f"seg_f1={metrics['segment_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
