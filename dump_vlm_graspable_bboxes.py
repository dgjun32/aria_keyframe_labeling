#!/usr/bin/env python3
"""
Dump VLM-predicted graspable-object bounding boxes for sampled frames.

This is for debugging and visual inspection. It does not compute accuracy,
because the current dataset does not contain bbox ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

import cv2
from google import genai
from google.genai import types

from eval_vlm_baseline import extract_cost, resolve_gemini_api_key
from eval_vlm_multisegment import MODEL_CHOICES


PROMPTS = {
    "exhaustive": """\
You are looking at a single frame from a first-person store video.

Your task: detect all distinct graspable physical objects that a robot gripper
could plausibly pick up from this frame. Be exhaustive and return as many
visible graspable objects as you can reliably identify.

Include small manipulable retail items such as snacks, cans, bottles, boxes,
and other portable products. Do not include hands, arms, shelves, baskets,
tables, or the background unless they are themselves clear graspable targets.

Return JSON:
{
  "graspable_objects": [
    {
      "label": "<short object label>",
      "bbox_xyxy_1000": [x1, y1, x2, y2],
      "confidence": <float 0.0 to 1.0>
    }
  ]
}

Important:
- bbox coordinates must be integers in [0, 1000]
- use a tight box around the physical object only
- do not emit duplicate boxes for the same visible object
- if no graspable object is visible, return an empty list
""",
    "interaction_high_precision": """\
You are looking at a single frame from a first-person store video.

Your task: return only the clearly visible physical objects that are most likely
to matter for immediate robot interaction, such as likely pick targets or likely
place targets. Prioritize precision over recall.

Keep objects only if they are:
- clearly visible and spatially separable
- realistically reachable or central to the active workspace
- plausible near-term pick/place interaction targets

Exclude objects that are:
- heavily occluded, tiny, ambiguous, reflected, duplicated by repeated patterns,
  or not clearly separable from neighbors
- deep background clutter that is unlikely to be interacted with soon

Include manipulable retail items and clear place targets like a basket or tray
only when they are visually salient and interaction-relevant.

Return JSON:
{
  "graspable_objects": [
    {
      "label": "<short object label>",
      "bbox_xyxy_1000": [x1, y1, x2, y2],
      "confidence": <float 0.0 to 1.0>
    }
  ]
}

Important:
- bbox coordinates must be integers in [0, 1000]
- use a tight box around the physical object only
- output only high-confidence, interaction-relevant objects
- do not emit duplicate boxes for the same visible object
- if you are uncertain, skip the object
- if no clear interaction-relevant object is visible, return an empty list
""",
    "interaction_reachable_sparse": """\
You are looking at a single frame from a first-person store video.

Your task: return only the clearly visible, individually separable objects that
are most likely to matter for near-term robot interaction.

Prioritize objects that are:
- front-most, reachable, and clearly isolated enough for a robot to act on
- likely pick targets or obvious place targets in the active workspace
- visible with clear object boundaries

Be conservative:
- do NOT enumerate every repeated item in a dense row or stack
- do NOT infer hidden or partially visible duplicates
- if several similar objects are tightly packed, keep only the few front-most
  clearly separable instances
- skip uncertain, tiny, background, reflected, or heavily occluded objects

Return JSON:
{
  "graspable_objects": [
    {
      "label": "<short object label>",
      "bbox_xyxy_1000": [x1, y1, x2, y2],
      "confidence": <float 0.0 to 1.0>
    }
  ]
}

Important:
- bbox coordinates must be integers in [0, 1000]
- use a tight box around the physical object only
- output only high-confidence, interaction-relevant objects
- prefer fewer, cleaner boxes over many uncertain boxes
- do not emit duplicate boxes for the same visible object
- if you are uncertain, skip the object
- if no clear interaction-relevant object is visible, return an empty list
""",
    "interaction_candidates": """\
You are looking at a single frame from a first-person store video.

Your task: return only the few objects that are most likely to matter for the
next robot interaction.

Focus on objects that are likely to be:
- a near-term pick target
- a near-term place target

Prioritize precision over recall.

Keep an object only if it is:
- clearly visible and spatially separable
- salient in the active workspace
- plausibly relevant for immediate robot interaction

Do not enumerate every repeated item in dense clutter. If many similar objects
appear together, keep only the most salient clearly separable candidates.

Return JSON:
{
  "interaction_candidates": [
    {
      "label": "<short object label>",
      "bbox_xyxy_1000": [x1, y1, x2, y2],
      "interaction_type": "pick_target" | "place_target" | "other_relevant",
      "interaction_score": <float 0.0 to 1.0>,
      "confidence": <float 0.0 to 1.0>
    }
  ]
}

Important:
- bbox coordinates must be integers in [0, 1000]
- use a tight box around the physical object only
- prefer fewer, cleaner candidates over many uncertain ones
- do not emit duplicate boxes for the same visible object
- if you are uncertain, skip the object
- if no clear interaction-relevant object is visible, return an empty list
""",

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--frame-stride", type=int, default=15,
                        help="Sample every N frames. 15 ~= 1fps for this dataset.")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--frame-idx", default=None, help="single frame index, or 'last'")
    parser.add_argument("--prompt-mode", default="interaction_reachable_sparse", choices=sorted(PROMPTS.keys()))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--exp-suffix", default="")
    return parser.parse_args()


def call_gemini_image(
    *,
    client: genai.Client,
    model_name: str,
    image_bytes: bytes,
    prompt: str,
    thinking_budget: int,
):
    parts = [
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


def encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("failed to encode frame")
    return bytes(buf)


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


def parse_response(response, max_objects: int | None) -> list[dict] | None:
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    objs = payload.get("graspable_objects")
    if not isinstance(objs, list):
        objs = payload.get("interaction_candidates")
    if not isinstance(objs, list):
        return None

    parsed = []
    selected = objs if max_objects is None else objs[:max_objects]
    for item in selected:
        if not isinstance(item, dict):
            continue
        bbox = _parse_bbox(item.get("bbox_xyxy_1000"))
        if bbox is None:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            interaction_score = float(item.get("interaction_score", confidence))
        except (TypeError, ValueError):
            interaction_score = confidence
        parsed.append(
            {
                "label": str(item.get("label", "")).strip() or "object",
                "bbox_xyxy_1000": bbox,
                "confidence": max(0.0, min(1.0, confidence)),
                "interaction_type": str(item.get("interaction_type", "")).strip() or None,
                "interaction_score": max(0.0, min(1.0, interaction_score)),
            }
        )
    return parsed


def bbox_to_pixels(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]


def draw_boxes(frame, objects):
    out = frame.copy()
    for idx, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox_px"]
        color = (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if obj.get("interaction_type"):
            label = (
                f"{obj['label']} {obj['interaction_type']} "
                f"{obj.get('interaction_score', obj['confidence']):.2f}"
            )
        else:
            label = f"{obj['label']} {obj['confidence']:.2f}"
        y = max(18, y1 - 6)
        cv2.rectangle(out, (x1, y - 16), (min(out.shape[1] - 1, x1 + 10 * len(label)), y + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    client = genai.Client(api_key=api_key)

    episode_dir = os.path.join(args.dataset_dir, args.episode)
    video_path = os.path.join(episode_dir, "video.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tag = f"graspable_bbox_{args.episode}_{args.model}_{args.prompt_mode}_stride{args.frame_stride}_tb{args.thinking_budget}"
    if args.exp_suffix:
        tag += f"_{args.exp_suffix}"
    out_dir = Path("results") / tag
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = out_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / args.frame_stride),
        (width, height),
    )

    rows = []
    infer_times = []
    total_cost = 0.0
    total_input = total_output = total_thinking = 0

    if args.frame_idx is not None:
        if str(args.frame_idx).strip().lower() == "last":
            frame_indices = [max(0, total_frames - 1)]
        else:
            frame_indices = [max(0, min(total_frames - 1, int(args.frame_idx)))]
    else:
        frame_indices = list(range(0, total_frames, max(1, args.frame_stride)))
    for sample_idx, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        image_bytes = encode_jpeg(frame)
        infer_start = time.perf_counter()
        response = call_gemini_image(
            client=client,
            model_name=args.model,
            image_bytes=image_bytes,
            prompt=PROMPTS[args.prompt_mode],
            thinking_budget=args.thinking_budget,
        )
        infer_time = time.perf_counter() - infer_start
        infer_times.append(infer_time)
        cost = extract_cost(response, args.model)
        total_cost += cost.get("cost_usd", 0.0)
        total_input += cost.get("input_tokens", 0)
        total_output += cost.get("output_tokens", 0)
        total_thinking += cost.get("thinking_tokens", 0)

        objects = parse_response(response, args.max_objects)
        if objects is None:
            objects = []

        for obj in objects:
            obj["bbox_px"] = bbox_to_pixels(obj["bbox_xyxy_1000"], width, height)

        vis = draw_boxes(frame, objects)
        frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), vis)
        writer.write(vis)

        rows.append(
            {
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps, 4),
                "num_objects": len(objects),
                "objects": [
                    {
                        "label": obj["label"],
                        "confidence": obj["confidence"],
                        "bbox_xyxy_1000": obj["bbox_xyxy_1000"],
                        "bbox_px": obj["bbox_px"],
                        "interaction_type": obj.get("interaction_type"),
                        "interaction_score": obj.get("interaction_score"),
                    }
                    for obj in objects
                ],
                "frame_path": str(frame_path),
                "cost": cost,
                "inference_time_sec": round(infer_time, 4),
                "raw_response": response.text,
            }
        )
        print(
            f"[{sample_idx}/{len(frame_indices)}] frame={frame_idx} "
            f"time={frame_idx/fps:.2f}s objs={len(objects)} infer={infer_time:.2f}s"
        )

    cap.release()
    writer.release()

    result = {
        "episode": args.episode,
        "model": args.model,
        "frame_stride": args.frame_stride,
        "thinking_budget": args.thinking_budget,
        "video_fps": fps,
        "total_frames": total_frames,
        "sampled_frames": len(rows),
        "mean_objects_per_sample": round(statistics.mean([row["num_objects"] for row in rows]), 4) if rows else 0.0,
        "mean_inference_time_sec": round(statistics.mean(infer_times), 4) if infer_times else 0.0,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thinking_tokens": total_thinking,
        "samples": rows,
    }

    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {json_path}")
    print(f"[saved] {overlay_path}")


if __name__ == "__main__":
    main()
