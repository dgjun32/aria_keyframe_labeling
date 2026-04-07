#!/usr/bin/env python3
"""Predict task_intervention keyframes and per-keyframe midpoint bounding boxes in one VLM call."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import statistics
import time

import cv2
from google import genai

from eval_vlm_baseline import call_gemini, extract_cost, prepare_video, resolve_gemini_api_key, append_jsonl
from eval_vlm_multisegment import MODEL_CHOICES, evaluate_episode_multi, load_ground_truth_multi
from eval_vlm_task_intervention import EVENT_TYPES

PROMPT_TEMPLATE = """\
You are analyzing a first-person human-robot interaction video recorded in a store.
The human wearer gives control information to the robot.

The spoken transcript is:
  "{transcript}"

The original audio track is included and on-video captions are visible.
Use the whole video context to identify the decisive control-information events.

A control-information event is a brief moment when the human gives information
that should change, refine, or confirm the robot's behavior.

Your task:
1. identify all distinct control-information keyframe intervals
2. for each interval, predict the important objects visible at that event's
   midpoint frame

For each event, return:
- event_type from: {event_types}
- short event_description
- short [start, end] interval whose midpoint lands on the decisive moment
- midpoint_objects: a few interaction-relevant boxes visible at that midpoint

midpoint_objects should contain only the objects that matter for control at that
moment, such as a likely pick target, a likely place target, or another strongly
relevant object. Prefer a few clean boxes over many uncertain ones.

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
          "interaction_type": "pick_target" | "place_target" | "other_relevant",
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
- midpoint_objects may be empty if no clear interaction-relevant object is visible.
- bbox coordinates must be integers in [0, 1000].
- Do not return human hand, human arm, robot arm, robot gripper, or robot body as objects.
- Do not enumerate every repeated object in dense clutter; keep only the most salient ones.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--audio-dir", default="./audio")
    parser.add_argument("--episodes", default="task_intervention_12")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--match-frame-tolerance", type=int, default=3)
    parser.add_argument("--target-resolution", default="256")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--exp-suffix", default="")
    return parser.parse_args()


def parse_target_resolution(value: str):
    txt = value.strip().lower()
    if txt in {"none", "raw", "original"}:
        return None
    size = int(txt)
    return (size, size)


def build_prompt(transcript: str, max_events: int) -> str:
    return PROMPT_TEMPLATE.format(
        transcript=transcript,
        event_types=", ".join(EVENT_TYPES),
        max_events=max_events,
    )


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
                parsed_objects.append({
                    "label": str(obj.get("label", "")).strip() or "object",
                    "bbox_xyxy_1000": bbox,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "interaction_type": str(obj.get("interaction_type", "")).strip() or None,
                    "interaction_score": max(0.0, min(1.0, interaction_score)),
                })
        parsed.append({
            "event_type": event_type,
            "event_description": str(item.get("event_description", "")).strip(),
            "start_sec": start,
            "end_sec": end,
            "midpoint_objects": parsed_objects,
        })
    parsed.sort(key=lambda e: (e["start_sec"], e["end_sec"]))
    return {"reasoning": str(payload.get("reasoning", "")), "events": parsed} if parsed else None


def normalize_prediction(pred: dict, gt: dict):
    fps = gt["fps"]
    duration_sec = gt["duration_sec"]
    total_frames = gt["total_frames"]
    normalized_events = []
    for idx, event in enumerate(pred["events"]):
        start_sec = max(0.0, min(float(event["start_sec"]), duration_sec))
        end_sec = max(0.0, min(float(event["end_sec"]), duration_sec))
        if start_sec > end_sec:
            start_sec, end_sec = end_sec, start_sec
        start_frame = max(0, min(total_frames - 1, int(round(start_sec * fps))))
        end_frame = max(0, min(total_frames - 1, int(round(end_sec * fps))))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        mid_sec = (start_sec + end_sec) / 2
        mid_frame = max(0, min(total_frames - 1, int(round(mid_sec * fps))))
        normalized_events.append({
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
        })
    return {
        "reasoning": pred.get("reasoning", ""),
        "events": normalized_events,
        "intervals": [
            {
                "pred_idx": e["pred_idx"],
                "start_sec": e["start_sec"],
                "end_sec": e["end_sec"],
                "start_frame": e["start_frame"],
                "end_frame": e["end_frame"],
                "mid_sec": e["mid_sec"],
                "mid_frame": e["mid_frame"],
            }
            for e in normalized_events
        ],
    }


def bbox_to_pixels(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]


def draw_boxes(frame, objects):
    out = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_px"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{obj['label']} {obj.get('interaction_type') or ''} {obj.get('interaction_score', obj['confidence']):.2f}".strip()
        y = max(18, y1 - 6)
        cv2.rectangle(out, (x1, y - 16), (min(out.shape[1] - 1, x1 + 10 * len(label)), y + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out


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
        vis = draw_boxes(frame, objects)
        frame_path = out_dir / f"event_{event['pred_idx']:02d}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), vis)
        rows.append({
            "pred_idx": event["pred_idx"],
            "frame_idx": frame_idx,
            "frame_path": str(frame_path),
            "objects": objects,
        })
    cap.release()
    return rows


def process_episode(args, episode_name: str, api_key: str, target_resolution):
    episode_dir = os.path.join(args.dataset_dir, episode_name)
    gt = load_ground_truth_multi(episode_dir)
    if gt is None:
        return {"episode": episode_name, "success": False, "error": "missing ground truth"}
    video_path = os.path.join(episode_dir, "video.mp4")
    transcript_path = os.path.join(episode_dir, "transcript.json")
    if not os.path.exists(video_path) or not os.path.exists(transcript_path):
        return {"episode": episode_name, "success": False, "error": "missing transcript/video"}
    with open(transcript_path) as f:
        transcript = json.load(f)
    audio_path = None
    if args.include_audio:
        candidate = os.path.join(args.audio_dir, f"{episode_name}_audio.wav")
        if os.path.exists(candidate):
            audio_path = candidate
    video_bytes = prepare_video(
        video_path=video_path,
        transcript=transcript,
        caption=args.caption,
        gaze_annot=False,
        target_resolution=target_resolution,
        gaze_data=None,
        include_audio=args.include_audio,
        audio_path=audio_path,
    )
    prompt = build_prompt(str(transcript.get("text", "")).strip(), args.max_events)
    client = genai.Client(api_key=api_key)
    t0 = time.perf_counter()
    response = call_gemini(client, args.model, video_bytes, prompt, video_fps=args.video_fps, thinking_budget=args.thinking_budget)
    infer_time = time.perf_counter() - t0
    pred = parse_response(response, args.max_events)
    cost = extract_cost(response, args.model)
    if pred is None:
        return {"episode": episode_name, "success": False, "error": "parse failure", "raw_response": response.text, "cost": cost}
    normalized = normalize_prediction(pred, gt)
    metrics, matched = evaluate_episode_multi(normalized, gt, match_frame_tolerance=args.match_frame_tolerance)
    out_dir = Path('results') / f"task_intervention_keyframe_bboxes_{args.model}_{episode_name}"
    if args.exp_suffix:
        out_dir = Path(str(out_dir) + f"_{args.exp_suffix}")
    frame_dir = out_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = save_midpoint_frames(video_path, normalized, frame_dir)
    for event, frame_row in zip(normalized["events"], frame_rows):
        event["midpoint_frame_path"] = frame_row["frame_path"]
        for obj in event["midpoint_objects"]:
            for saved_obj in frame_row["objects"]:
                if obj["label"] == saved_obj["label"] and obj["bbox_xyxy_1000"] == saved_obj["bbox_xyxy_1000"]:
                    obj["bbox_px"] = saved_obj["bbox_px"]
                    break
    return {
        "episode": episode_name,
        "success": True,
        "prediction": normalized,
        "metrics": metrics,
        "matched_details": matched,
        "cost": cost,
        "inference_time_sec": round(infer_time, 4),
        "raw_response": response.text,
        "result_dir": str(out_dir),
    }


def aggregate(results):
    good = [r for r in results if r.get("success") and r.get("metrics")]
    if not good:
        return {"num_episodes": len(results), "num_success": 0, "num_failures": len(results)}
    total_mid_hits = sum(r["metrics"]["pred_midpoint_hit_count"] for r in good)
    total_preds = sum(r["metrics"]["pred_interval_count"] for r in good)
    exact_count = sum(1 for r in good if r["metrics"]["exact_count_match"])
    return {
        "num_episodes": len(results),
        "num_success": len(good),
        "num_failures": len(results) - len(good),
        "exact_count_rate": round(exact_count / len(good), 4),
        "midpoint_exact": round(total_mid_hits / total_preds, 4) if total_preds else 0.0,
        "mean_pred_count": round(statistics.mean(r["metrics"]["pred_interval_count"] for r in good), 4),
        "mean_gt_count": round(statistics.mean(r["metrics"]["gt_interval_count"] for r in good), 4),
        "mean_inference_time_sec": round(statistics.mean(r["inference_time_sec"] for r in good), 4),
        "total_cost_usd": round(sum(r["cost"].get("cost_usd", 0.0) for r in good), 4),
    }


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    target_resolution = parse_target_resolution(args.target_resolution)
    episodes = [e.strip() for e in args.episodes.split(',') if e.strip()]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process_episode, args, ep, api_key, target_resolution): ep for ep in episodes}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if result.get('success'):
                print(f"[ok] {result['episode']}: pred={result['metrics']['pred_interval_count']} gt={result['metrics']['gt_interval_count']} mid={result['metrics']['pred_midpoint_hit_rate']:.3f}")
            else:
                print(f"[fail] {result['episode']}: {result.get('error')}")
    results.sort(key=lambda r: r['episode'])
    agg = aggregate(results)
    tag = f"task_intervention_keyframe_bboxes_{args.model}"
    if args.exp_suffix:
        tag += f"_{args.exp_suffix}"
    out_jsonl = Path('eval_results_task_intervention') / f"{tag}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for row in results:
        append_jsonl(str(out_jsonl), row)
    out_json = Path('results') / f"result_{tag}.json"
    out_json.write_text(json.dumps({"config": vars(args), "aggregate": agg, "results": results}, indent=2))
    print('[saved]', out_jsonl)
    print('[saved]', out_json)


if __name__ == '__main__':
    main()
