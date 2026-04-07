#!/usr/bin/env python3
"""Dump VLM-predicted object bounding boxes from a video in one pass.

Supports both dense graspable-object listing and a final-frame interaction-
candidate mode that uses the whole video/audio/caption context.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import cv2
from google import genai
from google.genai import types

from eval_vlm_baseline import extract_cost, prepare_video, resolve_gemini_api_key
from eval_vlm_multisegment import MODEL_CHOICES


PROMPTS = {
    "graspable_samples": """\
You are analyzing a first-person store video.

Focus only on the exact frames nearest these sampled timestamps (seconds):
{sample_times}

For each sampled timestamp, detect all distinct graspable physical objects that
are visibly present at that moment and that a robot gripper could plausibly pick
up.

Include small manipulable retail items such as snacks, cans, bottles, boxes,
and other portable products.

Do not include hands, arms, shelves, baskets, tables, or the background unless
that thing is itself a clear graspable target.

Return JSON:
{{
  "samples": [
    {{
      "time_sec": <float>,
      "objects": [
        {{
          "label": "<short object label>",
          "bbox_xyxy_1000": [x1, y1, x2, y2],
          "confidence": <float 0.0 to 1.0>
        }}
      ]
    }}
  ]
}}

Important:
- Return one sample entry for every requested timestamp.
- Keep the sample order the same as the requested timestamps.
- There is no fixed limit on object count. Return all visible graspable objects.
- bbox coordinates must be integers in [0, 1000].
- Use a tight box around the physical object only.
- Do not emit duplicate boxes for the same visible object at the same timestamp.
- If no graspable object is visible at a timestamp, return an empty object list.
""",
    "interaction_candidates_final": """\
You are analyzing a first-person store video with audio and on-video captions.

Use the whole video context, spoken instruction, and caption timing to infer
which objects matter for the final interaction state.

Focus only on the final frame nearest this timestamp (seconds):
{sample_times}

For that final moment, return only the few objects most likely to matter for
immediate robot interaction. Prioritize precision over recall.

Keep only clearly visible, spatially separable, interaction-relevant objects.
Typical categories are:
- near-term pick target
- near-term place target
- other relevant object that strongly constrains control

Do not enumerate every repeated object in dense clutter. If many similar items
are visible, keep only the most salient ones that matter for control.

Do not return the human hand, human arm, robot arm, robot gripper, or robot body as candidates.

Return JSON:
{{
  "samples": [
    {{
      "time_sec": <float>,
      "objects": [
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
- Return exactly one sample entry for the requested final timestamp.
- bbox coordinates must be integers in [0, 1000].
- Use a tight box around the physical object only.
- Prefer fewer, cleaner candidates over many uncertain boxes.
- Do not emit duplicate boxes for the same visible object.
- If uncertain, skip the object.
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--audio-dir", default="./audio")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--sample-every-sec", type=float, default=1.0)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--prompt-mode", default="interaction_candidates_final", choices=sorted(PROMPTS.keys()))
    parser.add_argument("--target-resolution", default="none", help="e.g. 256 or WxH; use 'none' to preserve source resolution")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--exp-suffix", default="")
    return parser.parse_args()


def parse_target_resolution(value: str):
    txt = str(value).strip().lower()
    if txt in {"none", "raw", "original", "keep"}:
        return None
    if "x" in txt:
        w, h = txt.split("x", 1)
        return (int(w), int(h))
    size = int(txt)
    return (size, size)


def call_gemini_video(*, client: genai.Client, model_name: str, video_bytes: bytes, prompt: str, video_fps: int, thinking_budget: int):
    parts = [
        types.Part(
            inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=video_fps),
        ),
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


def parse_response(response):
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return None

    parsed_samples = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            time_sec = float(sample.get("time_sec"))
        except (TypeError, ValueError):
            continue
        objects = sample.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        parsed_objects = []
        for item in objects:
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
            parsed_objects.append(
                {
                    "label": str(item.get("label", "")).strip() or "object",
                    "bbox_xyxy_1000": bbox,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "interaction_type": str(item.get("interaction_type", "")).strip() or None,
                    "interaction_score": max(0.0, min(1.0, interaction_score)),
                }
            )
        parsed_samples.append({"time_sec": time_sec, "objects": parsed_objects})
    return parsed_samples


def align_samples(requested_times, parsed_samples):
    if parsed_samples is None:
        return [{"requested_time_sec": t, "predicted_time_sec": None, "objects": []} for t in requested_times]
    if len(parsed_samples) == len(requested_times):
        return [
            {
                "requested_time_sec": req_t,
                "predicted_time_sec": pred.get("time_sec"),
                "objects": pred.get("objects", []),
            }
            for req_t, pred in zip(requested_times, parsed_samples)
        ]
    remaining = list(parsed_samples)
    aligned = []
    for req_t in requested_times:
        if not remaining:
            aligned.append({"requested_time_sec": req_t, "predicted_time_sec": None, "objects": []})
            continue
        best_idx = min(range(len(remaining)), key=lambda i: abs(float(remaining[i].get("time_sec", req_t)) - req_t))
        pred = remaining.pop(best_idx)
        aligned.append({
            "requested_time_sec": req_t,
            "predicted_time_sec": pred.get("time_sec"),
            "objects": pred.get("objects", []),
        })
    return aligned


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
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_px"]
        color = (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if obj.get("interaction_type"):
            label = f"{obj['label']} {obj['interaction_type']} {obj.get('interaction_score', obj['confidence']):.2f}"
        else:
            label = f"{obj['label']} {obj['confidence']:.2f}"
        y = max(18, y1 - 6)
        cv2.rectangle(out, (x1, y - 16), (min(out.shape[1] - 1, x1 + 10 * len(label)), y + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_transcript(episode_dir: str):
    transcript_path = os.path.join(episode_dir, "transcript.json")
    if os.path.exists(transcript_path):
        with open(transcript_path) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    client = genai.Client(api_key=api_key)

    episode_dir = os.path.join(args.dataset_dir, args.episode)
    video_path = os.path.join(episode_dir, "video.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    transcript = load_transcript(episode_dir)
    audio_path = None
    if args.include_audio:
        candidate_audio = os.path.join(args.audio_dir, f"{args.episode}_audio.wav")
        if os.path.exists(candidate_audio):
            audio_path = candidate_audio

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    step = max(0.1, float(args.sample_every_sec))
    if args.final_only:
        sample_times = [round(max(0.0, duration_sec - 1.0 / max(fps, 1.0)), 3)]
    else:
        sample_times = []
        t = 0.0
        while t <= max(0.0, duration_sec - 1e-6):
            sample_times.append(round(t, 3))
            t += step
        if sample_times and sample_times[-1] < duration_sec - 0.25 * step:
            sample_times.append(round(duration_sec, 3))
        elif not sample_times:
            sample_times = [0.0]

    video_bytes = prepare_video(
        video_path,
        caption=args.caption,
        gaze_annot=False,
        transcript=transcript,
        gaze_data=None,
        target_resolution=parse_target_resolution(args.target_resolution),
        include_audio=args.include_audio,
        audio_path=audio_path,
    )

    tag = f"graspable_bbox_video_{args.episode}_{args.model}_{args.prompt_mode}_vf{args.video_fps}_step{str(step).replace('.', 'p')}_tb{args.thinking_budget}"
    if args.caption:
        tag += "_cap"
    if args.include_audio:
        tag += "_audio"
    if args.final_only:
        tag += "_final"
    if args.exp_suffix:
        tag += f"_{args.exp_suffix}"
    out_dir = Path("results") / tag
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPTS[args.prompt_mode].format(sample_times=", ".join(f"{x:.2f}" for x in sample_times))

    infer_start = time.perf_counter()
    response = call_gemini_video(
        client=client,
        model_name=args.model,
        video_bytes=video_bytes,
        prompt=prompt,
        video_fps=args.video_fps,
        thinking_budget=args.thinking_budget,
    )
    infer_time = time.perf_counter() - infer_start
    cost = extract_cost(response, args.model)
    parsed_samples = parse_response(response)
    aligned = align_samples(sample_times, parsed_samples)

    cap = cv2.VideoCapture(video_path)
    overlay_path = out_dir / "overlay.mp4"
    overlay_fps = max(1.0, 1.0 / step)
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), overlay_fps, (width, height))

    rows = []
    for idx, sample in enumerate(aligned, start=1):
        req_t = float(sample["requested_time_sec"])
        frame_idx = min(total_frames - 1, max(0, int(round(req_t * fps)))) if total_frames > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        objects = sample.get("objects", [])
        for obj in objects:
            obj["bbox_px"] = bbox_to_pixels(obj["bbox_xyxy_1000"], width, height)
        vis = draw_boxes(frame, objects)
        frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), vis)
        writer.write(vis)
        rows.append(
            {
                "sample_idx": idx - 1,
                "frame_idx": frame_idx,
                "requested_time_sec": round(req_t, 4),
                "predicted_time_sec": round(float(sample["predicted_time_sec"]), 4) if sample["predicted_time_sec"] is not None else None,
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
            }
        )
        print(f"[{idx}/{len(aligned)}] req_t={req_t:.2f}s frame={frame_idx} objs={len(objects)}")

    cap.release()
    writer.release()

    result = {
        "episode": args.episode,
        "model": args.model,
        "video_fps": args.video_fps,
        "sample_every_sec": step,
        "thinking_budget": args.thinking_budget,
        "caption": bool(args.caption),
        "include_audio": bool(args.include_audio),
        "prompt_mode": args.prompt_mode,
        "final_only": bool(args.final_only),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 4),
        "sampled_moments": len(rows),
        "mean_objects_per_sample": round(statistics.mean([row["num_objects"] for row in rows]), 4) if rows else 0.0,
        "inference_time_sec": round(infer_time, 4),
        "total_cost_usd": cost.get("cost_usd", 0.0),
        "total_input_tokens": cost.get("input_tokens", 0),
        "total_output_tokens": cost.get("output_tokens", 0),
        "total_thinking_tokens": cost.get("thinking_tokens", 0),
        "raw_response": response.text,
        "samples": rows,
    }

    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {json_path}")
    print(f"[saved] {overlay_path}")


if __name__ == "__main__":
    main()
