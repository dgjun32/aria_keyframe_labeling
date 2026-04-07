#!/usr/bin/env python3
"""Measure Gemini video latency as clip duration increases.

This script trims a single episode video into multiple shorter clips,
submits each clip to Gemini with the same prompt, and records latency.
It is intended to isolate model response time as a function of video length.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
from google import genai

from eval_vlm_baseline import (
    PROMPT_CONFIG_NOTES,
    PROMPT_TEMPLATE,
    call_gemini,
    extract_cost,
    get_config_name,
    parse_response,
    prepare_video,
    resolve_gemini_api_key,
)
from eval_vlm_multisegment import build_multi_prompt


MODEL_CHOICES = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]


def _parse_target_resolution(value: str) -> tuple[int, int] | None:
    if value.lower() == "none":
        return None
    try:
        width, height = value.lower().split("x", 1)
        return (int(width), int(height))
    except Exception as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(
            "target resolution must be WIDTHxHEIGHT or 'none'"
        ) from exc


def _video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return frames / fps


def _trim_video(input_path: Path, duration_sec: int) -> Path:
    clipped = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    clipped.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-t",
            str(duration_sec),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-an",
            clipped.name,
        ],
        capture_output=True,
        check=True,
    )
    return Path(clipped.name)


def _load_episode_context(episode_dir: Path) -> tuple[dict, list[dict] | None]:
    transcript_path = episode_dir / "transcript.json"
    gaze_path = episode_dir / "gaze.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript: {transcript_path}")

    with transcript_path.open() as f:
        transcript = json.load(f)

    gaze_data = None
    if gaze_path.exists():
        with gaze_path.open() as f:
            gaze_data = json.load(f)

    return transcript, gaze_data


def _build_prompt(
    prompt_kind: str,
    transcript: dict,
    duration_sec: float,
    fps: float,
    max_intervals: int,
    caption: bool,
    gaze_annot: bool,
    prompt_variant: str,
) -> str:
    config_name = get_config_name(caption, gaze_annot)
    config_note = PROMPT_CONFIG_NOTES[config_name]

    if prompt_kind == "baseline":
        return PROMPT_TEMPLATE.format(
            transcript=transcript["text"],
            config_note=config_note,
            duration=duration_sec,
        )

    return build_multi_prompt(
        transcript=transcript["text"],
        duration_sec=duration_sec,
        fps=fps,
        max_intervals=max_intervals,
        config_note=config_note,
        prompt_variant=prompt_variant,
    )


def _format_table(rows: list[dict]) -> str:
    headers = [
        "sec",
        "size_mb",
        "prepare_s",
        "infer_s",
        "total_s",
        "ok",
        "in_tok",
        "out_tok",
        "think_tok",
        "cost_usd",
    ]
    body = []
    for row in rows:
        body.append(
            [
                str(row["clip_duration_sec"]),
                f"{row['video_size_mb']:.2f}",
                f"{row['prepare_time_sec']:.2f}",
                f"{row['inference_time_sec']:.2f}",
                f"{row['total_time_sec']:.2f}",
                "Y" if row["response_ok"] else "N",
                str(row["input_tokens"]),
                str(row["output_tokens"]),
                str(row["thinking_tokens"]),
                f"{row['cost_usd']:.4f}",
            ]
        )

    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(header), *(len(r[idx]) for r in body)))

    def _fmt(cols: list[str]) -> str:
        return " | ".join(val.ljust(widths[idx]) for idx, val in enumerate(cols))

    lines = [_fmt(headers), "-+-".join("-" * w for w in widths)]
    for row in body:
        lines.append(_fmt(row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Gemini video latency by clip length.")
    parser.add_argument("--episode", required=True, help="Episode directory name under dataset/")
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-kind", choices=["baseline", "multiseg"], default="baseline")
    parser.add_argument("--prompt-variant", default="pointing_subgoal_literal")
    parser.add_argument("--min-seconds", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=10)
    parser.add_argument("--step-seconds", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=-1, help="Gemini thinking budget. Use 0 for minimal thinking.")
    parser.add_argument("--max-intervals", type=int, default=10)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--gaze-annot", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--audio-dir", default="./preproc_files")
    parser.add_argument("--target-resolution", type=_parse_target_resolution, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.min_seconds <= 0 or args.max_seconds < args.min_seconds or args.step_seconds <= 0:
        raise ValueError("Invalid second range")

    episode_dir = Path(args.dataset_dir) / args.episode
    video_path = episode_dir / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    transcript, gaze_data = _load_episode_context(episode_dir)
    original_duration = _video_duration_sec(video_path)
    if original_duration <= 0:
        raise RuntimeError(f"Could not read video duration: {video_path}")

    max_seconds = min(args.max_seconds, int(original_duration))
    if max_seconds < args.min_seconds:
        raise ValueError(
            f"Episode is only {original_duration:.2f}s long; requested min {args.min_seconds}s"
        )

    api_key = resolve_gemini_api_key(args.api_key)
    client = genai.Client(api_key=api_key)

    audio_path = None
    if args.include_audio:
        candidate = Path(args.audio_dir) / f"{args.episode}_audio.wav"
        if candidate.exists():
            audio_path = candidate
        else:
            raise FileNotFoundError(f"Missing audio sidecar: {candidate}")

    rows: list[dict] = []
    print(
        f"[latency] episode={args.episode} model={args.model} prompt={args.prompt_kind} "
        f"range={args.min_seconds}-{max_seconds}s step={args.step_seconds}s"
    )

    for clip_sec in range(args.min_seconds, max_seconds + 1, args.step_seconds):
        clip_path = _trim_video(video_path, clip_sec)
        try:
            prepare_start = time.perf_counter()
            video_bytes = prepare_video(
                str(clip_path),
                caption=args.caption,
                gaze_annot=args.gaze_annot,
                transcript=transcript,
                gaze_data=gaze_data,
                target_resolution=args.target_resolution,
                include_audio=args.include_audio,
                audio_path=str(audio_path) if audio_path else None,
            )
            prepare_time = time.perf_counter() - prepare_start

            prompt = _build_prompt(
                prompt_kind=args.prompt_kind,
                transcript=transcript,
                duration_sec=float(clip_sec),
                fps=args.video_fps,
                max_intervals=args.max_intervals,
                caption=args.caption,
                gaze_annot=args.gaze_annot,
                prompt_variant=args.prompt_variant,
            )

            infer_start = time.perf_counter()
            response = call_gemini(
                client=client,
                model_name=args.model,
                video_bytes=video_bytes,
                prompt=prompt,
                video_fps=args.video_fps,
                thinking_budget=args.thinking_budget,
            )
            infer_time = time.perf_counter() - infer_start

            parsed = parse_response(response) if args.prompt_kind == "baseline" else None
            cost = extract_cost(response, args.model)
            total_time = prepare_time + infer_time

            row = {
                "clip_duration_sec": clip_sec,
                "video_size_mb": len(video_bytes) / (1024 * 1024),
                "prepare_time_sec": round(prepare_time, 4),
                "inference_time_sec": round(infer_time, 4),
                "total_time_sec": round(total_time, 4),
                "response_ok": bool(response.text) if args.prompt_kind == "multiseg" else parsed is not None,
                "input_tokens": cost["input_tokens"],
                "output_tokens": cost["output_tokens"],
                "thinking_tokens": cost["thinking_tokens"],
                "cost_usd": cost["cost_usd"],
                "response_excerpt": (response.text or "")[:200],
            }
            rows.append(row)
            print(
                f"  {clip_sec:2d}s  size={row['video_size_mb']:.2f}MB  "
                f"prepare={row['prepare_time_sec']:.2f}s  infer={row['inference_time_sec']:.2f}s  "
                f"ok={'Y' if row['response_ok'] else 'N'}"
            )
        finally:
            if clip_path.exists():
                os.unlink(clip_path)

    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"vlm_latency_{args.episode}_{args.model}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "episode": args.episode,
        "model": args.model,
        "prompt_kind": args.prompt_kind,
        "prompt_variant": args.prompt_variant if args.prompt_kind == "multiseg" else None,
        "config": {
            "caption": args.caption,
            "gaze_annot": args.gaze_annot,
            "include_audio": args.include_audio,
            "video_fps": args.video_fps,
            "thinking_budget": args.thinking_budget,
            "max_intervals": args.max_intervals,
            "target_resolution": args.target_resolution,
        },
        "rows": rows,
        "summary": {
            "mean_inference_time_sec": round(statistics.mean(r["inference_time_sec"] for r in rows), 4),
            "max_inference_time_sec": round(max(r["inference_time_sec"] for r in rows), 4),
            "min_inference_time_sec": round(min(r["inference_time_sec"] for r in rows), 4),
        },
    }
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("\n" + _format_table(rows))
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
