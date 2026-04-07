#!/usr/bin/env python3
"""
Gemini VLM evaluation for task_intervention control-information events.

This evaluator is specific to task_intervention* episodes, where the goal is
to detect the moments when the human gives actionable control information to
the robot. It predicts multiple event intervals and a short event label for
each interval, then evaluates the interval count and midpoint hit rate against
the labeled ground-truth segments.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import statistics
import time

from google import genai

from eval_vlm_baseline import (
    PROMPT_CONFIG_NOTES,
    _save_as_h264,
    append_jsonl,
    call_gemini,
    extract_cost,
    get_config_name,
    load_existing_results,
    prepare_video,
    resolve_gemini_api_key,
)
from eval_vlm_multisegment import (
    MODEL_CHOICES,
    evaluate_episode_multi,
    load_ground_truth_multi,
)


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


PROMPT_TEMPLATE = """\
You are analyzing a first-person human-robot interaction video recorded in a store.
The human wearer gives control information to the robot.

The spoken transcript is:
  "{transcript}"

{config_note}

The original audio track is included. Use both the audio timing and the burned-in
captions to detect when the human gives actionable control information.

Your task: identify all distinct **control-information events** in the video.

A control-information event is a brief moment when the human gives information
that should change, refine, or confirm the robot's behavior. Examples include:
  - selecting or indicating the target object
  - correcting the target ("not that one", "this one")
  - indicating the destination ("here")
  - adjusting motion ("more right", "left", "closer")
  - issuing control commands ("stop", "go")
  - specifying priority or order ("purple first")
  - giving a brief confirmation that still matters for control ("yes, here")

Ignore filler speech, repeated words that do not add new control information,
and passive commentary that does not change what the robot should do.

For each event, return:
  1. a short event_type from this closed set:
     {event_types}
  2. a short event_description that explains the specific control information
  3. a short time interval whose midpoint lands inside the decisive event moment

Important timing rules:
  - Center each interval on the decisive signaling moment itself.
  - Use visible pointing or hand indication when available.
  - Use the audio track and caption timing to anchor when the control message is spoken.
  - If the same message is repeated for emphasis without adding new information,
    treat it as one event, not multiple.
  - Keep intervals short and precise, typically 0.2-2.5 seconds.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return JSON:
{{
  "reasoning": "<brief explanation>",
  "num_events": <int>,
  "events": [
    {{
      "event_type": "<one of the allowed labels>",
      "event_description": "<short description>",
      "start": <float, seconds>,
      "end": <float, seconds>
    }}
  ]
}}

Important:
  - Return between 1 and {max_events} events.
  - num_events must equal len(events).
  - Events must be sorted by time.
  - Events should be non-overlapping.
  - Always return positive start/end values.
  - Never return -1.
"""


def _build_config_note(*, caption: bool, include_audio: bool) -> str:
    config_name = get_config_name(caption, False)
    note = PROMPT_CONFIG_NOTES[config_name]
    if include_audio:
        note += (
            "\nThe original audio track is also present, so use actual speech timing, pauses, "
            "corrections, and emphasis in addition to the visible captions."
        )
    return note


def build_prompt(
    *,
    transcript_text: str,
    duration_sec: float,
    fps: float,
    caption: bool,
    include_audio: bool,
    max_events: int,
) -> str:
    return PROMPT_TEMPLATE.format(
        transcript=transcript_text,
        config_note=_build_config_note(caption=caption, include_audio=include_audio),
        event_types=", ".join(EVENT_TYPES),
        duration=duration_sec,
        fps=fps,
        max_events=max_events,
    )


def _parse_event_item(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    if "start" not in item or "end" not in item:
        return None
    try:
        start = float(item["start"])
        end = float(item["end"])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if start > end:
        start, end = end, start
    if start < 0 or end < 0:
        return None

    event_type = str(item.get("event_type", "other_control")).strip() or "other_control"
    if event_type not in EVENT_TYPES:
        event_type = "other_control"

    description = str(
        item.get("event_description", item.get("description", item.get("label", "")))
    ).strip()
    return {
        "event_type": event_type,
        "event_description": description,
        "start_sec": start,
        "end_sec": end,
    }


def parse_intervention_response(response, max_events: int) -> dict | None:
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None

    reasoning = ""
    num_events = None
    raw_events = None

    if isinstance(payload, dict):
        reasoning = str(payload.get("reasoning", ""))
        if isinstance(payload.get("num_events"), int):
            num_events = int(payload["num_events"])
        events = payload.get("events")
        if isinstance(events, list):
            raw_events = events
        else:
            intervals = payload.get("keyframe_intervals")
            if isinstance(intervals, list):
                raw_events = intervals
    elif isinstance(payload, list):
        raw_events = payload

    if not isinstance(raw_events, list):
        return None

    events = []
    for item in raw_events:
        parsed = _parse_event_item(item)
        if parsed is not None:
            events.append(parsed)

    events.sort(key=lambda event: (event["start_sec"], event["end_sec"]))
    raw_count = len(events)
    if max_events > 0:
        events = events[:max_events]

    if not events:
        return None

    return {
        "reasoning": reasoning,
        "num_events": num_events,
        "events": events,
        "raw_event_count": raw_count,
        "truncated_event_count": max(0, raw_count - len(events)),
        "intervals": [
            {"start_sec": event["start_sec"], "end_sec": event["end_sec"]}
            for event in events
        ],
    }


def normalize_events(pred: dict, gt: dict) -> dict:
    fps = gt["fps"]
    duration_sec = gt["duration_sec"]
    total_frames = gt["total_frames"]

    normalized = []
    for idx, event in enumerate(pred["events"]):
        start_sec = max(0.0, min(float(event["start_sec"]), duration_sec))
        end_sec = max(0.0, min(float(event["end_sec"]), duration_sec))
        if start_sec > end_sec:
            start_sec, end_sec = end_sec, start_sec

        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        start_frame = max(0, min(total_frames - 1, start_frame))
        end_frame = max(0, min(total_frames - 1, end_frame))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        mid_sec = (start_sec + end_sec) / 2
        mid_frame = int(round(mid_sec * fps))
        mid_frame = max(0, min(total_frames - 1, mid_frame))

        normalized.append(
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
                "duration_sec": round(end_sec - start_sec, 4),
                "duration_frames": end_frame - start_frame + 1,
            }
        )

    normalized.sort(key=lambda event: (event["start_frame"], event["end_frame"], event["pred_idx"]))
    return {
        **pred,
        "events": normalized,
        "intervals": [
            {
                "pred_idx": event["pred_idx"],
                "start_sec": event["start_sec"],
                "end_sec": event["end_sec"],
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "mid_sec": event["mid_sec"],
                "mid_frame": event["mid_frame"],
                "duration_sec": event["duration_sec"],
                "duration_frames": event["duration_frames"],
            }
            for event in normalized
        ],
    }


def _ordered_midpoint_metrics(pred_events: list[dict], gt_segments: list[dict]) -> dict:
    hits = 0
    compare_count = min(len(pred_events), len(gt_segments))
    for idx in range(compare_count):
        pred_mid = pred_events[idx]["mid_frame"]
        gt = gt_segments[idx]
        if gt["start_frame"] <= pred_mid <= gt["end_frame"]:
            hits += 1

    gt_count = len(gt_segments)
    ordered_recall = hits / gt_count if gt_count else 0.0
    ordered_all_hit = (
        gt_count > 0 and len(pred_events) == gt_count and hits == gt_count
    )
    return {
        "ordered_midpoint_hit_count": hits,
        "ordered_midpoint_gt_count": gt_count,
        "ordered_midpoint_recall": round(ordered_recall, 4),
        "ordered_all_hit_episode": ordered_all_hit,
    }


def _serialize_gt(gt: dict) -> dict:
    return {
        "fps": gt["fps"],
        "total_frames": gt["total_frames"],
        "duration_sec": round(gt["duration_sec"], 4),
        "segments": gt["segments"],
    }


def process_episode(
    *,
    index: int,
    episode_name: str,
    dataset_dir: str,
    api_key: str,
    model: str,
    video_fps: int,
    thinking_budget: int,
    caption: bool,
    include_audio: bool,
    audio_dir: str,
    max_events: int,
    target_resolution: tuple[int, int] | None,
    match_frame_tolerance: int,
) -> dict:
    client = genai.Client(api_key=api_key)
    episode_dir = os.path.join(dataset_dir, episode_name)
    if not os.path.isdir(episode_dir):
        return {
            "episode_name": episode_name,
            "success": False,
            "error": "missing episode directory",
            "metrics": None,
        }

    gt = load_ground_truth_multi(episode_dir)
    if gt is None:
        return {
            "episode_name": episode_name,
            "success": False,
            "error": "missing ground truth",
            "metrics": None,
        }

    transcript_path = os.path.join(episode_dir, "transcript.json")
    video_path = os.path.join(episode_dir, "video.mp4")
    if not os.path.exists(transcript_path) or not os.path.exists(video_path):
        return {
            "episode_name": episode_name,
            "success": False,
            "error": "missing transcript or video",
            "metrics": None,
        }

    with open(transcript_path) as f:
        transcript = json.load(f)
    transcript_text = str(transcript.get("text", "")).strip()

    audio_path = None
    if include_audio:
        candidate_audio = os.path.join(audio_dir, f"{episode_name}_audio.wav")
        if os.path.exists(candidate_audio):
            audio_path = candidate_audio

    prepared_bytes = prepare_video(
        video_path=video_path,
        transcript=transcript,
        caption=caption,
        gaze_annot=False,
        target_resolution=target_resolution,
        gaze_data=None,
        include_audio=include_audio,
        audio_path=audio_path,
    )
    needs_overlay = caption or (target_resolution is not None) or include_audio
    prompt = build_prompt(
        transcript_text=transcript_text,
        duration_sec=gt["duration_sec"],
        fps=gt["fps"],
        caption=caption,
        include_audio=include_audio,
        max_events=max_events,
    )

    infer_start = time.perf_counter()
    response = call_gemini(
        client=client,
        model_name=model,
        video_bytes=prepared_bytes,
        prompt=prompt,
        video_fps=video_fps,
        thinking_budget=thinking_budget,
    )
    infer_time = time.perf_counter() - infer_start

    parsed = parse_intervention_response(response, max_events=max_events)
    cost = extract_cost(response, model)
    if parsed is None:
        return {
            "episode_name": episode_name,
            "success": False,
            "error": "parse failure",
            "raw_response": response.text,
            "cost": cost,
            "inference_time_sec": round(infer_time, 4),
            "metrics": None,
        }

    pred = normalize_events(parsed, gt)
    metrics, matched_details = evaluate_episode_multi(
        pred,
        gt,
        match_frame_tolerance=match_frame_tolerance,
    )
    metrics.update(_ordered_midpoint_metrics(pred["events"], gt["segments"]))

    if needs_overlay:
        debug_video_size = len(prepared_bytes)
    else:
        debug_video_size = len(prepared_bytes)

    return {
        "episode_name": episode_name,
        "success": True,
        "prompt": prompt,
        "prediction": pred,
        "ground_truth": _serialize_gt(gt),
        "metrics": metrics,
        "matched_details": matched_details,
        "cost": cost,
        "inference_time_sec": round(infer_time, 4),
        "prepared_video_bytes": debug_video_size,
    }


def aggregate_results(results: list[dict]) -> dict:
    successful = [result for result in results if result.get("metrics") is not None]
    agg = {
        "num_episodes": len(results),
        "num_success": len(successful),
        "num_failures": len(results) - len(successful),
    }
    if not successful:
        return agg

    exact_count_hits = sum(1 for result in successful if result["metrics"]["exact_count_match"])
    pred_counts = [result["metrics"]["pred_interval_count"] for result in successful]
    gt_counts = [result["metrics"]["gt_interval_count"] for result in successful]
    raw_mid_rates = [result["metrics"]["pred_midpoint_hit_rate"] for result in successful]
    total_mid_hits = sum(result["metrics"]["pred_midpoint_hit_count"] for result in successful)
    total_pred_intervals = sum(result["metrics"]["pred_interval_count"] for result in successful)
    ordered_hits = sum(result["metrics"]["ordered_midpoint_hit_count"] for result in successful)
    ordered_gt_total = sum(result["metrics"]["ordered_midpoint_gt_count"] for result in successful)
    ordered_all_hit_eps = sum(
        1 for result in successful if result["metrics"]["ordered_all_hit_episode"]
    )
    event_type_counter = Counter()
    for result in successful:
        for event in result["prediction"]["events"]:
            event_type_counter[event["event_type"]] += 1

    total_cost = sum(result.get("cost", {}).get("cost_usd", 0.0) for result in results)
    total_input = sum(result.get("cost", {}).get("input_tokens", 0) for result in results)
    total_output = sum(result.get("cost", {}).get("output_tokens", 0) for result in results)
    total_thinking = sum(result.get("cost", {}).get("thinking_tokens", 0) for result in results)
    infer_times = [result["inference_time_sec"] for result in successful]

    agg.update(
        {
            "exact_count_accuracy": {
                "hits": exact_count_hits,
                "total": len(successful),
                "rate": round(exact_count_hits / len(successful), 4),
            },
            "mean_pred_count": round(statistics.mean(pred_counts), 4),
            "mean_gt_count": round(statistics.mean(gt_counts), 4),
            "mean_abs_count_error": round(
                statistics.mean(result["metrics"]["abs_count_error"] for result in successful),
                4,
            ),
            "raw_mean_midpoint_hit_rate": round(statistics.mean(raw_mid_rates), 4),
            "midpoint_exact_accuracy": round(
                total_mid_hits / total_pred_intervals, 4
            )
            if total_pred_intervals
            else 0.0,
            "ordered_midpoint_recall": round(
                ordered_hits / ordered_gt_total, 4
            )
            if ordered_gt_total
            else 0.0,
            "ordered_all_hit_episode_rate": round(
                ordered_all_hit_eps / len(successful), 4
            ),
            "total_pred_intervals": total_pred_intervals,
            "total_gt_intervals": ordered_gt_total,
            "event_type_counts": dict(sorted(event_type_counter.items())),
            "mean_inference_time_sec": round(statistics.mean(infer_times), 4),
            "median_inference_time_sec": round(statistics.median(infer_times), 4),
            "total_cost_usd": round(total_cost, 4),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_thinking_tokens": total_thinking,
        }
    )
    return agg


def print_summary(agg: dict, model: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Gemini VLM Task Intervention Evaluation Summary")
    print(
        f"  Model: {model}  |  Episodes: {agg['num_episodes']}  "
        f"({agg['num_success']} success, {agg['num_failures']} failures)"
    )
    print(sep)
    if agg["num_success"] == 0:
        print("  No successful episodes to report.")
        print(sep)
        return

    ecc = agg["exact_count_accuracy"]
    print("\n  Event Count:")
    print(f"    Exact count match:      {ecc['rate']*100:5.1f}%  ({ecc['hits']}/{ecc['total']})")
    print(f"    Mean pred / gt count:   {agg['mean_pred_count']:.2f} / {agg['mean_gt_count']:.2f}")
    print(f"    Mean abs count error:   {agg['mean_abs_count_error']:.2f}")

    print("\n  Midpoint:")
    print(f"    Raw mean midpoint hit:  {agg['raw_mean_midpoint_hit_rate']*100:5.1f}%")
    print(f"    Midpoint exact:         {agg['midpoint_exact_accuracy']*100:5.1f}%")
    print(f"    Ordered midpoint recall:{agg['ordered_midpoint_recall']*100:5.1f}%")
    print(f"    Ordered all-hit episode:{agg['ordered_all_hit_episode_rate']*100:5.1f}%")

    print("\n  Runtime / Cost:")
    print(f"    Mean inference time:    {agg['mean_inference_time_sec']:.2f}s")
    print(f"    Median inference time:  {agg['median_inference_time_sec']:.2f}s")
    print(f"    Total cost:             ${agg['total_cost_usd']:.4f}")
    print(f"    Thinking tokens:        {agg['total_thinking_tokens']}")

    print("\n  Predicted Event Types:")
    for event_type, count in agg["event_type_counts"].items():
        print(f"    {event_type}: {count}")
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--audio-dir", default="./preproc_files")
    parser.add_argument("--episodes", default="",
                        help="Comma-separated episode names. Default: all task_intervention*")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", choices=MODEL_CHOICES)
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--match-frame-tolerance", type=int, default=3)
    parser.add_argument("--target-resolution", default="256",
                        help="'none' to preserve resolution, else square size like 256")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exp-suffix", default="")
    return parser.parse_args()


def _resolve_episode_names(dataset_dir: str, episodes_arg: str) -> list[str]:
    if episodes_arg.strip():
        return [name.strip() for name in episodes_arg.split(",") if name.strip()]
    return sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith("task_intervention") and os.path.isdir(os.path.join(dataset_dir, name))
    )


def _parse_target_resolution(value: str) -> tuple[int, int] | None:
    lowered = value.strip().lower()
    if lowered in {"none", "raw", "original"}:
        return None
    size = int(lowered)
    return (size, size)


def main():
    args = parse_args()
    api_key = resolve_gemini_api_key(args.api_key)
    dataset_dir = os.path.abspath(args.dataset_dir)
    audio_dir = os.path.abspath(args.audio_dir)
    target_resolution = _parse_target_resolution(args.target_resolution)
    episode_names = _resolve_episode_names(dataset_dir, args.episodes)

    tag = (
        f"task_intervention_{args.model}_audio_{'true' if args.include_audio else 'false'}"
        f"_cap_{'true' if args.caption else 'false'}"
        f"_fps{args.video_fps}_max{args.max_events}_tb{args.thinking_budget}"
    )
    if args.exp_suffix:
        tag += f"_{args.exp_suffix}"

    eval_dir = os.path.join(os.getcwd(), "eval_results_task_intervention")
    os.makedirs(eval_dir, exist_ok=True)
    jsonl_path = os.path.join(eval_dir, f"{tag}_results.jsonl")
    summary_path = os.path.join(eval_dir, f"{tag}_summary.json")

    completed = set()
    if args.resume and os.path.exists(jsonl_path):
        completed = load_existing_results(jsonl_path)

    print(
        f"[task-intervention] {len(episode_names)} episodes, model={args.model}, "
        f"video_fps={args.video_fps}, audio={'on' if args.include_audio else 'off'}, "
        f"caption={'on' if args.caption else 'off'}, thinking={args.thinking_budget}, "
        f"max_events={args.max_events}, resolution={target_resolution}"
    )

    pending = [name for name in episode_names if name not in completed]
    results_by_name: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_episode = {
            executor.submit(
                process_episode,
                index=index,
                episode_name=episode_name,
                dataset_dir=dataset_dir,
                api_key=api_key,
                model=args.model,
                video_fps=args.video_fps,
                thinking_budget=args.thinking_budget,
                caption=args.caption,
                include_audio=args.include_audio,
                audio_dir=audio_dir,
                max_events=args.max_events,
                target_resolution=target_resolution,
                match_frame_tolerance=args.match_frame_tolerance,
            ): episode_name
            for index, episode_name in enumerate(pending, start=1)
        }

        for future in as_completed(future_to_episode):
            episode_name = future_to_episode[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "episode_name": episode_name,
                    "success": False,
                    "error": f"exception: {exc}",
                    "metrics": None,
                }

            results_by_name[episode_name] = result
            append_jsonl(jsonl_path, result)

            if result.get("metrics") is None:
                print(f"  {episode_name}: FAIL ({result.get('error', 'unknown')})")
            else:
                print(
                    f"  {episode_name}: pred={result['metrics']['pred_interval_count']} "
                    f"gt={result['metrics']['gt_interval_count']} "
                    f"mid={result['metrics']['pred_midpoint_hit_rate']*100:.1f}% "
                    f"time={result['inference_time_sec']:.1f}s"
                )

    ordered_results = [results_by_name[name] for name in pending]
    agg = aggregate_results(ordered_results)
    print_summary(agg, args.model)

    with open(summary_path, "w") as f:
        json.dump(
            {
                "aggregate": agg,
                "config": {
                    "model": args.model,
                    "video_fps": args.video_fps,
                    "thinking_budget": args.thinking_budget,
                    "caption": args.caption,
                    "include_audio": args.include_audio,
                    "max_events": args.max_events,
                    "target_resolution": target_resolution,
                    "dataset_dir": dataset_dir,
                    "audio_dir": audio_dir,
                },
            },
            f,
            indent=2,
        )

    os.makedirs("results", exist_ok=True)
    results_json_name = f"result_{tag}.json"
    results_json_path = os.path.join("results", results_json_name)
    per_episode = []
    for result in ordered_results:
        entry = {
            "episode_name": result["episode_name"],
            "success": result.get("success", False),
            "metrics": result.get("metrics"),
        }
        if result.get("prediction"):
            entry["predicted_events"] = [
                {
                    "event_type": event["event_type"],
                    "event_description": event["event_description"],
                    "interval_sec": [event["start_sec"], event["end_sec"]],
                    "interval_frames": [event["start_frame"], event["end_frame"]],
                    "mid_frame": event["mid_frame"],
                    "mid_sec": event["mid_sec"],
                }
                for event in result["prediction"]["events"]
            ]
            entry["predicted_num_events"] = (
                result["prediction"].get("num_events")
                if result["prediction"].get("num_events") is not None
                else len(result["prediction"]["events"])
            )
            entry["prediction_reasoning"] = result["prediction"].get("reasoning", "")
        if result.get("ground_truth"):
            entry["gt_segments"] = result["ground_truth"]["segments"]
        if result.get("cost"):
            entry["cost"] = result["cost"]
        per_episode.append(entry)

    with open(results_json_path, "w") as f:
        json.dump(
            {
                "aggregate": agg,
                "results": per_episode,
                "config": {
                    "model": args.model,
                    "video_fps": args.video_fps,
                    "thinking_budget": args.thinking_budget,
                    "caption": args.caption,
                    "include_audio": args.include_audio,
                    "max_events": args.max_events,
                },
            },
            f,
            indent=2,
        )

    print(f"[saved] {jsonl_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {results_json_path}")


if __name__ == "__main__":
    main()
