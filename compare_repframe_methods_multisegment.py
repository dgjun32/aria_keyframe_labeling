#!/usr/bin/env python3
"""
Compare representative-frame selection methods for multi-segment VLM outputs.

The script evaluates several frame-selection rules inside each predicted interval:

1. midpoint
2. min_velocity
3. local_avg_center
4. hybrid_mid_gaze
5. plateau_center
6. equal_bin

It reports interval-level exact hit, hit@k, episode-level all-interval hit, and
ordered GT recall for each method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eval_gaze_refinement import (
    check_frame_hit,
    compute_angular_velocity_deg,
    load_gaze_pitch_yaw,
    select_min_velocity_frame,
)
from eval_gaze_refinement_multisegment import load_ground_truth_multi

VIDEO_FPS = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare representative-frame selection methods on multi-segment outputs."
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        required=True,
        help="One or more result_multiseg JSON files.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="./dataset",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--frame-tolerance",
        default="0,1,3,5",
        help="Comma-separated frame tolerance values.",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=11,
        help="Savitzky-Golay filter window length. Use 1 for no smoothing.",
    )
    parser.add_argument(
        "--savgol-poly",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order.",
    )
    parser.add_argument(
        "--local-radius",
        type=int,
        default=2,
        help="Half-window radius for local average velocity.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.6,
        help="Fraction of the interval kept around the center for local_avg_center.",
    )
    parser.add_argument(
        "--hybrid-lambda",
        type=float,
        default=0.75,
        help="Center penalty weight for the hybrid method.",
    )
    parser.add_argument(
        "--plateau-quantile",
        type=float,
        default=0.25,
        help="Quantile threshold used to define low-velocity plateaus.",
    )
    parser.add_argument(
        "--bin-count",
        type=int,
        default=5,
        help="Number of bins used by the equal-bin method.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save detailed JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save aggregate CSV output.",
    )
    return parser.parse_args()


def parse_interval(interval: object) -> tuple[int, int] | None:
    if isinstance(interval, dict) and "start_frame" in interval and "end_frame" in interval:
        try:
            return int(interval["start_frame"]), int(interval["end_frame"])
        except (TypeError, ValueError):
            return None
    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
        try:
            return int(interval[0]), int(interval[1])
        except (TypeError, ValueError):
            return None
    return None


def clamp_interval(start_frame: int, end_frame: int, total_frames: int) -> tuple[int, int]:
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return start_frame, end_frame


def interval_frames(start_frame: int, end_frame: int) -> np.ndarray:
    return np.arange(start_frame, end_frame + 1, dtype=np.int32)


def midpoint_frame(start_frame: int, end_frame: int) -> int:
    return int((start_frame + end_frame) // 2)


def compute_local_average(
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = interval_frames(start_frame, end_frame)
    if len(frames) == 0:
        return frames, np.array([], dtype=np.float64)

    local_avg = np.full(len(frames), np.inf, dtype=np.float64)
    for idx, frame in enumerate(frames):
        lo = max(start_frame, frame - radius)
        hi = min(end_frame, frame + radius)
        window = angular_velocity[lo : hi + 1]
        finite = window[np.isfinite(window)]
        if finite.size > 0:
            local_avg[idx] = float(np.mean(finite))
    return frames, local_avg


def tie_break_by_center(frames: np.ndarray, midpoint: float, values: np.ndarray, prefer_max: bool) -> int:
    if len(frames) == 0:
        return int(round(midpoint))
    target = np.nanmax(values) if prefer_max else np.nanmin(values)
    if not np.isfinite(target):
        distances = np.abs(frames - midpoint)
        return int(frames[np.argmin(distances)])

    if prefer_max:
        mask = np.isclose(values, target, equal_nan=False)
    else:
        mask = np.isclose(values, target, equal_nan=False)

    candidates = frames[mask]
    if len(candidates) == 0:
        distances = np.abs(frames - midpoint)
        return int(frames[np.argmin(distances)])

    distances = np.abs(candidates - midpoint)
    return int(candidates[np.argmin(distances)])


def select_local_avg_center_frame(
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    radius: int,
    center_fraction: float,
) -> tuple[int, float]:
    frames, local_avg = compute_local_average(angular_velocity, start_frame, end_frame, radius)
    length = len(frames)
    if length == 0:
        return start_frame, float("nan")

    keep_count = max(1, int(math.ceil(length * center_fraction)))
    trim = max(0, (length - keep_count) // 2)
    center_frames = frames[trim : trim + keep_count]
    center_avg = local_avg[trim : trim + keep_count]
    mid = (start_frame + end_frame) / 2.0
    frame = tie_break_by_center(center_frames, mid, center_avg, prefer_max=False)
    value = float(local_avg[frames.tolist().index(frame)])
    return frame, value


def select_hybrid_frame(
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    radius: int,
    hybrid_lambda: float,
) -> tuple[int, float]:
    frames, local_avg = compute_local_average(angular_velocity, start_frame, end_frame, radius)
    if len(frames) == 0:
        return start_frame, float("nan")

    finite = local_avg[np.isfinite(local_avg)]
    if finite.size == 0:
        return midpoint_frame(start_frame, end_frame), float("nan")

    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std < 1e-8:
        z_score = np.zeros_like(local_avg)
    else:
        z_score = (local_avg - mean) / std

    mid = (start_frame + end_frame) / 2.0
    interval_length = max(1.0, end_frame - start_frame + 1)
    center_penalty = np.abs(frames - mid) / interval_length
    score = -z_score - (hybrid_lambda * center_penalty)
    frame = tie_break_by_center(frames, mid, score, prefer_max=True)
    value = float(local_avg[frames.tolist().index(frame)])
    return frame, value


def contiguous_runs(frames: np.ndarray) -> list[np.ndarray]:
    if len(frames) == 0:
        return []
    runs: list[list[int]] = [[int(frames[0])]]
    for frame in frames[1:]:
        frame = int(frame)
        if frame == runs[-1][-1] + 1:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return [np.array(run, dtype=np.int32) for run in runs]


def select_plateau_center_frame(
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    radius: int,
    plateau_quantile: float,
) -> tuple[int, float]:
    frames, local_avg = compute_local_average(angular_velocity, start_frame, end_frame, radius)
    if len(frames) == 0:
        return start_frame, float("nan")

    finite_mask = np.isfinite(local_avg)
    if not finite_mask.any():
        return midpoint_frame(start_frame, end_frame), float("nan")

    finite_avg = local_avg[finite_mask]
    threshold = float(np.quantile(finite_avg, plateau_quantile))
    plateau_frames = frames[np.logical_and(finite_mask, local_avg <= threshold)]
    runs = contiguous_runs(plateau_frames)
    if not runs:
        return select_local_avg_center_frame(
            angular_velocity,
            start_frame,
            end_frame,
            radius,
            center_fraction=1.0,
        )

    mid = (start_frame + end_frame) / 2.0
    best_run = None
    best_key = None
    for run in runs:
        run_values = np.array([local_avg[frames.tolist().index(int(frame))] for frame in run], dtype=np.float64)
        run_center = (int(run[0]) + int(run[-1])) / 2.0
        key = (
            len(run),
            -float(np.mean(run_values)),
            -abs(run_center - mid),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_run = run

    assert best_run is not None
    center = int((int(best_run[0]) + int(best_run[-1])) // 2)
    center_idx = frames.tolist().index(center)
    return center, float(local_avg[center_idx])


def select_equal_bin_frame(
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    radius: int,
    bin_count: int,
) -> tuple[int, float]:
    frames, local_avg = compute_local_average(angular_velocity, start_frame, end_frame, radius)
    if len(frames) == 0:
        return start_frame, float("nan")

    num_bins = max(1, min(bin_count, len(frames)))
    bins = np.array_split(frames, num_bins)
    mid = (start_frame + end_frame) / 2.0

    best_frame = None
    best_value = None
    best_key = None
    for frame_bin in bins:
        idxs = [frames.tolist().index(int(frame)) for frame in frame_bin]
        values = local_avg[idxs]
        finite = values[np.isfinite(values)]
        bin_mean = float(np.mean(finite)) if finite.size > 0 else float("inf")
        bin_center = int((int(frame_bin[0]) + int(frame_bin[-1])) // 2)
        key = (
            -bin_mean,
            -abs(bin_center - mid),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_frame = bin_center
            best_value = bin_mean

    assert best_frame is not None
    return best_frame, float(best_value)


def evaluate_method_frame(
    method_name: str,
    angular_velocity: np.ndarray,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    gt_kf_set: set[int],
    frame_tolerances: list[int],
    args: argparse.Namespace,
) -> tuple[int, float, dict]:
    if method_name == "midpoint":
        frame = midpoint_frame(start_frame, end_frame)
        value = float(angular_velocity[frame]) if np.isfinite(angular_velocity[frame]) else float("nan")
    elif method_name == "min_velocity":
        frame, value = select_min_velocity_frame(
            angular_velocity,
            start_frame,
            end_frame,
            total_frames,
        )
    elif method_name == "local_avg_center":
        frame, value = select_local_avg_center_frame(
            angular_velocity,
            start_frame,
            end_frame,
            args.local_radius,
            args.center_fraction,
        )
    elif method_name == "hybrid_mid_gaze":
        frame, value = select_hybrid_frame(
            angular_velocity,
            start_frame,
            end_frame,
            args.local_radius,
            args.hybrid_lambda,
        )
    elif method_name == "plateau_center":
        frame, value = select_plateau_center_frame(
            angular_velocity,
            start_frame,
            end_frame,
            args.local_radius,
            args.plateau_quantile,
        )
    elif method_name == "equal_bin":
        frame, value = select_equal_bin_frame(
            angular_velocity,
            start_frame,
            end_frame,
            args.local_radius,
            args.bin_count,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    frame_eval = check_frame_hit(frame, gt_kf_set, total_frames, frame_tolerances)
    return frame, value, frame_eval


def summarize_method(
    method_name: str,
    metrics: dict,
    frame_tolerances: list[int],
) -> dict:
    total_intervals = metrics["total_intervals"]
    evaluated_episodes = metrics["evaluated_episodes"]
    total_gt_segments = metrics["total_gt_segments"]

    summary = {
        "method": method_name,
        "num_evaluated_episodes": evaluated_episodes,
        "num_evaluated_intervals": total_intervals,
        "exact_hit_rate": round(metrics["exact_hits"] / total_intervals, 4) if total_intervals > 0 else 0.0,
        "any_hit_episode_rate": round(metrics["any_episode_hits"] / evaluated_episodes, 4) if evaluated_episodes > 0 else 0.0,
        "all_interval_episode_rate": round(metrics["all_episode_hits"] / evaluated_episodes, 4) if evaluated_episodes > 0 else 0.0,
        "ordered_gt_hit_recall": round(metrics["ordered_gt_hits"] / total_gt_segments, 4) if total_gt_segments > 0 else 0.0,
        "ordered_all_hit_episode_rate": round(metrics["ordered_all_episode_hits"] / evaluated_episodes, 4) if evaluated_episodes > 0 else 0.0,
        "mean_nearest_gt_dist": round(float(np.mean(metrics["nearest_dists"])), 2) if metrics["nearest_dists"] else 0.0,
    }
    for tolerance in frame_tolerances:
        summary[f"hit@{tolerance}_rate"] = round(
            metrics["hits_by_tol"][tolerance] / total_intervals,
            4,
        ) if total_intervals > 0 else 0.0
    return summary


def init_metric_bucket(frame_tolerances: list[int]) -> dict:
    return {
        "exact_hits": 0,
        "hits_by_tol": {tolerance: 0 for tolerance in frame_tolerances},
        "nearest_dists": [],
        "any_episode_hits": 0,
        "all_episode_hits": 0,
        "ordered_gt_hits": 0,
        "ordered_all_episode_hits": 0,
        "evaluated_episodes": 0,
        "total_intervals": 0,
        "total_gt_segments": 0,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    frame_tolerances = [int(token) for token in args.frame_tolerance.split(",") if token]

    method_names = [
        "midpoint",
        "min_velocity",
        "local_avg_center",
        "hybrid_mid_gaze",
        "plateau_center",
        "equal_bin",
    ]

    detailed_results = []
    summary_rows = []

    for result_file in args.result_files:
        result_path = Path(result_file)
        payload = json.loads(result_path.read_text())
        config = payload.get("config", {})
        episodes = payload.get("episodes", [])

        metric_buckets = {method_name: init_metric_bucket(frame_tolerances) for method_name in method_names}
        per_episode_results = []

        for ep_data in episodes:
            episode_name = ep_data.get("episode")
            if not episode_name:
                continue

            episode_dir = dataset_dir / episode_name
            gt = load_ground_truth_multi(str(episode_dir))
            gaze_data = load_gaze_pitch_yaw(str(episode_dir))
            predicted_intervals = ep_data.get("predicted_intervals_frames") or []
            if gt is None or gaze_data is None or not predicted_intervals:
                continue

            pitch, yaw = gaze_data
            ang_vel = compute_angular_velocity_deg(
                pitch,
                yaw,
                VIDEO_FPS,
                args.savgol_window,
                args.savgol_poly,
            )

            total_frames = gt["total_frames"]
            gt_kf_set = gt["gt_kf_set"]
            gt_segments = gt["segments"]

            valid_intervals = []
            for interval in predicted_intervals:
                parsed = parse_interval(interval)
                if parsed is None:
                    continue
                valid_intervals.append(clamp_interval(parsed[0], parsed[1], total_frames))

            if not valid_intervals:
                continue

            interval_outputs = {method_name: [] for method_name in method_names}
            for method_name in method_names:
                metric_buckets[method_name]["evaluated_episodes"] += 1
                metric_buckets[method_name]["total_gt_segments"] += len(gt_segments)

            for interval_idx, (start_frame, end_frame) in enumerate(valid_intervals):
                for method_name in method_names:
                    frame, value, frame_eval = evaluate_method_frame(
                        method_name,
                        ang_vel,
                        start_frame,
                        end_frame,
                        total_frames,
                        gt_kf_set,
                        frame_tolerances,
                        args,
                    )
                    ordered_hit = False
                    if interval_idx < len(gt_segments):
                        segment = gt_segments[interval_idx]
                        ordered_hit = segment["start_frame"] <= frame <= segment["end_frame"]

                    interval_outputs[method_name].append(
                        {
                            "interval_idx": interval_idx,
                            "pred_interval_frames": [start_frame, end_frame],
                            "selected_frame": int(frame),
                            "value": None if not np.isfinite(value) else round(float(value), 4),
                            "exact_hit": bool(frame_eval["exact_hit"]),
                            "nearest_gt_dist": int(frame_eval["nearest_gt_dist"]),
                            "ordered_hit": bool(ordered_hit),
                            **{f"hit@{tol}": bool(frame_eval[f"hit@{tol}"]) for tol in frame_tolerances},
                        }
                    )

                    bucket = metric_buckets[method_name]
                    bucket["exact_hits"] += int(frame_eval["exact_hit"])
                    bucket["nearest_dists"].append(int(frame_eval["nearest_gt_dist"]))
                    for tolerance in frame_tolerances:
                        bucket["hits_by_tol"][tolerance] += int(frame_eval[f"hit@{tolerance}"])
                    bucket["total_intervals"] += 1

            for method_name in method_names:
                method_intervals = interval_outputs[method_name]
                exact_hits = sum(1 for item in method_intervals if item["exact_hit"])
                ordered_hits = sum(1 for item in method_intervals if item["ordered_hit"])
                bucket = metric_buckets[method_name]
                bucket["any_episode_hits"] += int(exact_hits > 0)
                bucket["all_episode_hits"] += int(exact_hits == len(method_intervals))
                bucket["ordered_gt_hits"] += ordered_hits
                bucket["ordered_all_episode_hits"] += int(
                    len(method_intervals) == len(gt_segments) and ordered_hits == len(gt_segments)
                )

            per_episode_results.append(
                {
                    "episode": episode_name,
                    "gt_segments": [
                        [segment["start_frame"], segment["end_frame"]]
                        for segment in gt_segments
                    ],
                    "predicted_intervals": [[start, end] for start, end in valid_intervals],
                    "methods": interval_outputs,
                }
            )

        for method_name in method_names:
            summary = summarize_method(method_name, metric_buckets[method_name], frame_tolerances)
            summary.update(
                {
                    "result_file": result_path.name,
                    "fps": int(config.get("video_fps")),
                    "caption": bool(config.get("caption")),
                    "gaze_annot": bool(config.get("gaze_annot")),
                    "model": config.get("model"),
                }
            )
            summary_rows.append(summary)

        detailed_results.append(
            {
                "result_file": result_path.name,
                "config": config,
                "method_params": {
                    "savgol_window": args.savgol_window,
                    "savgol_poly": args.savgol_poly,
                    "local_radius": args.local_radius,
                    "center_fraction": args.center_fraction,
                    "hybrid_lambda": args.hybrid_lambda,
                    "plateau_quantile": args.plateau_quantile,
                    "bin_count": args.bin_count,
                },
                "episodes": per_episode_results,
            }
        )

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(
            {
                "frame_tolerances": frame_tolerances,
                "results": detailed_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values(["fps", "method"]).to_csv(output_csv_path, index=False)

    print(f"Saved detailed JSON to: {output_json_path}")
    print(f"Saved aggregate CSV to: {output_csv_path}")


if __name__ == "__main__":
    main()
