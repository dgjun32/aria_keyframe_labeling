#!/usr/bin/env python3
"""
Gaze-Velocity Representative-Frame Comparison for Multi-Segment VLM Results
===========================================================================

Reads multi-segment VLM predictions from eval_vlm_multisegment.py and compares
two representative-frame selection rules inside each predicted interval:

1. Midpoint frame of the predicted interval
2. Minimum gaze angular-velocity frame inside the predicted interval

No VLM calls are made. This is a pure post-processing step.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import numpy as np

from eval_gaze_refinement import (
    check_frame_hit,
    compute_angular_velocity_deg,
    load_gaze_pitch_yaw,
    select_min_velocity_frame,
)


def load_ground_truth_multi(episode_dir: str) -> dict | None:
    """Load labels.npy and recover all GT keyframe segments."""
    labels_path = os.path.join(episode_dir, "labels.npy")
    if not os.path.exists(labels_path):
        return None

    labels = np.load(labels_path).astype(np.int8)
    total_frames = len(labels)
    if total_frames == 0:
        return None

    diffs = np.diff(labels)
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0] + 1)
    if labels[0] == 1:
        starts = [0] + starts
    if labels[-1] == 1:
        ends = ends + [total_frames]

    if not starts:
        return None

    segments = []
    for idx, (start, end_exclusive) in enumerate(zip(starts, ends)):
        end = int(end_exclusive - 1)
        if end < start:
            continue
        segments.append(
            {
                "segment_idx": idx,
                "start_frame": int(start),
                "end_frame": end,
            }
        )

    gt_kf_set = set(np.where(labels == 1)[0].tolist())
    return {
        "total_frames": total_frames,
        "segments": segments,
        "gt_kf_set": gt_kf_set,
    }


def _parse_interval(interval) -> tuple[int, int] | None:
    if isinstance(interval, dict):
        if "start_frame" in interval and "end_frame" in interval:
            try:
                return int(interval["start_frame"]), int(interval["end_frame"])
            except (TypeError, ValueError):
                return None
    elif isinstance(interval, (list, tuple)) and len(interval) >= 2:
        try:
            return int(interval[0]), int(interval[1])
        except (TypeError, ValueError):
            return None
    return None


def _matched_gt_map(ep_data: dict) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for match in ep_data.get("matches", []):
        try:
            pred_idx = int(match["pred_idx"])
            gt_idx = int(match["gt_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        mapping[pred_idx] = gt_idx
    return mapping


def _make_method_summary(
    frame_eval: dict,
    selected_frame: int,
    nearest_gt_dist: int,
    frame_tolerances: list[int],
    velocity_deg_s: float | None = None,
    ordered_hit: bool | None = None,
) -> dict:
    summary = {
        "selected_frame": selected_frame,
        "hit": frame_eval["exact_hit"],
        "nearest_gt_dist": nearest_gt_dist,
    }
    if velocity_deg_s is not None:
        summary["velocity_deg_s"] = round(velocity_deg_s, 2)
    if ordered_hit is not None:
        summary["ordered_hit"] = ordered_hit
    for k in frame_tolerances:
        summary[f"hit@{k}"] = frame_eval[f"hit@{k}"]
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare midpoint vs min-gaze-velocity selection on multi-segment VLM outputs"
    )
    parser.add_argument(
        "--vlm-results",
        required=True,
        help="Path to results JSON from eval_vlm_multisegment.py",
    )
    parser.add_argument("--dataset-dir", default="./dataset", help="Path to dataset/ directory")
    parser.add_argument("--frame-tolerance", default="0,1,3,5", help="Comma-separated frame tolerance values")
    parser.add_argument("--savgol-window", type=int, default=11, help="Savitzky-Golay filter window length (odd, >=3)")
    parser.add_argument("--savgol-poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    frame_tolerances = [int(token) for token in args.frame_tolerance.split(",")]

    with open(args.vlm_results) as f:
        vlm_data = json.load(f)

    vlm_config = vlm_data.get("config", {})
    vlm_episodes = vlm_data.get("episodes", [])
    print(f"[load] {len(vlm_episodes)} episodes from {args.vlm_results}")
    print(
        f"[config] model={vlm_config.get('model')}, "
        f"config_name={vlm_config.get('config_name')}, "
        f"max_intervals={vlm_config.get('max_intervals')}"
    )
    print(
        f"[gaze] savgol_window={args.savgol_window}, "
        f"savgol_poly={args.savgol_poly}"
    )

    print("\n[phase 1] Precomputing gaze angular velocity...")
    ep_gt: dict[str, dict] = {}
    ep_ang_vel: dict[str, np.ndarray] = {}
    episode_names = [episode["episode"] for episode in vlm_episodes]

    for episode_name in episode_names:
        episode_dir = os.path.join(args.dataset_dir, episode_name)

        gt = load_ground_truth_multi(episode_dir)
        if gt is None:
            continue
        ep_gt[episode_name] = gt

        gaze_data = load_gaze_pitch_yaw(episode_dir)
        if gaze_data is None:
            continue
        pitch, yaw = gaze_data
        ang_vel = compute_angular_velocity_deg(
            pitch,
            yaw,
            15.0,
            args.savgol_window,
            args.savgol_poly,
        )
        ep_ang_vel[episode_name] = ang_vel

    print(
        f"[phase 1] Done: {len(ep_ang_vel)} episodes with gaze velocity, "
        f"{len(ep_gt)} with GT labels"
    )

    print("\n[phase 2] Evaluating...\n")
    refined_episodes = []
    total_episodes = 0
    evaluated_episodes = 0
    total_intervals = 0

    midpoint_hits = 0
    gaze_hits = 0
    midpoint_dists: list[int] = []
    gaze_dists: list[int] = []
    midpoint_hits_by_tol = {k: 0 for k in frame_tolerances}
    gaze_hits_by_tol = {k: 0 for k in frame_tolerances}
    midpoint_all_episode_hits = 0
    gaze_all_episode_hits = 0
    midpoint_any_episode_hits = 0
    gaze_any_episode_hits = 0
    total_gt_segments = 0
    midpoint_gt_segment_hits = 0
    gaze_gt_segment_hits = 0
    midpoint_all_gt_episode_hits = 0
    gaze_all_gt_episode_hits = 0
    midpoint_ordered_gt_hits = 0
    gaze_ordered_gt_hits = 0
    midpoint_ordered_all_episode_hits = 0
    gaze_ordered_all_episode_hits = 0

    for ep_data in vlm_episodes:
        episode_name = ep_data["episode"]
        total_episodes += 1

        if episode_name not in ep_gt:
            print(f"  {episode_name}: SKIP (no GT)")
            continue
        if episode_name not in ep_ang_vel:
            print(f"  {episode_name}: SKIP (no gaze)")
            continue

        predicted_intervals = ep_data.get("predicted_intervals_frames", [])
        if not isinstance(predicted_intervals, list) or not predicted_intervals:
            print(f"  {episode_name}: SKIP (no VLM prediction)")
            continue

        gt = ep_gt[episode_name]
        ang_vel = ep_ang_vel[episode_name]
        total_frames = gt["total_frames"]
        gt_kf_set = gt["gt_kf_set"]
        gt_intervals_frames = [
            [segment["start_frame"], segment["end_frame"]]
            for segment in gt["segments"]
        ]
        matched_gt_lookup = _matched_gt_map(ep_data)

        interval_results = []
        valid_intervals = 0
        episode_midpoint_all_hit = True
        episode_gaze_all_hit = True

        for interval_idx, interval in enumerate(predicted_intervals):
            parsed_interval = _parse_interval(interval)
            if parsed_interval is None:
                continue
            start_frame, end_frame = parsed_interval
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))
            if end_frame < start_frame:
                start_frame, end_frame = end_frame, start_frame

            midpoint_frame = (start_frame + end_frame) // 2
            midpoint_eval = check_frame_hit(
                midpoint_frame,
                gt_kf_set,
                total_frames,
                frame_tolerances,
            )
            gaze_frame, gaze_vel = select_min_velocity_frame(
                ang_vel,
                start_frame,
                end_frame,
                total_frames,
            )
            gaze_eval = check_frame_hit(
                gaze_frame,
                gt_kf_set,
                total_frames,
                frame_tolerances,
            )

            midpoint_hits += int(midpoint_eval["exact_hit"])
            gaze_hits += int(gaze_eval["exact_hit"])
            midpoint_dists.append(midpoint_eval["nearest_gt_dist"])
            gaze_dists.append(gaze_eval["nearest_gt_dist"])
            for k in frame_tolerances:
                midpoint_hits_by_tol[k] += int(midpoint_eval[f"hit@{k}"])
                gaze_hits_by_tol[k] += int(gaze_eval[f"hit@{k}"])

            episode_midpoint_all_hit = episode_midpoint_all_hit and midpoint_eval["exact_hit"]
            episode_gaze_all_hit = episode_gaze_all_hit and gaze_eval["exact_hit"]
            total_intervals += 1
            valid_intervals += 1

            ordered_gt_interval = None
            midpoint_ordered_hit = False
            gaze_ordered_hit = False
            if interval_idx < len(gt["segments"]):
                ordered_segment = gt["segments"][interval_idx]
                ordered_gt_interval = [
                    ordered_segment["start_frame"],
                    ordered_segment["end_frame"],
                ]
                midpoint_ordered_hit = (
                    ordered_segment["start_frame"] <= midpoint_eval["frame"] <= ordered_segment["end_frame"]
                )
                gaze_ordered_hit = (
                    ordered_segment["start_frame"] <= gaze_eval["frame"] <= ordered_segment["end_frame"]
                )

            interval_result = {
                "interval_idx": interval_idx,
                "vlm_interval_frames": [start_frame, end_frame],
                "midpoint": _make_method_summary(
                    midpoint_eval,
                    midpoint_eval["frame"],
                    midpoint_eval["nearest_gt_dist"],
                    frame_tolerances,
                    ordered_hit=midpoint_ordered_hit if ordered_gt_interval is not None else None,
                ),
                "gaze_velocity": _make_method_summary(
                    gaze_eval,
                    gaze_eval["frame"],
                    gaze_eval["nearest_gt_dist"],
                    frame_tolerances,
                    velocity_deg_s=gaze_vel,
                    ordered_hit=gaze_ordered_hit if ordered_gt_interval is not None else None,
                ),
            }
            if ordered_gt_interval is not None:
                interval_result["ordered_gt_interval_frames"] = ordered_gt_interval
            matched_gt_idx = matched_gt_lookup.get(interval_idx)
            if matched_gt_idx is not None and 0 <= matched_gt_idx < len(gt_intervals_frames):
                interval_result["matched_gt_interval_frames"] = gt_intervals_frames[matched_gt_idx]
            interval_results.append(interval_result)

        if valid_intervals == 0:
            print(f"  {episode_name}: SKIP (no valid predicted intervals)")
            continue

        evaluated_episodes += 1
        midpoint_all_episode_hits += int(episode_midpoint_all_hit)
        gaze_all_episode_hits += int(episode_gaze_all_hit)

        midpoint_selected_frames = [
            interval_result["midpoint"]["selected_frame"]
            for interval_result in interval_results
        ]
        gaze_selected_frames = [
            interval_result["gaze_velocity"]["selected_frame"]
            for interval_result in interval_results
        ]

        gt_segment_count = len(gt["segments"])
        total_gt_segments += gt_segment_count

        midpoint_gt_segment_hit_count = sum(
            1
            for segment in gt["segments"]
            if any(
                segment["start_frame"] <= frame <= segment["end_frame"]
                for frame in midpoint_selected_frames
            )
        )
        gaze_gt_segment_hit_count = sum(
            1
            for segment in gt["segments"]
            if any(
                segment["start_frame"] <= frame <= segment["end_frame"]
                for frame in gaze_selected_frames
            )
        )

        midpoint_hit_count = sum(
            1 for interval_result in interval_results if interval_result["midpoint"]["hit"]
        )
        gaze_hit_count = sum(
            1 for interval_result in interval_results if interval_result["gaze_velocity"]["hit"]
        )
        midpoint_any_episode_hits += int(midpoint_hit_count > 0)
        gaze_any_episode_hits += int(gaze_hit_count > 0)
        midpoint_gt_segment_hits += midpoint_gt_segment_hit_count
        gaze_gt_segment_hits += gaze_gt_segment_hit_count
        midpoint_all_gt_episode_hits += int(midpoint_gt_segment_hit_count == gt_segment_count)
        gaze_all_gt_episode_hits += int(gaze_gt_segment_hit_count == gt_segment_count)

        midpoint_ordered_hit_count = sum(
            1
            for interval_result in interval_results
            if interval_result["midpoint"].get("ordered_hit") is True
        )
        gaze_ordered_hit_count = sum(
            1
            for interval_result in interval_results
            if interval_result["gaze_velocity"].get("ordered_hit") is True
        )
        midpoint_ordered_gt_hits += midpoint_ordered_hit_count
        gaze_ordered_gt_hits += gaze_ordered_hit_count
        midpoint_ordered_all_episode_hits += int(
            valid_intervals == gt_segment_count and midpoint_ordered_hit_count == gt_segment_count
        )
        gaze_ordered_all_episode_hits += int(
            valid_intervals == gt_segment_count and gaze_ordered_hit_count == gt_segment_count
        )

        episode_result = {
            "episode": episode_name,
            "gt_intervals_frames": gt_intervals_frames,
            "predicted_intervals_frames": [
                interval_result["vlm_interval_frames"]
                for interval_result in interval_results
            ],
            "intervals": interval_results,
            "episode_summary": {
                "pred_interval_count": valid_intervals,
                "gt_interval_count": gt_segment_count,
                "midpoint_hit_count": midpoint_hit_count,
                "gaze_velocity_hit_count": gaze_hit_count,
                "midpoint_any_hit": midpoint_hit_count > 0,
                "gaze_velocity_any_hit": gaze_hit_count > 0,
                "midpoint_all_hit": episode_midpoint_all_hit,
                "gaze_velocity_all_hit": episode_gaze_all_hit,
                "midpoint_gt_segment_hit_count": midpoint_gt_segment_hit_count,
                "gaze_velocity_gt_segment_hit_count": gaze_gt_segment_hit_count,
                "midpoint_gt_segment_recall": round(
                    midpoint_gt_segment_hit_count / gt_segment_count, 4
                ) if gt_segment_count > 0 else 0.0,
                "gaze_velocity_gt_segment_recall": round(
                    gaze_gt_segment_hit_count / gt_segment_count, 4
                ) if gt_segment_count > 0 else 0.0,
                "midpoint_all_gt_segments_hit": midpoint_gt_segment_hit_count == gt_segment_count,
                "gaze_velocity_all_gt_segments_hit": gaze_gt_segment_hit_count == gt_segment_count,
                "midpoint_ordered_hit_count": midpoint_ordered_hit_count,
                "gaze_velocity_ordered_hit_count": gaze_ordered_hit_count,
                "midpoint_ordered_gt_hit_recall": round(
                    midpoint_ordered_hit_count / gt_segment_count, 4
                ) if gt_segment_count > 0 else 0.0,
                "gaze_velocity_ordered_gt_hit_recall": round(
                    gaze_ordered_hit_count / gt_segment_count, 4
                ) if gt_segment_count > 0 else 0.0,
                "midpoint_ordered_all_hit": (
                    valid_intervals == gt_segment_count
                    and midpoint_ordered_hit_count == gt_segment_count
                ),
                "gaze_velocity_ordered_all_hit": (
                    valid_intervals == gt_segment_count
                    and gaze_ordered_hit_count == gt_segment_count
                ),
            },
        }
        refined_episodes.append(episode_result)

        mid_count = episode_result["episode_summary"]["midpoint_hit_count"]
        gaze_count = episode_result["episode_summary"]["gaze_velocity_hit_count"]
        print(
            f"  {episode_name}: "
            f"mid={mid_count}/{valid_intervals}  "
            f"gaze={gaze_count}/{valid_intervals}  "
            f"gt-covered(mid/gaze)="
            f"{midpoint_gt_segment_hit_count}/{gt_segment_count},"
            f"{gaze_gt_segment_hit_count}/{gt_segment_count}  "
            f"pred_intervals={valid_intervals}"
        )
        if args.verbose:
            for interval_result in interval_results:
                print(
                    f"    interval#{interval_result['interval_idx']}: "
                    f"pred={interval_result['vlm_interval_frames']}  "
                    f"mid=f{interval_result['midpoint']['selected_frame']} "
                    f"({'HIT' if interval_result['midpoint']['hit'] else 'MISS'})  "
                    f"gaze=f{interval_result['gaze_velocity']['selected_frame']} "
                    f"({'HIT' if interval_result['gaze_velocity']['hit'] else 'MISS'})"
                )

    print(f"\n{'='*76}")
    print("  Multi-Segment Gaze Refinement Summary")
    print(f"  Episodes: {total_episodes} total, {evaluated_episodes} evaluated")
    print(f"  Intervals evaluated: {total_intervals}")
    print(f"{'='*76}")

    if total_intervals > 0:
        midpoint_exact = midpoint_hits / total_intervals
        gaze_exact = gaze_hits / total_intervals
        midpoint_mean_dist = statistics.mean(midpoint_dists)
        gaze_mean_dist = statistics.mean(gaze_dists)
        midpoint_median_dist = statistics.median(midpoint_dists)
        gaze_median_dist = statistics.median(gaze_dists)

        print(
            f"\n  {'Metric':<32s} {'Midpoint':>14s}  "
            f"{'Gaze (min vel)':>14s}  {'Delta':>8s}"
        )
        print(f"  {'-'*32} {'-'*14}  {'-'*14}  {'-'*8}")

        exact_delta = (gaze_exact - midpoint_exact) * 100
        sign = "+" if exact_delta >= 0 else ""
        print(
            f"  {'Interval exact hit rate':<32s} "
            f"{midpoint_exact*100:>13.1f}%  "
            f"{gaze_exact*100:>13.1f}%  "
            f"{sign}{exact_delta:>6.1f}%"
        )

        for k in frame_tolerances:
            midpoint_rate = midpoint_hits_by_tol[k] / total_intervals
            gaze_rate = gaze_hits_by_tol[k] / total_intervals
            delta = (gaze_rate - midpoint_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"  {f'Interval hit@{k}':<32s} "
                f"{midpoint_rate*100:>13.1f}%  "
                f"{gaze_rate*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

        if evaluated_episodes > 0:
            midpoint_any_episode_rate = midpoint_any_episode_hits / evaluated_episodes
            gaze_any_episode_rate = gaze_any_episode_hits / evaluated_episodes
            delta = (gaze_any_episode_rate - midpoint_any_episode_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"\n  {'Episode any-hit rate':<32s} "
                f"{midpoint_any_episode_rate*100:>13.1f}%  "
                f"{gaze_any_episode_rate*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

            midpoint_all_episode_rate = midpoint_all_episode_hits / evaluated_episodes
            gaze_all_episode_rate = gaze_all_episode_hits / evaluated_episodes
            delta = (gaze_all_episode_rate - midpoint_all_episode_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"\n  {'Episode all-interval hit rate':<32s} "
                f"{midpoint_all_episode_rate*100:>13.1f}%  "
                f"{gaze_all_episode_rate*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

            midpoint_all_gt_episode_rate = midpoint_all_gt_episode_hits / evaluated_episodes
            gaze_all_gt_episode_rate = gaze_all_gt_episode_hits / evaluated_episodes
            delta = (gaze_all_gt_episode_rate - midpoint_all_gt_episode_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"  {'Episode all-GT-covered rate':<32s} "
                f"{midpoint_all_gt_episode_rate*100:>13.1f}%  "
                f"{gaze_all_gt_episode_rate*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

            midpoint_ordered_all_episode_rate = midpoint_ordered_all_episode_hits / evaluated_episodes
            gaze_ordered_all_episode_rate = gaze_ordered_all_episode_hits / evaluated_episodes
            delta = (gaze_ordered_all_episode_rate - midpoint_ordered_all_episode_rate) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"  {'Episode ordered-all-hit rate':<32s} "
                f"{midpoint_ordered_all_episode_rate*100:>13.1f}%  "
                f"{gaze_ordered_all_episode_rate*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

        if total_gt_segments > 0:
            midpoint_gt_segment_recall = midpoint_gt_segment_hits / total_gt_segments
            gaze_gt_segment_recall = gaze_gt_segment_hits / total_gt_segments
            delta = (gaze_gt_segment_recall - midpoint_gt_segment_recall) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"\n  {'GT-segment hit recall':<32s} "
                f"{midpoint_gt_segment_recall*100:>13.1f}%  "
                f"{gaze_gt_segment_recall*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

            midpoint_ordered_gt_hit_recall = midpoint_ordered_gt_hits / total_gt_segments
            gaze_ordered_gt_hit_recall = gaze_ordered_gt_hits / total_gt_segments
            delta = (gaze_ordered_gt_hit_recall - midpoint_ordered_gt_hit_recall) * 100
            sign = "+" if delta >= 0 else ""
            print(
                f"  {'GT ordered-hit recall':<32s} "
                f"{midpoint_ordered_gt_hit_recall*100:>13.1f}%  "
                f"{gaze_ordered_gt_hit_recall*100:>13.1f}%  "
                f"{sign}{delta:>6.1f}%"
            )

        print(
            f"\n  {'Nearest GT dist (mean)':<32s} "
            f"{midpoint_mean_dist:>13.1f}   {gaze_mean_dist:>13.1f}"
        )
        print(
            f"  {'Nearest GT dist (median)':<32s} "
            f"{midpoint_median_dist:>13.1f}   {gaze_median_dist:>13.1f}"
        )

        aggregate = {
            "num_total_episodes": total_episodes,
            "num_evaluated_episodes": evaluated_episodes,
            "num_evaluated_intervals": total_intervals,
            "midpoint": {
                "exact_accuracy": round(midpoint_exact, 4),
                "mean_nearest_gt_dist": round(midpoint_mean_dist, 2),
                "median_nearest_gt_dist": round(midpoint_median_dist, 2),
                "any_hit_episode_rate": round(
                    midpoint_any_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "all_interval_episode_rate": round(
                    midpoint_all_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "all_gt_covered_episode_rate": round(
                    midpoint_all_gt_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "gt_segment_hit_recall": round(
                    midpoint_gt_segment_hits / total_gt_segments, 4
                ) if total_gt_segments > 0 else 0.0,
                "ordered_all_hit_episode_rate": round(
                    midpoint_ordered_all_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "ordered_gt_hit_recall": round(
                    midpoint_ordered_gt_hits / total_gt_segments, 4
                ) if total_gt_segments > 0 else 0.0,
            },
            "gaze_velocity": {
                "exact_accuracy": round(gaze_exact, 4),
                "mean_nearest_gt_dist": round(gaze_mean_dist, 2),
                "median_nearest_gt_dist": round(gaze_median_dist, 2),
                "any_hit_episode_rate": round(
                    gaze_any_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "all_interval_episode_rate": round(
                    gaze_all_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "all_gt_covered_episode_rate": round(
                    gaze_all_gt_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "gt_segment_hit_recall": round(
                    gaze_gt_segment_hits / total_gt_segments, 4
                ) if total_gt_segments > 0 else 0.0,
                "ordered_all_hit_episode_rate": round(
                    gaze_ordered_all_episode_hits / evaluated_episodes, 4
                ) if evaluated_episodes > 0 else 0.0,
                "ordered_gt_hit_recall": round(
                    gaze_ordered_gt_hits / total_gt_segments, 4
                ) if total_gt_segments > 0 else 0.0,
            },
            "num_total_gt_segments": total_gt_segments,
        }
        for k in frame_tolerances:
            aggregate["midpoint"][f"hit@{k}_rate"] = round(
                midpoint_hits_by_tol[k] / total_intervals,
                4,
            )
            aggregate["gaze_velocity"][f"hit@{k}_rate"] = round(
                gaze_hits_by_tol[k] / total_intervals,
                4,
            )
    else:
        aggregate = {
            "num_total_episodes": total_episodes,
            "num_evaluated_episodes": 0,
            "num_evaluated_intervals": 0,
        }
        print("  No intervals to evaluate.")

    print(f"{'='*76}")

    input_basename = os.path.basename(args.vlm_results)
    output_name = f"refined_{input_basename}"
    output_dir = os.path.dirname(args.vlm_results) or "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    output_data = {
        "source": args.vlm_results,
        "vlm_config": vlm_config,
        "gaze_config": {
            "savgol_window": args.savgol_window,
            "savgol_poly": args.savgol_poly,
        },
        "episodes": refined_episodes,
        "aggregate": aggregate,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
