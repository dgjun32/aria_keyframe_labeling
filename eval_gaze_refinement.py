#!/usr/bin/env python3
"""
Gaze-Velocity Keyframe Refinement (Post-Processing)
====================================================
Reads VLM-predicted keyframe intervals from the results JSON produced by
eval_vlm_baseline.py, then refines each prediction using gaze angular
velocity — selecting the frame with **minimum** angular velocity (fixation)
within the VLM-predicted interval.

No VLM calls are made; this is a pure post-processing step.

Usage:
  python eval_gaze_refinement.py \\
      --vlm-results results/result_gemini-2.5-pro_gaze_false_cap_false_fps2.json

  python eval_gaze_refinement.py \\
      --vlm-results results/result_gemini-2.5-pro_gaze_true_cap_true_fps2.json \\
      --savgol-window 15 --savgol-poly 3

Output:
  results/refined_{original_filename}
  with per-episode: mid_frame_hit, gaze_frame_hit, both evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import numpy as np
from scipy.signal import savgol_filter

# ── constants ─────────────────────────────────────────────────────────────
VIDEO_FPS = 15  # ground-truth frame rate


# ══════════════════════════════════════════════════════════════════════════
#  GAZE ANGULAR VELOCITY (same logic as eval_vlm_gaze_velocity.py)
# ══════════════════════════════════════════════════════════════════════════
def load_gaze_pitch_yaw(episode_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pitch and yaw arrays (radians) from gaze.json."""
    gaze_path = os.path.join(episode_dir, "gaze.json")
    if not os.path.exists(gaze_path):
        return None

    with open(gaze_path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames or "pitch" not in frames[0]:
        return None

    pitch = np.array([fr["pitch"] for fr in frames], dtype=np.float64)
    yaw = np.array([fr["yaw"] for fr in frames], dtype=np.float64)
    return pitch, yaw


def compute_angular_velocity_deg(
    pitch: np.ndarray,
    yaw: np.ndarray,
    fps: float,
    savgol_window: int = 11,
    savgol_poly: int = 2,
) -> np.ndarray:
    """Compute gaze angular velocity in deg/s from pitch/yaw (radians).

    Pipeline:
      1. Savitzky-Golay smooth pitch & yaw
      2. Convert to 3D unit vectors on the unit sphere
      3. Angular distance between consecutive frames
      4. Multiply by fps -> deg/s
    """
    T = len(pitch)

    win = savgol_window
    if win % 2 == 0:
        win += 1
    if T < win:
        win = max(T | 1, 3)
    if savgol_poly >= win:
        savgol_poly = win - 1

    pitch_s = savgol_filter(pitch, win, savgol_poly)
    yaw_s = savgol_filter(yaw, win, savgol_poly)

    x = np.cos(pitch_s) * np.cos(yaw_s)
    y = np.cos(pitch_s) * np.sin(yaw_s)
    z = np.sin(pitch_s)

    dot = np.clip(
        x[:-1] * x[1:] + y[:-1] * y[1:] + z[:-1] * z[1:],
        -1.0, 1.0,
    )
    angular_dist = np.arccos(dot) * fps  # rad/s
    angular_velocity = np.concatenate([[0.0], angular_dist])
    return np.rad2deg(angular_velocity)


def select_min_velocity_frame(
    ang_vel: np.ndarray,
    start_frame: int,
    end_frame: int,
    total_frames: int,
) -> tuple[int, float]:
    """Select frame with minimum angular velocity within [start, end].

    Returns (frame_index, velocity_at_frame).
    """
    lo = max(0, start_frame)
    hi = min(total_frames - 1, end_frame)
    if lo > hi:
        lo, hi = hi, lo

    segment = ang_vel[lo: hi + 1]
    local_argmin = int(np.argmin(segment))
    frame_idx = lo + local_argmin
    return frame_idx, float(ang_vel[frame_idx])


# ══════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════
def load_ground_truth(episode_dir: str) -> dict | None:
    """Load labels.npy -> extract GT keyframe set."""
    labels_path = os.path.join(episode_dir, "labels.npy")
    if not os.path.exists(labels_path):
        return None

    labels = np.load(labels_path).astype(np.int8)
    T = len(labels)

    diffs = np.diff(labels)
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0] + 1)
    if labels[0] == 1:
        starts = [0] + starts
    if labels[-1] == 1:
        ends = ends + [T]

    if not starts:
        return None

    kf_start_frame = int(starts[0])
    kf_end_frame = int(ends[0] - 1)
    gt_kf_set = set(np.where(labels == 1)[0].tolist())

    return {
        "total_frames": T,
        "kf_start_frame": kf_start_frame,
        "kf_end_frame": kf_end_frame,
        "gt_kf_set": gt_kf_set,
    }


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════
def check_frame_hit(
    frame: int,
    gt_kf_set: set[int],
    total_frames: int,
    frame_tolerances: list[int],
) -> dict:
    """Check if a selected frame hits GT keyframes at various tolerances."""
    frame = max(0, min(total_frames - 1, frame))
    exact_hit = frame in gt_kf_set

    hits = {}
    for k in frame_tolerances:
        lo = max(0, frame - k)
        hi = min(total_frames - 1, frame + k)
        hits[f"hit@{k}"] = any(f in gt_kf_set for f in range(lo, hi + 1))

    nearest = min(abs(frame - gf) for gf in gt_kf_set) if gt_kf_set else total_frames

    return {
        "frame": frame,
        "exact_hit": exact_hit,
        "nearest_gt_dist": nearest,
        **hits,
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Gaze-velocity keyframe refinement (post-processing)"
    )
    parser.add_argument("--vlm-results", required=True,
                        help="Path to results JSON from eval_vlm_baseline.py "
                             "(e.g. results/result_gemini-2.5-pro_gaze_false"
                             "_cap_false_fps2.json)")
    parser.add_argument("--dataset-dir", default="./dataset",
                        help="Path to dataset/ directory")
    parser.add_argument("--frame-tolerance", default="0,1,3,5",
                        help="Comma-separated frame tolerance values")
    parser.add_argument("--savgol-window", type=int, default=11,
                        help="Savitzky-Golay filter window length (odd, >=3)")
    parser.add_argument("--savgol-poly", type=int, default=2,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    frame_tolerances = [int(t) for t in args.frame_tolerance.split(",")]

    # ── Load VLM results ──
    with open(args.vlm_results) as f:
        vlm_data = json.load(f)

    vlm_config = vlm_data.get("config", {})
    vlm_episodes = vlm_data.get("episodes", [])
    print(f"[load] {len(vlm_episodes)} episodes from {args.vlm_results}")
    print(f"[config] model={vlm_config.get('model')}, "
          f"config_name={vlm_config.get('config_name')}")
    print(f"[gaze] savgol_window={args.savgol_window}, "
          f"savgol_poly={args.savgol_poly}")

    # ── Phase 1: Precompute angular velocity ──
    print(f"\n[phase 1] Precomputing gaze angular velocity...")
    dataset_dir = args.dataset_dir
    ep_ang_vel: dict[str, np.ndarray] = {}
    ep_gt: dict[str, dict] = {}

    # Gather episode names from the results
    ep_names = [ep["episode"] for ep in vlm_episodes]

    for ep_name in ep_names:
        ep_dir = os.path.join(dataset_dir, ep_name)

        gt = load_ground_truth(ep_dir)
        if gt is None:
            continue
        ep_gt[ep_name] = gt

        gaze_data = load_gaze_pitch_yaw(ep_dir)
        if gaze_data is None:
            continue
        pitch, yaw = gaze_data

        ang_vel = compute_angular_velocity_deg(
            pitch, yaw, VIDEO_FPS,
            args.savgol_window, args.savgol_poly,
        )
        ep_ang_vel[ep_name] = ang_vel

    print(f"[phase 1] Done: {len(ep_ang_vel)} episodes with gaze velocity, "
          f"{len(ep_gt)} with GT labels")

    # ── Phase 2: Refine each episode ──
    print(f"\n[phase 2] Evaluating...\n")
    refined_episodes = []
    n_total = 0
    n_success = 0

    for ep_data in vlm_episodes:
        ep_name = ep_data["episode"]
        n_total += 1

        # Check prerequisites
        if ep_name not in ep_gt:
            print(f"  {ep_name}: SKIP (no GT)")
            continue
        if ep_name not in ep_ang_vel:
            print(f"  {ep_name}: SKIP (no gaze)")
            continue
        if "predicted_interval_frames" not in ep_data:
            print(f"  {ep_name}: SKIP (no VLM prediction)")
            continue

        gt = ep_gt[ep_name]
        ang_vel = ep_ang_vel[ep_name]
        T = gt["total_frames"]
        gt_kf_set = gt["gt_kf_set"]

        pred_start_f, pred_end_f = ep_data["predicted_interval_frames"]
        pred_start_f = max(0, min(pred_start_f, T - 1))
        pred_end_f = max(0, min(pred_end_f, T - 1))

        # ── Midpoint selection (baseline) ──
        mid_frame = (pred_start_f + pred_end_f) // 2
        mid_eval = check_frame_hit(mid_frame, gt_kf_set, T, frame_tolerances)

        # ── Gaze min-velocity selection ──
        gaze_frame, gaze_vel = select_min_velocity_frame(
            ang_vel, pred_start_f, pred_end_f, T,
        )
        gaze_eval = check_frame_hit(gaze_frame, gt_kf_set, T, frame_tolerances)

        n_success += 1

        # Build episode result
        ep_result = {
            "episode": ep_name,
            "vlm_interval_frames": [pred_start_f, pred_end_f],
            "gt_interval_frames": [gt["kf_start_frame"], gt["kf_end_frame"]],
            "iou": ep_data.get("iou"),
            "midpoint": {
                "selected_frame": mid_eval["frame"],
                "hit": mid_eval["exact_hit"],
                "nearest_gt_dist": mid_eval["nearest_gt_dist"],
                **{f"hit@{k}": mid_eval[f"hit@{k}"]
                   for k in frame_tolerances},
            },
            "gaze_velocity": {
                "selected_frame": gaze_eval["frame"],
                "velocity_deg_s": round(gaze_vel, 2),
                "hit": gaze_eval["exact_hit"],
                "nearest_gt_dist": gaze_eval["nearest_gt_dist"],
                **{f"hit@{k}": gaze_eval[f"hit@{k}"]
                   for k in frame_tolerances},
            },
        }
        refined_episodes.append(ep_result)

        # Log
        mid_mark = "HIT" if mid_eval["exact_hit"] else "MISS"
        gaze_mark = "HIT" if gaze_eval["exact_hit"] else "MISS"
        print(f"  {ep_name}:  "
              f"mid={mid_mark}(f{mid_eval['frame']})  "
              f"gaze={gaze_mark}(f{gaze_eval['frame']}, "
              f"{gaze_vel:.1f} deg/s)  "
              f"pred=[{pred_start_f}-{pred_end_f}]  "
              f"gt=[{gt['kf_start_frame']}-{gt['kf_end_frame']}]")

    # ── Aggregate ──
    print(f"\n{'='*72}")
    print(f"  Gaze-Velocity Refinement Summary")
    print(f"  Episodes: {n_total} total, {n_success} evaluated")
    print(f"  savgol_window={args.savgol_window}, "
          f"savgol_poly={args.savgol_poly}")
    print(f"{'='*72}")

    if n_success > 0:
        # Midpoint stats
        mid_hits = sum(1 for e in refined_episodes
                       if e["midpoint"]["hit"])
        gaze_hits = sum(1 for e in refined_episodes
                        if e["gaze_velocity"]["hit"])

        mid_dists = [e["midpoint"]["nearest_gt_dist"]
                     for e in refined_episodes]
        gaze_dists = [e["gaze_velocity"]["nearest_gt_dist"]
                      for e in refined_episodes]

        print(f"\n  {'Metric':<30s} {'Midpoint':>14s}  "
              f"{'Gaze (min vel)':>14s}  {'Delta':>8s}")
        print(f"  {'-'*30} {'-'*14}  {'-'*14}  {'-'*8}")

        m_rate = mid_hits / n_success * 100
        g_rate = gaze_hits / n_success * 100
        delta = g_rate - m_rate
        sign = "+" if delta >= 0 else ""
        print(f"  {'Exact match (+-0)':<30s} {m_rate:>13.1f}%  "
              f"{g_rate:>13.1f}%  {sign}{delta:>6.1f}%")

        for k in frame_tolerances:
            m_k = sum(1 for e in refined_episodes
                      if e["midpoint"][f"hit@{k}"])
            g_k = sum(1 for e in refined_episodes
                      if e["gaze_velocity"][f"hit@{k}"])
            m_r = m_k / n_success * 100
            g_r = g_k / n_success * 100
            d = g_r - m_r
            s = "+" if d >= 0 else ""
            label = f"frame_hit@+-{k} frames"
            print(f"  {label:<30s} {m_r:>13.1f}%  "
                  f"{g_r:>13.1f}%  {s}{d:>6.1f}%")

        m_mean = statistics.mean(mid_dists)
        g_mean = statistics.mean(gaze_dists)
        m_med = statistics.median(mid_dists)
        g_med = statistics.median(gaze_dists)
        print(f"\n  {'Nearest GT dist (mean)':<30s} "
              f"{m_mean:>13.1f}   {g_mean:>13.1f}")
        print(f"  {'Nearest GT dist (median)':<30s} "
              f"{m_med:>13.1f}   {g_med:>13.1f}")

        # Build aggregate dict
        agg = {
            "num_total": n_total,
            "num_evaluated": n_success,
            "midpoint": {
                "exact_accuracy": round(mid_hits / n_success, 4),
                "mean_nearest_gt_dist": round(m_mean, 2),
                "median_nearest_gt_dist": round(m_med, 2),
            },
            "gaze_velocity": {
                "exact_accuracy": round(gaze_hits / n_success, 4),
                "mean_nearest_gt_dist": round(g_mean, 2),
                "median_nearest_gt_dist": round(g_med, 2),
            },
        }
        for k in frame_tolerances:
            m_k = sum(1 for e in refined_episodes
                      if e["midpoint"][f"hit@{k}"])
            g_k = sum(1 for e in refined_episodes
                      if e["gaze_velocity"][f"hit@{k}"])
            agg["midpoint"][f"hit@{k}_rate"] = round(m_k / n_success, 4)
            agg["gaze_velocity"][f"hit@{k}_rate"] = round(g_k / n_success, 4)
    else:
        agg = {"num_total": n_total, "num_evaluated": 0}
        print("  No episodes to evaluate.")

    print(f"{'='*72}")

    # ── Save results ──
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
        "aggregate": agg,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
