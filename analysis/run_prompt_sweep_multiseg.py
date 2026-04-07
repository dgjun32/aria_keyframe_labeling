#!/usr/bin/env python3
"""
Run a prompt-variant sweep for eval_vlm_multisegment.py and rank the results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_VARIANTS = [
    "baseline",
    "count_then_localize",
    "audio_anchor",
    "phase_decompose",
    "tight_contact",
    "ordered_focus",
    "self_check",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-segment prompt sweep and rank prompt variants.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to run the eval scripts")
    parser.add_argument("--dataset-dir", default="./dataset", help="Dataset directory")
    parser.add_argument("--episode-prefix", default=None, help="Only include episodes whose names start with this prefix")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode names; overrides --episode-prefix")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model")
    parser.add_argument("--video-fps", type=int, default=4, help="FPS hint sent to Gemini")
    parser.add_argument("--max-intervals", type=int, default=2, help="Maximum number of intervals")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count")
    parser.add_argument("--match-frame-tolerance", type=int, default=3, help="Frame gap tolerance for segment matching")
    parser.add_argument("--caption", action="store_true", help="Enable caption overlay")
    parser.add_argument("--gaze-annot", action="store_true", help="Enable gaze overlay")
    parser.add_argument("--include-audio", action="store_true", help="Mux sidecar audio into input videos")
    parser.add_argument("--audio-dir", default="./preproc_files", help="Directory containing {episode}_audio.wav")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated prompt variants to run",
    )
    parser.add_argument("--resume", action="store_true", help="Resume existing eval JSONL files where possible")
    parser.add_argument("--exp-suffix", default="", help="Optional suffix appended to eval output filenames")
    parser.add_argument("--output-json", default="results/prompt_sweep_multiseg_summary.json", help="Summary JSON output")
    parser.add_argument("--output-csv", default="results/prompt_sweep_multiseg_summary.csv", help="Summary CSV output")
    return parser.parse_args()


def resolve_episodes(args: argparse.Namespace) -> str:
    if args.episodes:
        return ",".join(token.strip() for token in args.episodes.split(",") if token.strip())

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    names = sorted(
        path.name for path in dataset_dir.iterdir()
        if path.is_dir() and (args.episode_prefix is None or path.name.startswith(args.episode_prefix))
    )
    return ",".join(names)


def build_result_path(args: argparse.Namespace, variant: str) -> Path:
    name = (
        "results/"
        f"result_multiseg_{args.model}"
        f"_audio_{'true' if args.include_audio else 'false'}"
        f"_gaze_{'true' if args.gaze_annot else 'false'}"
        f"_cap_{'true' if args.caption else 'false'}"
        f"_fps{args.video_fps}_max{args.max_intervals}_prompt_{variant}.json"
    )
    if args.exp_suffix:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_{args.exp_suffix}{ext}"
    return PROJECT_ROOT / name


def build_refined_path(args: argparse.Namespace, variant: str) -> Path:
    name = (
        "results/"
        f"refined_result_multiseg_{args.model}"
        f"_audio_{'true' if args.include_audio else 'false'}"
        f"_gaze_{'true' if args.gaze_annot else 'false'}"
        f"_cap_{'true' if args.caption else 'false'}"
        f"_fps{args.video_fps}_max{args.max_intervals}_prompt_{variant}.json"
    )
    if args.exp_suffix:
        stem, ext = os.path.splitext(name)
        name = f"{stem}_{args.exp_suffix}{ext}"
    return PROJECT_ROOT / name


def run_command(command: list[str]) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def extract_row(args: argparse.Namespace, variant: str) -> dict:
    raw = load_json(build_result_path(args, variant))
    refined = load_json(build_refined_path(args, variant))

    raw_agg = raw["aggregate"]
    refined_mid = refined["aggregate"]["midpoint"]
    refined_gaze = refined["aggregate"]["gaze_velocity"]

    return {
        "prompt_variant": variant,
        "num_success": raw_agg["num_success"],
        "num_failures": raw_agg["num_failures"],
        "count_acc": raw_agg["exact_count_accuracy"]["rate"],
        "seg_f1": raw_agg["segment_f1"]["mean"],
        "frame_iou": raw_agg["frame_iou"]["mean"],
        "pred_mid_hit": raw_agg["pred_midpoint_hit_rate"]["mean"],
        "mid_exact": refined_mid["exact_accuracy"],
        "mid_any_ep": refined_mid["any_hit_episode_rate"],
        "mid_all_ep": refined_mid["all_interval_episode_rate"],
        "mid_ord_gt": refined_mid["ordered_gt_hit_recall"],
        "mid_ord_ep": refined_mid["ordered_all_hit_episode_rate"],
        "gaze_exact": refined_gaze["exact_accuracy"],
        "total_cost_usd": raw_agg["total_cost_usd"],
    }


def write_outputs(rows: list[dict], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(rows, f, indent=2)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row["mid_ord_ep"],
            row["mid_ord_gt"],
            row["seg_f1"],
            row["mid_exact"],
            row["count_acc"],
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    episodes = resolve_episodes(args)
    variants = [token.strip() for token in args.variants.split(",") if token.strip()]
    if not episodes:
        raise ValueError("No episodes resolved for the prompt sweep.")

    rows = []
    for variant in variants:
        eval_cmd = [
            args.python_bin,
            "eval_vlm_multisegment.py",
            "--dataset-dir", args.dataset_dir,
            "--episodes", episodes,
            "--model", args.model,
            "--video-fps", str(args.video_fps),
            "--max-intervals", str(args.max_intervals),
            "--workers", str(args.workers),
            "--match-frame-tolerance", str(args.match_frame_tolerance),
            "--prompt-variant", variant,
        ]
        if args.caption:
            eval_cmd.append("--caption")
        if args.gaze_annot:
            eval_cmd.append("--gaze-annot")
        if args.include_audio:
            eval_cmd += ["--include-audio", "--audio-dir", args.audio_dir]
        if args.exp_suffix:
            eval_cmd += ["--exp-suffix", args.exp_suffix]
        if args.resume:
            eval_cmd.append("--resume")

        run_command(eval_cmd)

        refined_cmd = [
            args.python_bin,
            "eval_gaze_refinement_multisegment.py",
            "--vlm-results",
            str(build_result_path(args, variant)),
            "--dataset-dir",
            args.dataset_dir,
        ]
        run_command(refined_cmd)
        rows.append(extract_row(args, variant))

    ranked_rows = sort_rows(rows)
    write_outputs(ranked_rows, PROJECT_ROOT / args.output_json, PROJECT_ROOT / args.output_csv)

    print("\nRanked prompt variants:")
    for idx, row in enumerate(ranked_rows, start=1):
        print(
            f"{idx}. {row['prompt_variant']}: "
            f"mid_ord_ep={row['mid_ord_ep']:.4f}, "
            f"mid_ord_gt={row['mid_ord_gt']:.4f}, "
            f"seg_f1={row['seg_f1']:.4f}, "
            f"mid_exact={row['mid_exact']:.4f}, "
            f"count_acc={row['count_acc']:.4f}"
        )

    print(f"\n[saved] {PROJECT_ROOT / args.output_json}")
    print(f"[saved] {PROJECT_ROOT / args.output_csv}")


if __name__ == "__main__":
    main()
