#!/usr/bin/env python3
"""
Summarize multi-segment VLM result files and optional gaze-refinement outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _collect_result_paths(results_dir: str, contains: str | None) -> list[str]:
    names = sorted(
        name for name in os.listdir(results_dir)
        if name.startswith("result_multiseg_") and name.endswith(".json")
    )
    if contains:
        names = [name for name in names if contains in name]
    return [os.path.join(results_dir, name) for name in names]


def _build_rows(results_dir: str, contains: str | None) -> list[dict[str, str]]:
    rows = []
    for result_path in _collect_result_paths(results_dir, contains):
        result_data = _load_json(result_path)
        if result_data is None:
            continue

        config = result_data.get("config", {})
        agg = result_data.get("aggregate", {})
        refined_name = f"refined_{os.path.basename(result_path)}"
        refined_path = os.path.join(results_dir, refined_name)
        refined_data = _load_json(refined_path)

        row = {
            "file": os.path.basename(result_path),
            "fps": str(config.get("video_fps", "-")),
            "cap": "T" if config.get("caption") else "F",
            "gaze": "T" if config.get("gaze_annot") else "F",
            "seg_f1": _fmt_num(_get_nested(agg, "segment_f1", "mean")),
            "count_acc": _fmt_rate(_get_nested(agg, "exact_count_accuracy", "rate")),
            "pred_mid_hit": _fmt_rate(_get_nested(agg, "pred_midpoint_hit_rate", "mean")),
            "frame_iou": _fmt_num(_get_nested(agg, "frame_iou", "mean")),
            "mid_exact": "-",
            "gaze_exact": "-",
            "mid_hit@1": "-",
            "gaze_hit@1": "-",
            "mid_hit@3": "-",
            "gaze_hit@3": "-",
            "mid_any_ep": "-",
            "gaze_any_ep": "-",
            "mid_all_ep": "-",
            "gaze_all_ep": "-",
            "mid_ord_gt": "-",
            "gaze_ord_gt": "-",
            "mid_ord_ep": "-",
            "gaze_ord_ep": "-",
        }

        if refined_data is not None:
            refined_agg = refined_data.get("aggregate", {})
            row["mid_exact"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "exact_accuracy"))
            row["gaze_exact"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "exact_accuracy"))
            row["mid_hit@1"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "hit@1_rate"))
            row["gaze_hit@1"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "hit@1_rate"))
            row["mid_hit@3"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "hit@3_rate"))
            row["gaze_hit@3"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "hit@3_rate"))
            row["mid_any_ep"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "any_hit_episode_rate"))
            row["gaze_any_ep"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "any_hit_episode_rate"))
            row["mid_all_ep"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "all_interval_episode_rate"))
            row["gaze_all_ep"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "all_interval_episode_rate"))
            row["mid_ord_gt"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "ordered_gt_hit_recall"))
            row["gaze_ord_gt"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "ordered_gt_hit_recall"))
            row["mid_ord_ep"] = _fmt_rate(_get_nested(refined_agg, "midpoint", "ordered_all_hit_episode_rate"))
            row["gaze_ord_ep"] = _fmt_rate(_get_nested(refined_agg, "gaze_velocity", "ordered_all_hit_episode_rate"))

        rows.append(row)
    return rows


def _print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No matching result files found.")
        return

    columns = [
        "fps",
        "cap",
        "gaze",
        "seg_f1",
        "count_acc",
        "pred_mid_hit",
        "frame_iou",
        "mid_exact",
        "gaze_exact",
        "mid_hit@1",
        "gaze_hit@1",
        "mid_hit@3",
        "gaze_hit@3",
        "mid_any_ep",
        "gaze_any_ep",
        "mid_all_ep",
        "gaze_all_ep",
        "mid_ord_gt",
        "gaze_ord_gt",
        "mid_ord_ep",
        "gaze_ord_ep",
        "file",
    ]
    widths = {
        column: max(len(column), max(len(row[column]) for row in rows))
        for column in columns
    }

    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize result_multiseg and refined_result_multiseg files"
    )
    parser.add_argument("--results-dir", default="./results", help="Directory containing result_multiseg*.json")
    parser.add_argument(
        "--contains",
        default=None,
        help="Only include filenames containing this substring",
    )
    args = parser.parse_args()

    rows = _build_rows(args.results_dir, args.contains)
    _print_table(rows)


if __name__ == "__main__":
    main()
