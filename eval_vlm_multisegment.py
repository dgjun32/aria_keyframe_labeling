#!/usr/bin/env python3
"""
Gemini VLM Multi-Segment Evaluation for Keyframe Detection
==========================================================

This script keeps the existing single-interval baseline intact and adds a
separate evaluator that asks Gemini to return 1..N keyframe intervals.
It is intended for datasets whose ground truth contains multiple labeled
segments per episode, such as pick-and-place tasks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import statistics
import time
from functools import lru_cache

import cv2
import numpy as np
from google import genai

from eval_vlm_baseline import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_VIDEO_FPS_HINT,
    DEFAULT_THINKING_BUDGET,
    PROMPT_CONFIG_NOTES,
    VIDEO_FPS,
    _save_as_h264,
    append_jsonl,
    call_gemini,
    extract_cost,
    get_config_name,
    load_existing_results,
    prepare_video,
    resolve_gemini_api_key,
)


MODEL_CHOICES = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]


PROMPT_TEMPLATE_MULTI = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and interacts with a specific object.

A spoken instruction was given:
  "{transcript}"

{config_note}

{variant_note}

Your task: identify all distinct **keyframe time ranges** that capture the
moments of interaction with the target object mentioned in the instruction.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture: when the hand reaches toward, touches, grasps, places, or
    releases the target object.
  - Gaze fixation: when the wearer's gaze stabilises on the target object
    (if gaze information is available).

Some videos contain only one interaction moment. Others contain multiple
separate moments, such as a pick phase and a later place phase.
Return between 1 and {max_intervals} intervals. If there are multiple clearly
separated interaction phases, return each phase as its own interval.
If there is only one interaction phase, return exactly one interval.

Each interval should be short and precise, typically 0.2-2.0 seconds.
Do not merge two clearly separate events into a single long interval.
If unsure, prefer fewer precise intervals over many noisy intervals.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of which cues you used>",
  "keyframe_intervals": [
    {{"start": <float, seconds>, "end": <float, seconds>}}
  ]
}}

Important:
  - Return between 1 and {max_intervals} intervals.
  - Intervals must be sorted by time.
  - Intervals should be non-overlapping.
  - Always return positive start/end values.
  - Never return -1.
"""


PROMPT_TEMPLATE_MULTI_SUBGOAL = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and interacts with specific objects.

A spoken instruction was given:
  "{transcript}"

{config_note}

Your task has two steps:

Step 1: Determine the number of keyframes.
Count the number of distinct target-object interaction events that are required
to fulfill the instruction. Count interaction events, not just noun mentions.
For pick-and-place tasks, this is often a pick event followed by a later place
event, but only output the events that are clearly supported by the video.

Step 2: Identify each keyframe time range.
For each keyframe in chronological order, identify a short interval whose
midpoint lands inside the decisive interaction moment for that subgoal.

Use these rules for pick-and-place actions:
  - Pick event: center the interval on the moment when the object is securely
    grasped and begins to leave its original support, not on the initial reach.
  - Place event: center the interval on the moment when the object makes final
    contact at the destination and the hand is releasing it, not on the
    approach or carry motion.
  - Ignore long transport motion between pick and place.

The keyframe intervals are determined by the convergence of multiple cues:
  - Utterance timing: when the spoken instruction refers to the target object
    or destination.
  - Hand-object interaction: grasp, lift-off, placement, release, or stable
    contact at the destination.
  - Gaze fixation: when the wearer's gaze stabilises on the target object
    or destination (if gaze information is available).

Each interval should be short and precise, typically 0.2-2.0 seconds.
Do not merge two clearly separate events into a single long interval.
If unsure, prefer fewer precise intervals over extra speculative ones.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of which cues you used>",
  "num_keyframes": <int>,
  "object_references": ["<object or destination 1>", "<object or destination 2>"],
  "subgoal_instructions": ["pick up <object>", "place it <destination>"],
  "keyframe_intervals": [
    {{"start": <float, seconds>, "end": <float, seconds>}}
  ]
}}

Important:
  - All lists must have the same length, equal to num_keyframes.
  - Return between 1 and {max_intervals} intervals.
  - Intervals must be sorted chronologically.
  - Intervals should be non-overlapping.
  - Always return positive start/end values.
  - Never return -1.
"""


PROMPT_TEMPLATE_MULTI_SUBGOAL_LITERAL = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and interacts with specific objects.

A spoken instruction was given:
  "{transcript}"

{config_note}

Your task has two steps:

**Step 1: Determine the number of keyframes.**
Analyze the spoken instruction and count the number of distinct object
references or interaction moments. Each time the instruction refers to a
target object (by pointing words like "this", "that", or by naming it),
it corresponds to one keyframe - the moment the wearer interacts with
that object.

Examples:
  - "I want this cupcake" -> 1 keyframe
    subgoal_instructions: ["pick up this cupcake"]
  - "give me this cupcake and this beverage" -> 2 keyframes
    subgoal_instructions: ["pick up this cupcake", "pick up this beverage"]
  - "put that one in this basket" -> 2 keyframes
    subgoal_instructions: ["pick up that one", "place it in this basket"]
  - "put that one in this basket and the coca cola in this basket"
    -> 4 keyframes
    subgoal_instructions: ["pick up that one", "place it in this basket",
      "pick up the coca cola", "place it in this basket"]
  - "point to this soda" -> 1 keyframe
    subgoal_instructions: ["pick up this soda"]

IMPORTANT: Each subgoal_instruction MUST use one of exactly two verb forms:
  1. "pick up <object>" - when the wearer selects / grabs / wants an object.
  2. "place it <destination>" - when the wearer puts / places the held
     object at a destination (e.g., "place it in this basket",
     "place it here", "place it on the shelf").
  Do NOT use any other verbs (e.g., "point to", "grab", "put", "get",
  "move", "take"). Always rephrase to "pick up" or "place it".

**Step 2: Identify each keyframe's time range.**
For each keyframe (in chronological order), identify the precise time interval
that captures the moment of interaction with the corresponding object.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture (pointing): when the hand/finger clearly points to a
    specific object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target object
    (if gaze information is available).

Prioritise intervals where the object indicated by hand pointing and the gaze
point are spatially aligned (same object/region). Use utterance timing to
confirm that this aligned object is the instructed target.

Each interval should be short and precise, typically 0.2-2.0 seconds.
Do not merge two clearly separate events into a single long interval.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of which cues you used>",
  "num_keyframes": <int, number of distinct interaction moments>,
  "object_references": ["<object 1>", "<object 2>", ...],
  "subgoal_instructions": ["pick up <object>", "place it <destination>", ...],
  "keyframe_intervals": [
    {{"start": <float, seconds>, "end": <float, seconds>}}
  ]
}}

Important:
  - All lists (object_references, subgoal_instructions, keyframe_intervals)
    must have the same length, equal to num_keyframes.
  - Maximum {max_intervals} intervals allowed.
  - Intervals must be sorted chronologically.
  - Intervals should be non-overlapping.
  - Always return positive start/end values.
  - Never return -1.
"""


PROMPT_TEMPLATE_MULTI_POINTING_LITERAL = """\
You are analyzing a first-person (egocentric) video recorded in a store.
The wearer is looking at items on shelves and visually indicates target objects
or destinations with the hand or finger.

A spoken instruction was given:
  "{transcript}"

{config_note}

Your task has two steps:

**Step 1: Determine the number of keyframes.**
Analyze the spoken instruction and count the number of distinct deictic
reference moments that should appear in the video. A deictic reference moment
is when the wearer clearly indicates a target object or destination region with
the hand or finger, typically for words such as "this", "that", "here", or a
named target/destination.

Examples:
  - "I want this cupcake" -> 1 keyframe
    subgoal_instructions: ["point to this cupcake"]
  - "give me this cupcake and this beverage" -> 2 keyframes
    subgoal_instructions: ["point to this cupcake", "point to this beverage"]
  - "put that one in this basket" -> 2 keyframes
    subgoal_instructions: ["point to that one", "point to this basket"]
  - "put that one in this basket and the coca cola in this basket"
    -> 4 keyframes
    subgoal_instructions: ["point to that one", "point to this basket",
      "point to the coca cola", "point to this basket"]
  - "point to this soda" -> 1 keyframe
    subgoal_instructions: ["point to this soda"]

IMPORTANT: Each subgoal_instruction MUST use exactly one verb form:
  - "point to <object or destination>"
  Do NOT rewrite the action into other manipulation verbs.

**Step 2: Identify each keyframe's time range.**
For each keyframe (in chronological order), identify the precise short interval
that captures the clearest pointing/indicating moment for the corresponding
object or destination.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object or
    destination.
  - Hand/finger pointing: when the hand direction clearly indicates a specific
    object or region.
  - Gaze fixation: when the wearer's gaze stabilises on that same object or
    region (if gaze information is available).

Prioritise intervals where the hand/finger indication is most explicit and
spatially aligned with the intended object or destination. Center the interval
on the strongest pointing frame itself, not on earlier arm motion or later
aftermath. Only use visually observed pointing/reference moments and do not
invent extra events that are not clearly shown.

Each interval should be short and precise, typically 0.2-2.0 seconds.
Do not merge two clearly separate pointing events into a single long interval.

The video is {duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of which cues you used>",
  "num_keyframes": <int, number of distinct reference moments>,
  "object_references": ["<object or destination 1>", "<object or destination 2>", ...],
  "subgoal_instructions": ["point to <object or destination>", ...],
  "keyframe_intervals": [
    {{"start": <float, seconds>, "end": <float, seconds>}}
  ]
}}

Important:
  - All lists (object_references, subgoal_instructions, keyframe_intervals)
    must have the same length, equal to num_keyframes.
  - Maximum {max_intervals} intervals allowed.
  - Intervals must be sorted chronologically.
  - Intervals should be non-overlapping.
  - Always return positive start/end values.
  - Never return -1.
"""


PROMPT_VARIANT_NOTES = {
    "baseline": (
        "Use the visible evidence in the video to localize the interaction phases."
    ),
    "event_auto_count": (
        "Determine the number of distinct target-object interaction events from the evidence in the video.\n"
        "Do not assume there are always two events.\n"
        "Return exactly as many intervals as there are clearly separated relevant interaction events, up to the allowed maximum.\n"
        "Ignore unrelated hand motion, context, and long carry or dwell periods between events."
    ),
    "event_auto_count_centered": (
        "Determine the number of distinct target-object interaction events from the evidence in the video.\n"
        "Do not assume there are always two events.\n"
        "Return exactly as many intervals as there are clearly separated relevant interaction events, up to the allowed maximum.\n"
        "For each event, center the interval on the decisive hand-object interaction frame so the midpoint stays inside the keyframe region.\n"
        "Do not pad an interval with long approach or aftermath frames."
    ),
    "pick_place_transition": (
        "For pick-and-place actions, prefer state-transition moments over approach motion.\n"
        "The first interval should center on the pick moment when the object is securely grasped and begins to leave its original support, not on the early reach.\n"
        "The second interval should center on the place moment when the object first settles at the destination and the hand is releasing it, not on the approach path.\n"
        "Ignore long carrying motion between pick and place."
    ),
    "pick_place_release_focus": (
        "For pick-and-place actions, use two short intervals around the decisive state changes.\n"
        "Pick interval: center the midpoint near lift-off, after grasp is established.\n"
        "Place interval: center the midpoint later, near final contact and release, after the object reaches the destination.\n"
        "Do not anchor the place interval to the beginning of the approach."
    ),
    "pick_place_state_change_strict": (
        "Detect each distinct target-object state change.\n"
        "For the pick event, the midpoint should land when the object is already being lifted from its original position.\n"
        "For the place event, the midpoint should land when the object is in contact with the destination and the hand is releasing or has just released it.\n"
        "If an interval would place its midpoint on reach, transport, or hover frames, shift it later and tighten it."
    ),
    "pick_place_subgoal_extract": (
        "First infer the subgoal sequence and the number of keyframes, then localize each interval.\n"
        "For pick-and-place actions, use `pick up <object>` for the pick event and `place it <destination>` for the place event.\n"
        "Each interval midpoint should land on the decisive state change, not on approach or carry frames."
    ),
    "pick_place_subgoal_literal": (
        "Use the literal two-step keyframe extraction format with num_keyframes, object_references, subgoal_instructions, and keyframe_intervals."
    ),
    "pointing_subgoal_literal": (
        "Use the literal two-step keyframe extraction format for pointing-only reference moments with num_keyframes, object_references, subgoal_instructions, and keyframe_intervals."
    ),
    "count_then_localize": (
        "First decide how many distinct interaction phases are present: usually 1 or 2.\n"
        "Then localize each phase separately.\n"
        "Do not output two intervals unless the video clearly shows two separate interaction moments."
    ),
    "audio_anchor": (
        "Use the audio track and the burned-in captions together.\n"
        "The spoken timing helps determine when the wearer commits to the requested action.\n"
        "Prefer intervals that begin near the relevant spoken phrase and tighten around the actual hand-object event."
    ),
    "phase_decompose": (
        "Decompose the task into chronological phases.\n"
        "For example, if the video shows an earlier first interaction and a later second interaction,\n"
        "return them as interval 1 and interval 2 in time order.\n"
        "Avoid combining the two phases into one broad interval."
    ),
    "phase_decompose_strict": (
        "Decompose the task into chronological phases and preserve their order.\n"
        "If there are two distinct interaction phases, interval 1 must describe the earlier phase and interval 2 the later phase.\n"
        "Before answering, internally check that the midpoint of each interval lands on the decisive hand-object interaction,\n"
        "not on approach frames or aftermath frames."
    ),
    "phase_decompose_tight": (
        "Decompose the task into chronological phases.\n"
        "For each phase, choose the shortest interval that still captures the decisive contact, grasp, pickup, placement, or release.\n"
        "Keep each interval center close to the target interaction itself, and avoid wide intervals that include too much lead-in or follow-through."
    ),
    "phase_balanced": (
        "Decompose the task into chronological phases.\n"
        "Use speech and captions to identify the correct object, but set the final interval boundaries from the visible hand-object interaction.\n"
        "For each phase, keep the interval wide enough that its midpoint stays inside the main interaction, even when the visible manipulation lasts longer than the spoken phrase.\n"
        "Avoid sliding an interval too late just because the later frames look stronger."
    ),
    "phase_centered": (
        "Decompose the task into chronological phases.\n"
        "For each phase, first identify the single most decisive frame where the hand is firmly engaging the correct object.\n"
        "Then expand a short interval around that frame so the midpoint remains on the interaction itself.\n"
        "Use the spoken instruction only as support, not as the sole anchor for timing."
    ),
    "phase_early_anchor": (
        "Decompose the task into chronological phases and preserve their order.\n"
        "Interval 1 must stay anchored to the first sustained interaction with the first requested object, not drift forward toward the later motion.\n"
        "If the first interaction spans many frames, include enough of it so the interval midpoint still lands inside that first phase.\n"
        "Apply the same rule to interval 2 for the later phase."
    ),
    "phase_balanced_two_bias": (
        "These videos often contain two distinct target-object interaction phases.\n"
        "Prefer two time-ordered intervals when the wearer clearly touches or grasps two requested objects.\n"
        "Still set each interval from the visible hand-object interaction rather than only the spoken phrase,\n"
        "and widen an interval slightly if needed so its midpoint remains inside the phase."
    ),
    "double_pickup_bias": (
        "These videos often contain two distinct target-object interaction phases.\n"
        "Prefer returning two time-ordered intervals unless the video clearly contains only one relevant interaction phase.\n"
        "The first interval should stay anchored to the first target interaction even if the later interaction is visually stronger."
    ),
    "tight_contact": (
        "Choose the shortest interval that still captures the decisive interaction.\n"
        "The center of the interval should fall close to the moment of contact, grasp, placement, or release,\n"
        "not on the approach before it or the aftermath after it."
    ),
    "ordered_focus": (
        "Order matters.\n"
        "Interval 1 should correspond to the earliest relevant interaction phase, interval 2 to the next one.\n"
        "If there are two distinct phases, make sure the first interval is anchored to the first phase rather than the stronger later event."
    ),
    "self_check": (
        "Before answering, internally verify four things:\n"
        "1. The interval count matches the number of distinct interaction phases.\n"
        "2. Each interval is temporally tight.\n"
        "3. Intervals are sorted and non-overlapping.\n"
        "4. The midpoint of each interval lands on the target interaction rather than context frames."
    ),
}


def get_prompt_variant_note(prompt_variant: str) -> str:
    if prompt_variant not in PROMPT_VARIANT_NOTES:
        raise KeyError(f"Unknown prompt variant: {prompt_variant}")
    return PROMPT_VARIANT_NOTES[prompt_variant]


def build_multi_prompt(
    *,
    prompt_variant: str,
    transcript_text: str,
    duration_sec: float,
    fps: float,
    config_note: str,
    max_intervals: int,
) -> str:
    if prompt_variant == "pick_place_subgoal_extract":
        return PROMPT_TEMPLATE_MULTI_SUBGOAL.format(
            transcript=transcript_text,
            duration=duration_sec,
            fps=fps,
            config_note=config_note,
            max_intervals=max_intervals,
        )
    if prompt_variant == "pick_place_subgoal_literal":
        return PROMPT_TEMPLATE_MULTI_SUBGOAL_LITERAL.format(
            transcript=transcript_text,
            duration=duration_sec,
            fps=fps,
            config_note=config_note,
            max_intervals=max_intervals,
        )
    if prompt_variant == "pointing_subgoal_literal":
        return PROMPT_TEMPLATE_MULTI_POINTING_LITERAL.format(
            transcript=transcript_text,
            duration=duration_sec,
            fps=fps,
            config_note=config_note,
            max_intervals=max_intervals,
        )

    return PROMPT_TEMPLATE_MULTI.format(
        transcript=transcript_text,
        duration=duration_sec,
        fps=fps,
        config_note=config_note,
        variant_note=get_prompt_variant_note(prompt_variant),
        max_intervals=max_intervals,
    )


def _get_video_meta(video_path: str) -> tuple[int, float]:
    if not os.path.exists(video_path):
        return 0, float(VIDEO_FPS)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        fps = float(VIDEO_FPS)
    return total_frames, float(fps)


def _coerce_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_segments_from_labels(labels: np.ndarray) -> list[dict]:
    labels = labels.astype(np.int8)
    if labels.size == 0:
        return []

    diffs = np.diff(labels)
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0] + 1)
    if labels[0] == 1:
        starts = [0] + starts
    if labels[-1] == 1:
        ends = ends + [len(labels)]

    segments = []
    for start, end_exclusive in zip(starts, ends):
        end = int(end_exclusive - 1)
        if end >= start:
            segments.append(
                {
                    "start_frame": int(start),
                    "end_frame": end,
                }
            )
    return segments


def _load_segments_from_annotations(episode_dir: str) -> tuple[list[dict], float | None]:
    ann_path = os.path.join(episode_dir, "annotations.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            ann_data = json.load(f)
        fps = ann_data.get("fps")
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = None

        annotations = ann_data.get("annotations", [])
        if isinstance(annotations, list):
            segments = []
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                kind = ann.get("kind")
                if kind not in (None, "label"):
                    continue
                start = _coerce_int(ann.get("start_frame", ann.get("start")))
                end = _coerce_int(ann.get("end_frame", ann.get("end")))
                if start is None or end is None:
                    continue
                if end < start:
                    start, end = end, start
                if start < 0:
                    continue
                segments.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                    }
                )
            if segments:
                return segments, fps

    seg_path = os.path.join(episode_dir, "segments.json")
    if os.path.exists(seg_path):
        with open(seg_path) as f:
            seg_data = json.load(f)
        raw_segments = seg_data.get("segments", [])
        if isinstance(raw_segments, list):
            segments = []
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                start = _coerce_int(seg.get("start_frame", seg.get("start")))
                end = _coerce_int(seg.get("end_frame", seg.get("end")))
                if start is None or end is None:
                    continue
                if end < start:
                    start, end = end, start
                if start < 0:
                    continue
                segments.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                    }
                )
            if segments:
                return segments, None

    return [], None


def load_ground_truth_multi(episode_dir: str) -> dict | None:
    labels_path = os.path.join(episode_dir, "labels.npy")
    video_path = os.path.join(episode_dir, "video.mp4")

    total_frames, video_fps = _get_video_meta(video_path)
    labels: np.ndarray | None = None
    if os.path.exists(labels_path):
        labels = np.load(labels_path).astype(np.int8)
        total_frames = len(labels)

    segments, ann_fps = _load_segments_from_annotations(episode_dir)
    fps = ann_fps or video_fps or float(VIDEO_FPS)

    if segments:
        if labels is None:
            if total_frames <= 0:
                total_frames = max(seg["end_frame"] for seg in segments) + 1
            labels = np.zeros(total_frames, dtype=np.int8)
            for seg in segments:
                start = max(0, min(total_frames - 1, seg["start_frame"]))
                end = max(0, min(total_frames - 1, seg["end_frame"]))
                if end >= start:
                    labels[start:end + 1] = 1
    elif labels is not None:
        segments = _extract_segments_from_labels(labels)

    if labels is None or not segments:
        return None

    gt_segments = []
    for idx, seg in enumerate(sorted(segments, key=lambda s: (s["start_frame"], s["end_frame"]))):
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        mid_frame = int(round((start + end) / 2))
        gt_segments.append(
            {
                "segment_idx": idx,
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start / fps, 4),
                "end_sec": round(end / fps, 4),
                "mid_frame": mid_frame,
                "mid_sec": round(mid_frame / fps, 4),
                "duration_frames": end - start + 1,
                "duration_sec": round((end - start + 1) / fps, 4),
            }
        )

    gt_kf_set = set(np.where(labels == 1)[0].tolist())
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if fps > 0 else 0.0,
        "segments": gt_segments,
        "labels": labels,
        "gt_kf_set": gt_kf_set,
    }


def _parse_interval_item(item) -> tuple[float, float] | None:
    if isinstance(item, dict):
        candidate_keys = [
            ("start", "end"),
            ("keyframe_start", "keyframe_end"),
            ("start_sec", "end_sec"),
        ]
        for start_key, end_key in candidate_keys:
            if start_key in item and end_key in item:
                try:
                    return float(item[start_key]), float(item[end_key])
                except (TypeError, ValueError):
                    return None
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            return float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
    return None


def parse_multisegment_response(response, max_intervals: int) -> dict | None:
    try:
        payload = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        return None

    reasoning = ""
    raw_intervals = None
    num_keyframes = None
    object_references = []
    subgoal_instructions = []
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning", "")
        if isinstance(payload.get("num_keyframes"), int):
            num_keyframes = int(payload["num_keyframes"])
        if isinstance(payload.get("object_references"), list):
            object_references = [str(item) for item in payload["object_references"]]
        if isinstance(payload.get("subgoal_instructions"), list):
            subgoal_instructions = [str(item) for item in payload["subgoal_instructions"]]
        for key in ("keyframe_intervals", "intervals", "segments", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_intervals = value
                break
        if raw_intervals is None and {
            "keyframe_start",
            "keyframe_end",
        }.issubset(payload.keys()):
            raw_intervals = [payload]
    elif isinstance(payload, list):
        raw_intervals = payload

    if not isinstance(raw_intervals, list):
        return None

    intervals = []
    for item in raw_intervals:
        parsed = _parse_interval_item(item)
        if parsed is None:
            continue
        start_sec, end_sec = parsed
        if not (math.isfinite(start_sec) and math.isfinite(end_sec)):
            continue
        if start_sec > end_sec:
            start_sec, end_sec = end_sec, start_sec
        if start_sec < 0 or end_sec < 0:
            continue
        intervals.append(
            {
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )

    intervals.sort(key=lambda seg: (seg["start_sec"], seg["end_sec"]))
    raw_count = len(intervals)
    if max_intervals > 0:
        intervals = intervals[:max_intervals]

    if not intervals:
        return None

    return {
        "reasoning": reasoning,
        "num_keyframes": num_keyframes,
        "object_references": object_references,
        "subgoal_instructions": subgoal_instructions,
        "intervals": intervals,
        "raw_interval_count": raw_count,
        "truncated_interval_count": max(0, raw_count - len(intervals)),
    }


def normalize_prediction(pred: dict, gt: dict) -> dict:
    fps = gt["fps"]
    duration_sec = gt["duration_sec"]
    total_frames = gt["total_frames"]

    normalized = []
    for idx, interval in enumerate(pred["intervals"]):
        start_sec = max(0.0, min(float(interval["start_sec"]), duration_sec))
        end_sec = max(0.0, min(float(interval["end_sec"]), duration_sec))
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

    normalized.sort(key=lambda seg: (seg["start_frame"], seg["end_frame"], seg["pred_idx"]))
    return {
        **pred,
        "intervals": normalized,
    }


def _frame_set(intervals: list[dict]) -> set[int]:
    frames: set[int] = set()
    for interval in intervals:
        frames.update(range(interval["start_frame"], interval["end_frame"] + 1))
    return frames


def _pair_metrics(pred_interval: dict, gt_interval: dict, fps: float) -> dict:
    ps, pe = pred_interval["start_frame"], pred_interval["end_frame"]
    gs, ge = gt_interval["start_frame"], gt_interval["end_frame"]

    intersection = max(0, min(pe, ge) - max(ps, gs) + 1)
    pred_len = pe - ps + 1
    gt_len = ge - gs + 1
    union = pred_len + gt_len - intersection
    iou = intersection / union if union > 0 else 0.0

    if intersection > 0:
        gap_frames = 0
    elif pe < gs:
        gap_frames = gs - pe
    else:
        gap_frames = ps - ge

    midpoint_error_frames = abs(pred_interval["mid_frame"] - gt_interval["mid_frame"])
    return {
        "iou": iou,
        "intersection_frames": intersection,
        "gap_frames": gap_frames,
        "midpoint_error_frames": midpoint_error_frames,
        "midpoint_error_sec": midpoint_error_frames / fps if fps > 0 else 0.0,
    }


def _match_intervals(
    pred_intervals: list[dict],
    gt_segments: list[dict],
    fps: float,
    match_frame_tolerance: int,
) -> tuple[list[tuple[int, int]], list[list[dict]]]:
    n_pred = len(pred_intervals)
    n_gt = len(gt_segments)
    pair_table = [
        [_pair_metrics(pred, gt, fps) for gt in gt_segments]
        for pred in pred_intervals
    ]

    @lru_cache(maxsize=None)
    def solve(pred_idx: int, used_mask: int):
        if pred_idx >= n_pred:
            return (0, 0.0, 0.0, 0.0, ())

        best = solve(pred_idx + 1, used_mask)
        best_score = best[:4]

        for gt_idx in range(n_gt):
            if used_mask & (1 << gt_idx):
                continue
            pair = pair_table[pred_idx][gt_idx]
            if pair["gap_frames"] > match_frame_tolerance:
                continue

            rest = solve(pred_idx + 1, used_mask | (1 << gt_idx))
            candidate = (
                rest[0] + 1,
                rest[1] + pair["iou"],
                rest[2] - pair["midpoint_error_frames"],
                rest[3] - pair["gap_frames"],
                ((pred_idx, gt_idx),) + rest[4],
            )
            if candidate[:4] > best_score:
                best = candidate
                best_score = candidate[:4]

        return best

    matched_pairs = sorted(solve(0, 0)[4])
    return matched_pairs, pair_table


def evaluate_episode_multi(
    pred: dict,
    gt: dict,
    match_frame_tolerance: int,
) -> tuple[dict, list[dict]]:
    pred_intervals = pred["intervals"]
    gt_segments = gt["segments"]
    fps = gt["fps"]

    pred_count = len(pred_intervals)
    gt_count = len(gt_segments)

    pred_frames = _frame_set(pred_intervals)
    gt_frames = gt["gt_kf_set"]
    inter_frames = len(pred_frames & gt_frames)
    union_frames = len(pred_frames | gt_frames)

    frame_iou = inter_frames / union_frames if union_frames else 0.0
    gt_frame_coverage = inter_frames / len(gt_frames) if gt_frames else 0.0
    pred_frame_precision = inter_frames / len(pred_frames) if pred_frames else 0.0

    midpoint_hits = sum(1 for interval in pred_intervals if interval["mid_frame"] in gt_frames)
    midpoint_nearest = []
    for interval in pred_intervals:
        if gt_frames:
            midpoint_nearest.append(
                min(abs(interval["mid_frame"] - gt_frame) for gt_frame in gt_frames)
            )
        else:
            midpoint_nearest.append(gt["total_frames"])

    matched_pairs, pair_table = _match_intervals(
        pred_intervals,
        gt_segments,
        fps,
        match_frame_tolerance,
    )
    matched_details = []
    matched_ious = []
    matched_mid_errors = []
    for pred_idx, gt_idx in matched_pairs:
        pair = pair_table[pred_idx][gt_idx]
        pred_interval = pred_intervals[pred_idx]
        gt_interval = gt_segments[gt_idx]
        matched_ious.append(pair["iou"])
        matched_mid_errors.append(pair["midpoint_error_sec"])
        matched_details.append(
            {
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "pred_interval_frames": [
                    pred_interval["start_frame"],
                    pred_interval["end_frame"],
                ],
                "gt_interval_frames": [
                    gt_interval["start_frame"],
                    gt_interval["end_frame"],
                ],
                "iou": round(pair["iou"], 4),
                "gap_frames": int(pair["gap_frames"]),
                "midpoint_error_sec": round(pair["midpoint_error_sec"], 4),
            }
        )

    best_pred_ious = []
    for pred_idx in range(pred_count):
        if gt_count:
            best_pred_ious.append(max(pair_table[pred_idx][gt_idx]["iou"] for gt_idx in range(gt_count)))
        else:
            best_pred_ious.append(0.0)

    best_gt_ious = []
    for gt_idx in range(gt_count):
        if pred_count:
            best_gt_ious.append(max(pair_table[pred_idx][gt_idx]["iou"] for pred_idx in range(pred_count)))
        else:
            best_gt_ious.append(0.0)

    matched_count = len(matched_pairs)
    segment_precision = matched_count / pred_count if pred_count else 0.0
    segment_recall = matched_count / gt_count if gt_count else 0.0
    if segment_precision + segment_recall > 0:
        segment_f1 = 2 * segment_precision * segment_recall / (segment_precision + segment_recall)
    else:
        segment_f1 = 0.0

    metrics = {
        "pred_interval_count": pred_count,
        "gt_interval_count": gt_count,
        "count_error": pred_count - gt_count,
        "abs_count_error": abs(pred_count - gt_count),
        "matched_count": matched_count,
        "segment_precision": round(segment_precision, 4),
        "segment_recall": round(segment_recall, 4),
        "segment_f1": round(segment_f1, 4),
        "exact_count_match": pred_count == gt_count,
        "all_gt_matched": matched_count == gt_count,
        "all_pred_matched": matched_count == pred_count,
        "frame_iou": round(frame_iou, 4),
        "gt_frame_coverage": round(gt_frame_coverage, 4),
        "pred_frame_precision": round(pred_frame_precision, 4),
        "pred_midpoint_hit_count": midpoint_hits,
        "pred_midpoint_hit_rate": round(midpoint_hits / pred_count, 4) if pred_count else 0.0,
        "pred_midpoint_nearest_dist_mean": round(statistics.mean(midpoint_nearest), 4) if midpoint_nearest else 0.0,
        "mean_matched_iou": round(statistics.mean(matched_ious), 4) if matched_ious else 0.0,
        "mean_matched_midpoint_error_sec": round(statistics.mean(matched_mid_errors), 4) if matched_mid_errors else 0.0,
        "best_pred_iou_mean": round(statistics.mean(best_pred_ious), 4) if best_pred_ious else 0.0,
        "best_gt_iou_mean": round(statistics.mean(best_gt_ious), 4) if best_gt_ious else 0.0,
    }
    return metrics, matched_details


def aggregate_metrics(results: list[dict]) -> dict:
    successful = [r for r in results if r.get("metrics") is not None]
    n = len(successful)
    if n == 0:
        return {
            "num_episodes": len(results),
            "num_success": 0,
            "num_failures": len(results),
        }

    def _agg(key: str) -> dict:
        values = [r["metrics"][key] for r in successful]
        return {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if n > 1 else 0.0,
        }

    exact_count_hits = sum(1 for r in successful if r["metrics"]["exact_count_match"])
    all_gt_hits = sum(1 for r in successful if r["metrics"]["all_gt_matched"])
    all_pred_hits = sum(1 for r in successful if r["metrics"]["all_pred_matched"])

    agg = {
        "num_episodes": len(results),
        "num_success": n,
        "num_failures": len(results) - n,
        "exact_count_accuracy": {
            "hits": exact_count_hits,
            "total": n,
            "rate": round(exact_count_hits / n, 4),
        },
        "all_gt_matched_rate": {
            "hits": all_gt_hits,
            "total": n,
            "rate": round(all_gt_hits / n, 4),
        },
        "all_pred_matched_rate": {
            "hits": all_pred_hits,
            "total": n,
            "rate": round(all_pred_hits / n, 4),
        },
    }

    for key in (
        "pred_interval_count",
        "gt_interval_count",
        "abs_count_error",
        "matched_count",
        "segment_precision",
        "segment_recall",
        "segment_f1",
        "frame_iou",
        "gt_frame_coverage",
        "pred_frame_precision",
        "pred_midpoint_hit_rate",
        "pred_midpoint_nearest_dist_mean",
        "mean_matched_iou",
        "mean_matched_midpoint_error_sec",
        "best_pred_iou_mean",
        "best_gt_iou_mean",
    ):
        agg[key] = _agg(key)

    agg["total_pred_intervals"] = sum(r["metrics"]["pred_interval_count"] for r in successful)
    agg["total_gt_intervals"] = sum(r["metrics"]["gt_interval_count"] for r in successful)
    agg["total_matched_intervals"] = sum(r["metrics"]["matched_count"] for r in successful)

    total_cost = sum(r.get("cost", {}).get("cost_usd", 0) for r in results)
    total_input = sum(r.get("cost", {}).get("input_tokens", 0) for r in results)
    total_output = sum(r.get("cost", {}).get("output_tokens", 0) for r in results)
    total_thinking = sum(r.get("cost", {}).get("thinking_tokens", 0) for r in results)
    agg["total_cost_usd"] = round(total_cost, 4)
    agg["total_input_tokens"] = total_input
    agg["total_output_tokens"] = total_output
    agg["total_thinking_tokens"] = total_thinking
    return agg


def print_summary(agg: dict, model: str, max_intervals: int):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Gemini VLM Multi-Segment Evaluation Summary")
    print(
        f"  Model: {model}  |  Episodes: {agg['num_episodes']}  "
        f"({agg['num_success']} success, {agg['num_failures']} failures)  "
        f"|  Max intervals: {max_intervals}"
    )
    print(sep)

    if agg["num_success"] == 0:
        print("  No successful episodes to report.")
        print(sep)
        return

    print("\n  Count Prediction:")
    ecc = agg["exact_count_accuracy"]
    print(f"    Exact count match:     {ecc['rate']*100:5.1f}%  ({ecc['hits']}/{ecc['total']})")
    print(f"    Mean abs count error:  {agg['abs_count_error']['mean']:.2f}")
    print(
        f"    Mean pred / gt count:  {agg['pred_interval_count']['mean']:.2f} / "
        f"{agg['gt_interval_count']['mean']:.2f}"
    )

    print("\n  Segment Matching:")
    gtr = agg["all_gt_matched_rate"]
    pdr = agg["all_pred_matched_rate"]
    print(f"    All GT matched:        {gtr['rate']*100:5.1f}%  ({gtr['hits']}/{gtr['total']})")
    print(f"    All pred matched:      {pdr['rate']*100:5.1f}%  ({pdr['hits']}/{pdr['total']})")
    print(f"    Mean matched count:    {agg['matched_count']['mean']:.2f}")
    print(f"    Mean seg precision:    {agg['segment_precision']['mean']:.3f}")
    print(f"    Mean seg recall:       {agg['segment_recall']['mean']:.3f}")
    print(f"    Mean seg F1:           {agg['segment_f1']['mean']:.3f}")
    print(f"    Mean matched IoU:      {agg['mean_matched_iou']['mean']:.3f}")

    print("\n  Frame-Level Overlap:")
    print(f"    Mean frame IoU:        {agg['frame_iou']['mean']:.3f}")
    print(f"    Mean GT coverage:      {agg['gt_frame_coverage']['mean']:.3f}")
    print(f"    Mean pred precision:   {agg['pred_frame_precision']['mean']:.3f}")
    print(f"    Mean midpoint hit:     {agg['pred_midpoint_hit_rate']['mean']:.3f}")

    print("\n  Cost:")
    print(f"    Total: ${agg['total_cost_usd']:.4f}")
    print(f"    Per episode: ${agg['total_cost_usd']/agg['num_episodes']:.4f}")
    print(sep)


def _serialize_gt(gt: dict) -> dict:
    return {
        "fps": gt["fps"],
        "total_frames": gt["total_frames"],
        "duration_sec": round(gt["duration_sec"], 4),
        "segments": gt["segments"],
    }


def _save_debug_video(
    debug_dir: str | None,
    index: int,
    video_bytes: bytes,
    needs_overlay: bool,
) -> None:
    if debug_dir is None:
        return
    debug_video_path = os.path.join(debug_dir, f"input_video_{index}.mp4")
    if needs_overlay:
        with open(debug_video_path, "wb") as f:
            f.write(video_bytes)
    else:
        _save_as_h264(video_bytes, debug_video_path)


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
    gaze_annot: bool,
    include_audio: bool,
    audio_dir: str,
    target_resolution: tuple[int, int] | None,
    debug_dir: str | None,
    config_name: str,
    prompt_variant: str,
    max_intervals: int,
    match_frame_tolerance: int,
    verbose: bool,
) -> dict:
    episode_dir = os.path.join(dataset_dir, episode_name)
    if not os.path.isdir(episode_dir):
        return {
            "episode": episode_name,
            "index": index,
            "result": None,
            "tag": "SKIP (missing episode directory)",
            "verbose_lines": [],
        }

    gt = load_ground_truth_multi(episode_dir)
    if gt is None:
        return {
            "episode": episode_name,
            "index": index,
            "result": None,
            "tag": "SKIP (no multi-segment labels)",
            "verbose_lines": [],
        }

    transcript_path = os.path.join(episode_dir, "transcript.json")
    if not os.path.exists(transcript_path):
        return {
            "episode": episode_name,
            "index": index,
            "result": None,
            "tag": "SKIP (no transcript)",
            "verbose_lines": [],
        }
    with open(transcript_path) as f:
        transcript = json.load(f)

    gaze_frames = None
    if gaze_annot:
        gaze_path = os.path.join(episode_dir, "gaze.json")
        if os.path.exists(gaze_path):
            with open(gaze_path) as f:
                gaze_frames = json.load(f).get("frames", [])

    video_path = os.path.join(episode_dir, "video.mp4")
    if not os.path.exists(video_path):
        return {
            "episode": episode_name,
            "index": index,
            "result": None,
            "tag": "SKIP (no video)",
            "verbose_lines": [],
        }

    audio_path = None
    if include_audio:
        candidate_audio = os.path.join(audio_dir, f"{episode_name}_audio.wav")
        if os.path.exists(candidate_audio):
            audio_path = candidate_audio

    video_bytes = prepare_video(
        video_path,
        caption=caption,
        gaze_annot=gaze_annot,
        transcript=transcript,
        gaze_data=gaze_frames,
        target_resolution=target_resolution,
        include_audio=include_audio,
        audio_path=audio_path,
    )
    needs_overlay = caption or gaze_annot or (target_resolution is not None) or include_audio
    _save_debug_video(debug_dir, index, video_bytes, needs_overlay)

    prompt = build_multi_prompt(
        prompt_variant=prompt_variant,
        transcript_text=transcript["text"],
        duration_sec=gt["duration_sec"],
        fps=gt["fps"],
        config_note=PROMPT_CONFIG_NOTES[config_name],
        max_intervals=max_intervals,
    )

    client = genai.Client(api_key=api_key)
    prediction = None
    raw_text = ""
    cost_info = {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cost_usd": 0,
    }

    t0 = time.time()
    for attempt in range(3):
        try:
            response = call_gemini(
                client,
                model,
                video_bytes,
                prompt,
                video_fps=video_fps,
                thinking_budget=thinking_budget,
            )
            raw_text = response.text
            cost_info = extract_cost(response, model)
            prediction = parse_multisegment_response(response, max_intervals)
            break
        except Exception as exc:
            wait = 2 ** (attempt + 1)
            if attempt == 2:
                raw_text = str(exc)
            time.sleep(wait)
    inference_time = round(time.time() - t0, 3)

    metrics = None
    matches = []
    normalized_prediction = None
    if prediction is not None:
        normalized_prediction = normalize_prediction(prediction, gt)
        metrics, matches = evaluate_episode_multi(
            normalized_prediction,
            gt,
            match_frame_tolerance,
        )

    result = {
        "episode": episode_name,
        "gt": _serialize_gt(gt),
        "prediction": normalized_prediction,
        "metrics": metrics,
        "matches": matches,
        "cost": cost_info,
        "inference_time_sec": inference_time,
        "raw_response": raw_text if normalized_prediction is None else "",
    }

    if metrics is not None:
        tag = (
            f"matched={metrics['matched_count']}/{metrics['gt_interval_count']}  "
            f"pred={metrics['pred_interval_count']}  "
            f"count_ok={'Y' if metrics['exact_count_match'] else 'N'}  "
            f"segF1={metrics['segment_f1']:.3f}  "
            f"frameIoU={metrics['frame_iou']:.3f}  "
            f"cost=${cost_info['cost_usd']:.4f}  "
            f"time={inference_time:.1f}s"
        )
    else:
        tag = "PARSE_FAILURE"

    verbose_lines: list[str] = []
    if verbose and normalized_prediction and metrics is not None:
        pred_ranges = [
            [interval["start_frame"], interval["end_frame"]]
            for interval in normalized_prediction["intervals"]
        ]
        gt_ranges = [
            [segment["start_frame"], segment["end_frame"]]
            for segment in gt["segments"]
        ]
        verbose_lines.append(f"    Pred intervals: {pred_ranges}")
        verbose_lines.append(f"    GT intervals:   {gt_ranges}")
        if normalized_prediction.get("reasoning"):
            verbose_lines.append(
                f"    Reason: {normalized_prediction['reasoning'][:200]}"
            )

    return {
        "episode": episode_name,
        "index": index,
        "result": result,
        "tag": tag,
        "verbose_lines": verbose_lines,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gemini VLM multi-segment evaluation for keyframe detection"
    )
    parser.add_argument("--dataset-dir", default="./dataset", help="Path to dataset/ directory")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, choices=MODEL_CHOICES)
    parser.add_argument("--output-dir", default="./eval_results_multiseg")
    parser.add_argument("--resume", action="store_true", help="Skip episodes that already have results")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode names (default: all)")
    parser.add_argument("--video-fps", type=int, default=DEFAULT_GEMINI_VIDEO_FPS_HINT, help="FPS hint sent to Gemini for video sampling")
    parser.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET, help="Gemini thinking budget. Use 0 for minimal/off thinking.")
    parser.add_argument("--caption", action="store_true", help="Overlay real-time transcript captions on the video")
    parser.add_argument("--gaze-annot", action="store_true", help="Overlay gaze point (green marker) on the video")
    parser.add_argument("--include-audio", action="store_true", help="Mux sidecar audio into the MP4 sent to Gemini")
    parser.add_argument("--audio-dir", default="./preproc_files", help="Directory containing {episode}_audio.wav sidecar audio")
    parser.add_argument("--target-resolution", default="256", help="Resize video before overlay; use 'none' to keep original resolution")
    parser.add_argument("--api-key", default="", help="Gemini API key (optional if GEMINI_API_KEY or .env is set)")
    parser.add_argument("--debug", action="store_true", help="Save the exact VLM input videos")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        default="baseline",
        choices=sorted(PROMPT_VARIANT_NOTES.keys()),
        help="Prompt template variant to use for multi-segment evaluation",
    )
    parser.add_argument(
        "--exp-suffix",
        default="",
        help="Optional suffix appended to output filenames so repeated runs do not overwrite each other",
    )
    parser.add_argument("--max-intervals", type=int, default=4, help="Maximum number of intervals the VLM may return")
    parser.add_argument("--match-frame-tolerance", type=int, default=3, help="Allowed frame gap when matching predicted intervals to GT intervals")
    parser.add_argument("--workers", type=int, default=4, help="Number of episode workers for parallel VLM calls")
    args = parser.parse_args()

    target_resolution = None
    if args.target_resolution and args.target_resolution.lower() != "none":
        if "x" in args.target_resolution:
            width_str, height_str = args.target_resolution.split("x")
            target_resolution = (int(width_str), int(height_str))
        else:
            n = int(args.target_resolution)
            target_resolution = (n, n)

    if args.episodes:
        episode_names = [name.strip() for name in args.episodes.split(",") if name.strip()]
    else:
        episode_names = sorted(
            name for name in os.listdir(args.dataset_dir)
            if os.path.isdir(os.path.join(args.dataset_dir, name))
        )

    tag_parts = ["multiseg", args.model, f"fps{args.video_fps}", f"max{args.max_intervals}"]
    tag_parts.append(f"prompt-{args.prompt_variant}")
    if args.include_audio:
        tag_parts.append("audio")
    if args.caption:
        tag_parts.append("cap")
    if args.gaze_annot:
        tag_parts.append("gaze")
    if args.exp_suffix:
        tag_parts.append(args.exp_suffix)
    exp_tag = "_".join(tag_parts)

    config_name = get_config_name(args.caption, args.gaze_annot)
    print(
        f"[eval-multiseg] {len(episode_names)} episodes, model={args.model}, "
        f"video_fps={args.video_fps}, config={config_name}, prompt={args.prompt_variant}, max_intervals={args.max_intervals}"
        + (f", resolution={target_resolution}" if target_resolution else "")
        + (", audio=on" if args.include_audio else ", audio=off")
        + f", workers={max(1, args.workers)}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = os.path.join("debug", f"multiseg_{config_name}")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[debug] saving input videos to {debug_dir}/")

    results_path = os.path.join(args.output_dir, f"{exp_tag}_results.jsonl")
    completed = {}
    if args.resume:
        completed = load_existing_results(results_path)
        print(f"[resume] {len(completed)} episodes already done")

    api_key = resolve_gemini_api_key(args.api_key)
    all_results: list[dict] = []
    episode_order = {episode_name: index for index, episode_name in enumerate(episode_names)}
    pending_jobs = []
    for index, episode_name in enumerate(episode_names):
        if episode_name in completed:
            all_results.append(completed[episode_name])
            continue
        pending_jobs.append((index, episode_name))

    worker_count = max(1, min(args.workers, len(pending_jobs) or 1))
    if pending_jobs:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_episode,
                    index=index,
                    episode_name=episode_name,
                    dataset_dir=args.dataset_dir,
                    api_key=api_key,
                    model=args.model,
                    video_fps=args.video_fps,
                    thinking_budget=args.thinking_budget,
                    caption=args.caption,
                    gaze_annot=args.gaze_annot,
                    include_audio=args.include_audio,
                    audio_dir=args.audio_dir,
                    target_resolution=target_resolution,
                    debug_dir=debug_dir,
                    config_name=config_name,
                    prompt_variant=args.prompt_variant,
                    max_intervals=args.max_intervals,
                    match_frame_tolerance=args.match_frame_tolerance,
                    verbose=args.verbose,
                ): (index, episode_name)
                for index, episode_name in pending_jobs
            }

            for future in as_completed(future_map):
                index, episode_name = future_map[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    print(
                        f"  [{index+1}/{len(episode_names)}] {episode_name}: "
                        f"ERROR ({exc})"
                    )
                    continue
                print(
                    f"  [{index+1}/{len(episode_names)}] {episode_name}: "
                    f"{outcome['tag']}"
                )
                for line in outcome["verbose_lines"]:
                    print(line)
                result = outcome["result"]
                if result is None:
                    continue
                all_results.append(result)
                append_jsonl(results_path, result)

    all_results.sort(key=lambda item: episode_order.get(item["episode"], len(episode_order)))

    agg = aggregate_metrics(all_results)
    print_summary(agg, args.model, args.max_intervals)

    summary_path = os.path.join(args.output_dir, f"{exp_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "video_fps": args.video_fps,
                "thinking_budget": args.thinking_budget,
                "caption": args.caption,
                "gaze_annot": args.gaze_annot,
                "include_audio": args.include_audio,
                "prompt_variant": args.prompt_variant,
                "exp_suffix": args.exp_suffix,
                "max_intervals": args.max_intervals,
                "match_frame_tolerance": args.match_frame_tolerance,
                "workers": worker_count,
                **agg,
            },
            f,
            indent=2,
        )

    results_json_dir = "results"
    os.makedirs(results_json_dir, exist_ok=True)
    results_json_name = (
        f"result_multiseg_{args.model}_audio_{'true' if args.include_audio else 'false'}"
        f"_gaze_{'true' if args.gaze_annot else 'false'}"
        f"_cap_{'true' if args.caption else 'false'}"
        f"_fps{args.video_fps}_max{args.max_intervals}_prompt_{args.prompt_variant}.json"
    )
    if args.exp_suffix:
        stem, ext = os.path.splitext(results_json_name)
        results_json_name = f"{stem}_{args.exp_suffix}{ext}"
    results_json_path = os.path.join(results_json_dir, results_json_name)

    episodes_data = []
    for result in all_results:
        episode_entry = {"episode": result["episode"]}
        if result.get("prediction"):
            episode_entry["predicted_intervals_sec"] = [
                [interval["start_sec"], interval["end_sec"]]
                for interval in result["prediction"]["intervals"]
            ]
            episode_entry["predicted_intervals_frames"] = [
                [interval["start_frame"], interval["end_frame"]]
                for interval in result["prediction"]["intervals"]
            ]
            episode_entry["prediction_reasoning"] = result["prediction"].get("reasoning", "")
            if result["prediction"].get("num_keyframes") is not None:
                episode_entry["predicted_num_keyframes"] = result["prediction"]["num_keyframes"]
            if result["prediction"].get("object_references"):
                episode_entry["predicted_object_references"] = result["prediction"]["object_references"]
            if result["prediction"].get("subgoal_instructions"):
                episode_entry["predicted_subgoal_instructions"] = result["prediction"]["subgoal_instructions"]
        if result.get("gt"):
            episode_entry["gt_intervals_frames"] = [
                [segment["start_frame"], segment["end_frame"]]
                for segment in result["gt"]["segments"]
            ]
        if result.get("metrics"):
            episode_entry.update(result["metrics"])
        if result.get("matches"):
            episode_entry["matches"] = result["matches"]
        episode_entry["inference_time_sec"] = result.get("inference_time_sec")
        episodes_data.append(episode_entry)

    with open(results_json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "model": args.model,
                    "include_audio": args.include_audio,
                    "gaze_annot": args.gaze_annot,
                    "caption": args.caption,
                    "video_fps": args.video_fps,
                    "config_name": config_name,
                    "prompt_variant": args.prompt_variant,
                    "max_intervals": args.max_intervals,
                    "match_frame_tolerance": args.match_frame_tolerance,
                    "workers": worker_count,
                },
                "episodes": episodes_data,
                "aggregate": agg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n[saved] {results_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {results_json_path}")


if __name__ == "__main__":
    main()
