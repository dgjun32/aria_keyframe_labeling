#!/usr/bin/env python3
"""Asynchronous keyframe-queue simulation using periodic VLM calls.

The script simulates a real-time scenario where:
1. The first 10 s of video are sent to a VLM to **initialize** a robot-action
   keyframe queue (pick-up / place-it pairs).
2. Every 10 s of *local* timer thereafter, the latest 10-second video clip is
   sent to the VLM which may **update** the queue (NoOp / Append / Replace /
   Remove).
3. Wall-clock latency of each VLM call advances the simulated global clock,
   mimicking real-time behaviour.

Outputs per episode:
    eval_results_async/{episode}/
        keyframe_queue_status.json   – full log of every VLM interaction
        offline_simulation.mp4       – side-by-side video + queue visualisation
        segmented_videos/            – each 10-s clip sent to the VLM
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import os
import re
import subprocess
import tempfile
import time
import warnings
from typing import Any

import cv2
import numpy as np
from google import genai

from openai import OpenAI

from eval_vlm_baseline import (
    DEFAULT_THINKING_BUDGET,
    VIDEO_FPS,
    _draw_caption,
    _get_active_caption,
    _ffmpeg_reencode_h264,
    _save_as_h264,
    build_thinking_config,
    call_gemini,
    extract_cost,
    resolve_gemini_api_key,
    resolve_openai_api_key,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════


def _extract_json_string(text: str) -> str:
    """Extract JSON from markdown code fences or raw text."""
    # Try ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find outermost { ... } or [ ... ]
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text.strip()


AUDIO_MODE_NONE = "none"
AUDIO_MODE_WHISPER = "whisper"  # Option 1: Whisper STT → caption overlay + prompt text
AUDIO_MODE_VOICED = (
    "voiced-video"  # Option 2: merge mp4+wav → send audio to Gemini natively
)
AUDIO_MODE_WHISPER_VOICED = "whisper+voiced-video"
SIMULATION_TIMING_FIXED = "fixed_cadence"
SIMULATION_TIMING_REAL = "real_latency"
SIMULATION_TIMING_CHOICES = [SIMULATION_TIMING_FIXED, SIMULATION_TIMING_REAL]

SEGMENT_DURATION_SEC = 10.0
INIT_SEGMENT_SEC = 10.0
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
MODEL_CHOICES = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]

DEFAULT_TASK_VOCAB_HINTS = """\
SCENE-SPECIFIC TASK VOCABULARY:
- If the pointed target matches one of the canonical objects or baskets below, use the exact subtask_instruction string shown here instead of paraphrasing.
- Accuracy is more important than recall. Never force a canonical match just because an object looks similar.
- The scene evidence, finger-pointing evidence, and task description must agree exactly before you use one of the canonical task strings below.
- Do not map to a neighbouring grid cell, a similarly colored item, or the wrong basket.
- If the target is ambiguous, partially occluded, or not exactly aligned with the scene/task evidence, do NOT guess. Fall back to a descriptive phrase or remain conservative.
- Coordinate convention is from the wearer's point of view.
- Row indices increase from top to bottom.
- Column indices increase from left to right.
- The tabletop pick targets form a 2x3 grid.
- (row 1, column 1) is the top-left target in the image.
- (row 1, column 3) is the top-right target in the image.
- (row 2, column 1) is the bottom-left target in the image.
- (row 2, column 3) is the bottom-right target in the image.
- "top basket" means the basket farther from the wearer / higher in the image.
- "bottom basket" means the basket closer to the wearer / lower in the image.
- Determine the intended canonical object primarily from where the human's finger/hand is pointing.
- If the finger is ambiguous, use gaze and speech only as tie-breakers, not as the primary target selector.
- When the finger clearly indicates one grid cell, map that cell to the corresponding canonical object string below.

Preferred pick strings:
- pick up the red pringles at row 1, column 1
- pick up the green pringles at row 1, column 2
- pick up the purple pringles at row 1, column 3
- pick up the yellow jelly at row 2, column 3

Preferred place strings:
- place it to the top basket
- place it to the bottom basket

If the human refers to a target outside this vocabulary, fall back to the normal descriptive rule.
"""


def audio_mode_has_whisper(audio_mode: str) -> bool:
    return audio_mode in (AUDIO_MODE_WHISPER, AUDIO_MODE_WHISPER_VOICED)


def audio_mode_has_native_audio(audio_mode: str) -> bool:
    return audio_mode in (AUDIO_MODE_VOICED, AUDIO_MODE_WHISPER_VOICED)


def load_task_vocab_hints(task_vocab_file: str | None) -> str:
    """Load optional scene/task vocabulary hints from a text file."""
    if not task_vocab_file:
        return DEFAULT_TASK_VOCAB_HINTS

    try:
        with open(task_vocab_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read task vocabulary file: {task_vocab_file} ({exc})"
        ) from exc

    return text or DEFAULT_TASK_VOCAB_HINTS

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class KeyframeQueueElement:
    """A single element in the robot-action keyframe queue."""

    keyframe_description: str  # fine-grained visual description
    subtask_instruction: str  # "pick up X" or "place it Y"
    start_sec: float  # interval start on the ORIGINAL video timeline
    end_sec: float  # interval end on the ORIGINAL video timeline
    keyframe_image_path: str  # path to saved keyframe PNG
    pointing_peak_sec: float | None = None  # exact moment finger most clearly indicates target on the original video timeline

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def _parse_interval(cls, d: dict) -> tuple[float, float]:
        """Extract (start, end) seconds from VLM response dict."""
        interval = d.get("keyframe_interval", {})
        if isinstance(interval, dict) and "start" in interval:
            start = float(interval["start"])
            end = float(interval.get("end", start))
        else:
            start = float(
                d.get("start_sec", d.get("start", d.get("keyframe_time_sec", 0.0)))
            )
            end = float(d.get("end_sec", d.get("end", start)))
        if end < start:
            start, end = end, start
        return round(start, 3), round(end, 3)

    @classmethod
    def _parse_peak_sec(cls, d: dict, start: float, end: float) -> float | None:
        """Extract the exact pointing moment, clamped to the interval."""
        value = d.get("pointing_peak_sec", d.get("keyframe_time_sec"))
        if value is None:
            return None
        try:
            peak = float(value)
        except (TypeError, ValueError):
            return None
        peak = max(start, min(end, peak))
        return round(peak, 3)


def extract_keyframe_from_clip(
    clip_bytes: bytes,
    start_sec: float,
    end_sec: float,
    save_path: str,
    target_sec: float | None = None,
) -> str:
    """Extract mid-frame from clip bytes and save as PNG.

    start_sec/end_sec are clip-relative. Returns save_path.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(clip_bytes)
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_sec = target_sec if target_sec is not None else (start_sec + end_sec) / 2.0
    frame_sec = max(start_sec, min(end_sec, frame_sec))
    mid_frame = int(round(frame_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    os.unlink(tmp.name)

    if ret and frame is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, frame)
    else:
        # Fallback: black image
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, np.zeros((64, 64, 3), dtype=np.uint8))

    return save_path


def build_queue_elements(
    parsed_elements: list[dict],
    clip_bytes: bytes,
    keyframe_dir: str,
    prefix: str = "kf",
    clip_start_sec: float = 0.0,
) -> list[KeyframeQueueElement]:
    """Parse VLM output dicts and extract keyframe images from clip.

    Returns list of KeyframeQueueElement with saved PNG paths.
    """
    os.makedirs(keyframe_dir, exist_ok=True)
    elements = []
    for i, d in enumerate(parsed_elements):
        rel_start, rel_end = KeyframeQueueElement._parse_interval(d)
        rel_peak_sec = KeyframeQueueElement._parse_peak_sec(d, rel_start, rel_end)
        abs_start = round(clip_start_sec + rel_start, 3)
        abs_end = round(clip_start_sec + rel_end, 3)
        abs_peak_sec = (
            round(clip_start_sec + rel_peak_sec, 3)
            if rel_peak_sec is not None
            else None
        )
        img_path = os.path.join(
            keyframe_dir, f"{prefix}_{i}_{abs_start:.2f}s_{abs_end:.2f}s.png"
        )
        extract_keyframe_from_clip(
            clip_bytes,
            rel_start,
            rel_end,
            img_path,
            target_sec=rel_peak_sec,
        )
        elements.append(
            KeyframeQueueElement(
                keyframe_description=str(d.get("keyframe_description", "")),
                subtask_instruction=str(d.get("subtask_instruction", "")),
                start_sec=abs_start,
                end_sec=abs_end,
                pointing_peak_sec=abs_peak_sec,
                keyframe_image_path=img_path,
            )
        )
    return elements


class KeyframeQueue:
    """Managed queue of robot-action keyframe elements."""

    def __init__(self) -> None:
        self._elements: list[KeyframeQueueElement] = []

    # ── mutations ──

    def initialize(self, elements: list[KeyframeQueueElement]) -> None:
        self._elements = list(elements)

    def append(self, elements: list[KeyframeQueueElement]) -> None:
        self._elements.extend(elements)

    def replace(self, target_idx: int, new_element: KeyframeQueueElement) -> None:
        if 0 <= target_idx < len(self._elements):
            self._elements[target_idx] = new_element
        else:
            print(
                f"  [WARN] Replace idx {target_idx} out of range (queue len={len(self._elements)}), skipping"
            )

    def remove(self, target_idx: int) -> None:
        if 0 <= target_idx < len(self._elements):
            self._elements.pop(target_idx)
        else:
            print(
                f"  [WARN] Remove idx {target_idx} out of range (queue len={len(self._elements)}), skipping"
            )

    def apply_progress_and_update(
        self,
        *,
        completed_indices: list[int] | None,
        action: str,
        target_idx: int | None,
        new_elements: list[KeyframeQueueElement],
    ) -> None:
        """Apply progress + intervention using indices from the ORIGINAL queue snapshot.

        `completed_indices` and `target_idx` both refer to the queue ordering as shown
        to the VLM in the prompt, before this update is applied.
        """
        original = list(self._elements)
        valid_completed = {
            idx for idx in (completed_indices or []) if 0 <= idx < len(original)
        }

        if target_idx is not None and not (0 <= target_idx < len(original)):
            print(
                f"  [WARN] target_idx {target_idx} out of range "
                f"(queue len={len(original)}), ignoring intervention target"
            )
            target_idx = None

        updated: list[KeyframeQueueElement] = []
        for idx, elem in enumerate(original):
            if idx in valid_completed:
                continue

            if action == "Remove" and target_idx == idx:
                continue

            if action == "Replace" and target_idx == idx:
                if new_elements:
                    updated.append(new_elements[0])
                else:
                    print(
                        f"  [WARN] Replace requested for idx {idx} but no new element provided"
                    )
                    updated.append(elem)
                continue

            updated.append(elem)

        if action == "Append" and new_elements:
            updated.extend(new_elements)

        self._elements = updated

    def noop(self) -> None:
        pass

    # ── read ──

    def snapshot(self) -> list[dict]:
        return [e.to_dict() for e in self._elements]

    def format_for_prompt(self) -> str:
        if not self._elements:
            return "(queue is empty)"
        lines = []
        for i, e in enumerate(self._elements):
            lines.append(
                f"[{i}] {e.subtask_instruction} | keyframe: {e.keyframe_description} "
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._elements)

    def __getitem__(self, idx: int) -> KeyframeQueueElement:
        return self._elements[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_INIT = """\
You are analyzing the first {clip_duration:.1f} seconds of a first-person \
(egocentric) video recorded in a store. A human is giving instructions to a \
robot by pointing at objects and speaking.

Your task has two steps. You MUST complete them in order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Step 1 — Visual Grounding: Extract keyframes**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Watch the video and identify every moment when the human points at or \
indicates an object/location. Each such moment is a **keyframe**.

Use the **convergence of multiple cues** to pinpoint each keyframe:
  - Utterance timing: when the spoken instruction refers to a target \
(words like "this", "that", or naming an object).
  - Hand gesture: when the hand/finger clearly points to a specific \
object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target \
(if gaze overlay is available).

Prioritise intervals where hand pointing and gaze are spatially aligned \
(same object/region). Use utterance timing to confirm which object is meant.

Accuracy is critical. Do NOT loosely match the scene to a convenient task \
label. The object/location you describe must exactly match what is visually \
indicated in the clip. If the target is ambiguous, partially visible, or only \
approximately matches a known task target, do not guess.

Each keyframe interval should be short and precise (0.2–2.0 s). \
Do not merge separate pointing events into one long interval.
For each keyframe, you must also output the exact instant when the finger most \
clearly points to the target.

Record each keyframe with:
  - **keyframe_description**: fine-grained visual description of what \
the human is pointing at (color, shape, brand, size, shelf position, \
distinguishing features). E.g., "small red cupcake with white frosting \
on the 2nd shelf from top, left side".
  - **keyframe_interval**: {{"start": <float>, "end": <float>}} in seconds.
  - **pointing_peak_sec**: the single timestamp inside keyframe_interval when \
the fingertip most precisely indicates the intended target.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Step 2 — Contextual Recomposition: Build action queue**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the keyframes from Step 1, construct a robot **action queue** — \
the sequence of atomic pick-and-place steps a robot would execute.

The human instruction may reference multiple objects and destinations in \
a single sentence. You must decompose it into individual steps. Examples:

  - "Put this, this, and this to that basket" \
→ 3 keyframes (obj1, obj2, obj3) + 1 keyframe (basket) \
→ 6 action queue elements:
    [pick up obj1, place it in basket,
     pick up obj2, place it in basket,
     pick up obj3, place it in basket]

  - "Give me this cupcake and this bottle" \
→ 2 keyframes → 2 action queue elements:
    [pick up the red cupcake, pick up the green bottle]

  - "Put this here" \
→ 2 keyframes (object + destination) → 2 action queue elements:
    [pick up <object>, place it <destination>]

RULES for subtask_instruction:
  - MUST use exactly one of two verb forms:
    1. "pick up <object>"
    2. "place it <destination>"
  - Do NOT use any other verb (e.g., "point to", "grab", "put", "get", \
"move", "take"). Always rephrase to "pick up" or "place it".

Each action queue element inherits the keyframe_interval from the \
corresponding keyframe in Step 1. For recomposed elements (e.g., \
"place it in basket" repeated for multiple objects), use the \
keyframe_interval of the destination keyframe.

When mapping a keyframe to a concrete task step, require an exact match \
between the observed scene, the pointed target, and the task description. \
Do not substitute a nearby/similar object or destination just to fit the task vocabulary.

{task_vocabulary_section}

{transcript_section}
{video_view_section}
The video is {clip_duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<brief explanation of cues used in Step 1>",
  "human_instruction_summary": "<one-line summary of what the human asked>",
  "keyframe_queue": [
    {{
      "keyframe_description": "<detailed visual description of what is pointed at>",
      "keyframe_interval": {{"start": <float>, "end": <float>}},
      "pointing_peak_sec": <float>
    }}
  ],
  "action_queue": [
    {{
      "keyframe_description": "<detailed visual description>",
      "subtask_instruction": "<pick up X or place it Y>",
      "keyframe_interval": {{"start": <float>, "end": <float>}},
      "pointing_peak_sec": <float>
    }}
  ]
}}

Important:
  - All timestamps are relative to this clip (0.0 = beginning of clip).
  - keyframe_queue contains ONLY the raw pointing moments (N keyframes).
  - action_queue contains the recomposed robot steps (may be longer than \
keyframe_queue due to pick-up/place-it pairing).
  - Accuracy is more important than coverage. Do not invent or force extra task steps.
  - Intervals must be sorted chronologically.
  - Each interval should be short and precise (0.2–2.0 seconds).
  - Always return positive start/end values.
  - pointing_peak_sec must lie inside the corresponding keyframe_interval.
"""


PROMPT_UPDATE = """\
You are monitoring a robot task execution from a first-person (egocentric) \
video in a store. You are receiving a new {clip_duration:.1f}-second video clip \
showing what just happened (covering {clip_start_sec:.1f}s to \
{clip_end_sec:.1f}s of the original video).

The robot is executing a pick-and-place task. The current keyframe queue \
(ordered list of remaining subtask steps) is:

{queue_status}

The queue order above is the remaining execution order. Use the queue index \
to decide target_idx for Replace or Remove.

You must reason about TWO things separately:
1. **Execution progress**: which current queue steps were actually completed in this clip.
2. **Human intervention**: whether the human changed/added/cancelled tasks in this clip.

Work in this order:
1. Write a short factual clip_summary.
2. List the concrete observed_events in chronological order.
3. Decide which current queue steps are completed, using ONLY those factual observations.
4. Separately decide whether there is a human intervention (NoOp / Append / Replace / Remove).

Your task: Analyze the new video clip and decide whether the queue needs to \
be modified. A human may intervene during the task to change, add, or cancel \
steps. Look for:
- Human speech or gestures indicating a change in the task
- Pointing at new objects or destinations
- Verbal corrections ("not that one", "also grab this", "forget about that")

Choose exactly ONE action:

1. **NoOp**: No change needed. The robot is working normally, no human \
intervention detected, or the person is simply observing without meaningful \
instruction.

2. **Append**: Add new element(s) to the END of the queue. Use when the \
human clearly and explicitly adds new objects/destinations to the task.
Do NOT append based on a weak guess, vague motion, or a merely similar object.

3. **Replace**: Replace a specific queue element at a given index. Use when \
the human changes a target object or destination for a specific step.
The replacement means that the task at that queue index has changed to a new \
task. The new replacement element must contain the updated subtask_instruction \
and updated keyframe_description for the new target/destination.

4. **Remove**: Remove a specific queue element at a given index. Use when \
the human cancels a specific step.

The keyframe intervals are determined by the **convergence of multiple cues**:
  - Utterance timing: when the instruction refers to the target object.
  - Hand gesture (pointing): when the hand/finger clearly points to a \
specific object or region.
  - Gaze fixation: when the wearer's gaze stabilises on the target object \
(if gaze information is available).

Prioritise intervals where the object indicated by hand pointing and the gaze \
point are spatially aligned (same object/region). Use utterance timing to \
confirm that this aligned object is the instructed target.

Accuracy is critical. Do NOT loosely match the observed scene to a convenient \
task string. Only update the queue when the scene evidence and task meaning \
match clearly and specifically.

Each interval should be short and precise, typically 0.2-2.0 seconds.
For each new or replaced element, you must also output the exact instant when \
the finger most clearly points to that target.

RULES:
- subtask_instruction MUST use ONLY "pick up <object>" or "place it <destination>"
- Do NOT use any other verbs (e.g., "point to", "grab", "put", "get", \
"move", "take"). Always rephrase to "pick up" or "place it".
- keyframe_description must be fine-grained with visual attributes (color, \
shape, brand name, size, shelf position, etc.)
- The timestamps you output inside new_elements must be **relative to this \
clip** (0.0 = start of this clip, {clip_duration:.1f} = end of this clip)
- pointing_peak_sec must be the exact finger-pointing moment and must lie \
inside keyframe_interval
- Apply contextual recomposition: if the human says "also put this in that \
basket", append TWO elements (pick up + place it), not one
- Prefer NoOp over a speculative Append/Replace/Remove.
- For Replace or Remove, only target a queue index when the intervention clearly refers to that specific remaining step.
- For Replace, the replacement element must clearly describe what the task changed FROM/TO at that queue position.
- Never use a canonical task label unless the pointed scene and the task description match exactly.
- For Append, only append when the new target/destination is clearly introduced and visually grounded in this clip.
- completed_queue_indices must be derived from clip_summary and observed_events, not from a guess.
- completed_queue_indices must refer to steps that were clearly executed in this clip, not just started or approached.
- For a pick step, mark it completed only if the object is securely grasped and clearly lifted/moved off its original support.
- For a place step, mark it completed only if the object reaches the destination and is released / left stably there.
- Do NOT mark a step completed from reaching, hovering, pointing, touching, partial contact, or carrying alone.
- If the completion evidence is weak or ambiguous, leave completed_queue_indices empty.
- Both completed_queue_indices and target_idx refer to the queue exactly as shown above, before this update is applied.

{task_vocabulary_section}

{transcript_section}
{video_view_section}
This video clip is {clip_duration:.1f} seconds long (recorded at {fps:.1f} FPS).

Return your answer as JSON:
{{
  "reasoning": "<explanation of what you observed in the clip>",
  "clip_summary": "<brief factual summary of what happened in this clip>",
  "observed_events": [
    "<short event 1>",
    "<short event 2>"
  ],
  "completion_evidence": [
    {{
      "queue_idx": <int>,
      "evidence": "<short factual evidence grounded in clip_summary / observed_events>"
    }}
  ],
  "completed_queue_indices": [<int>, <int>],
  "action": "<NoOp|Append|Replace|Remove>",
  "target_idx": <int or null>,
  "new_elements": [
    {{
      "keyframe_description": "<detailed visual description>",
      "subtask_instruction": "<pick up X or place it Y>",
      "keyframe_interval": {{"start": <float, seconds>, "end": <float, seconds>}},
      "pointing_peak_sec": <float, seconds>
    }}
  ]
}}

completed_queue_indices:
  - list of current queue indices that were clearly completed during this clip
  - use [] if no current queue step was clearly completed

completion_evidence:
  - include one item for every completed queue index
  - each evidence string must be factual and directly supported by the clip
  - if you cannot give concrete evidence, do not mark that queue index as completed

You may return action="NoOp" while still listing completed_queue_indices if \
the robot simply executed planned steps without any human intervention.

For NoOp: set target_idx to null, new_elements to []
For Append: set target_idx to null, new_elements to the list of new elements
For Replace: set target_idx to the index to replace, new_elements to \
[single replacement element]
For Remove: set target_idx to the index to remove, new_elements to []
"""


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODE FILE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════


def resolve_episode_files(dataset_dir: str, episode_name: str) -> dict[str, str]:
    """Return paths to all files for an episode."""
    base = os.path.join(dataset_dir, episode_name)
    return {
        "rgb": f"{base}_rgb.mp4",
        "rgb_with_gaze": f"{base}_rgb_with_gaze.mp4",
        "gaze": f"{base}_gaze.npz",
        "audio": f"{base}_audio.wav",
        "pitch_yaw": f"{base}_pitch_yaw.npz",
    }


def list_episode_names(dataset_dir: str) -> list[str]:
    """Discover unique episode names from the dataset directory."""
    names = []
    suffix = "_rgb.mp4"
    for f in sorted(os.listdir(dataset_dir)):
        if f.endswith(suffix) and not f.endswith("_with_gaze_rgb.mp4"):
            names.append(f[: -len(suffix)])
    return names


def load_gaze_points(gaze_path: str) -> np.ndarray | None:
    """Load per-frame gaze points from .npz as shape [N, 2]."""
    if not gaze_path or not os.path.exists(gaze_path):
        return None
    try:
        data = np.load(gaze_path)
        if "gaze" not in data.files:
            return None
        gaze = np.asarray(data["gaze"], dtype=np.float32)
        if gaze.ndim != 2 or gaze.shape[1] != 2:
            return None
        return gaze
    except Exception as exc:
        print(f"  [WARN] Failed to load gaze data from {gaze_path}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def get_video_meta(video_path: str) -> tuple[int, float]:
    """Return (total_frames, fps) for a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total, fps


def cut_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str | None = None,
    target_resolution: tuple[int, int] | None = None,
    target_fps: float | None = None,
    gaze_points: np.ndarray | None = None,
    gaze_crop_zoom: float = 1.0,
) -> bytes:
    """Cut video[start_sec:end_sec], optionally resize and subsample, return H264 bytes.

    If *target_fps* is set and lower than original fps, frames are subsampled
    so the output video truly has fewer frames (reducing data sent to VLM).
    Also saves to *output_path* if given.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    out_w, out_h = target_resolution if target_resolution else (orig_w, orig_h)
    out_fps = target_fps if (target_fps and target_fps < fps) else fps
    # Frame step: e.g. original 15fps, target 4fps → keep every ~3.75th frame
    frame_step = fps / out_fps if out_fps < fps else 1.0

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, out_fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_src_frames = end_frame - start_frame
    next_sample = 0.0  # accumulator for subsampling
    crop_enabled = gaze_points is not None and gaze_crop_zoom > 1.0
    crop_size = int(round(min(orig_w, orig_h) / gaze_crop_zoom)) if crop_enabled else 0
    crop_size = max(32, min(crop_size, min(orig_w, orig_h))) if crop_enabled else 0
    center_x = orig_w / 2.0
    center_y = orig_h / 2.0

    def _extract_gaze_crop(frame: np.ndarray, abs_frame_idx: int) -> np.ndarray:
        nonlocal center_x, center_y

        if (
            gaze_points is not None
            and 0 <= abs_frame_idx < len(gaze_points)
            and np.isfinite(gaze_points[abs_frame_idx]).all()
        ):
            gx = float(gaze_points[abs_frame_idx][0])
            gy = float(gaze_points[abs_frame_idx][1])
            if 0 <= gx < orig_w and 0 <= gy < orig_h:
                center_x = 0.7 * center_x + 0.3 * gx
                center_y = 0.7 * center_y + 0.3 * gy

        half = crop_size / 2.0
        cx = min(max(center_x, half), orig_w - half)
        cy = min(max(center_y, half), orig_h - half)
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = min(orig_w, x1 + crop_size)
        y2 = min(orig_h, y1 + crop_size)
        x1 = max(0, x2 - crop_size)
        y1 = max(0, y2 - crop_size)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    for src_idx in range(total_src_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if src_idx >= next_sample:
            abs_frame_idx = start_frame + src_idx
            if crop_enabled:
                frame = _extract_gaze_crop(frame, abs_frame_idx)
            elif target_resolution and (
                frame.shape[1] != out_w or frame.shape[0] != out_h
            ):
                frame = cv2.resize(
                    frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4
                )
            writer.write(frame)
            next_sample += frame_step

    writer.release()
    cap.release()

    # Re-encode to H264
    h264_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_tmp.close()
    _ffmpeg_reencode_h264(tmp.name, h264_tmp.name)
    os.unlink(tmp.name)

    with open(h264_tmp.name, "rb") as f:
        video_bytes = f.read()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(video_bytes)

    os.unlink(h264_tmp.name)
    return video_bytes


def extract_frame(
    video_path: str, frame_idx: int, size: tuple[int, int] | None = None
) -> np.ndarray | None:
    """Extract a single frame from a video, optionally resize. Returns BGR."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    if size and (frame.shape[1] != size[0] or frame.shape[0] != size[1]):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def cut_audio_segment(audio_path: str, start_sec: float, end_sec: float) -> str:
    """Cut audio_path[start_sec:end_sec] to a temp WAV file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    duration = end_sec - start_sec
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            audio_path,
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            tmp.name,
        ],
        check=True,
    )
    return tmp.name


def merge_video_audio(video_bytes: bytes, audio_segment_path: str) -> bytes:
    """Merge H264 video bytes with an audio WAV file → MP4 with audio track."""
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video.write(video_bytes)
    tmp_video.close()

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            tmp_video.name,
            "-i",
            audio_segment_path,
            "-c:v",
            "copy",
            "-ac",
            "1",  # downmix to mono
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-shortest",
            tmp_out.name,
        ],
        check=True,
    )

    with open(tmp_out.name, "rb") as f:
        result = f.read()

    os.unlink(tmp_video.name)
    os.unlink(tmp_out.name)
    return result


def whisper_transcribe_segment(
    audio_path: str, start_sec: float, end_sec: float
) -> dict:
    """Transcribe a segment of audio using OpenAI Whisper API.

    Returns {"text": "...", "words": [{"word": "...", "start": <abs>, "end": <abs>}]}
    where start/end are in the ORIGINAL video timeline (offset by start_sec).
    """
    seg_path = cut_audio_segment(audio_path, start_sec, end_sec)
    try:
        client = OpenAI()  # uses OPENAI_API_KEY env var
        with open(seg_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        result = {"text": response.text, "words": []}
        if hasattr(response, "words") and response.words:
            for w in response.words:
                # Offset timestamps to original video timeline
                result["words"].append(
                    {
                        "word": w.word,
                        "start": round(w.start + start_sec, 3),
                        "end": round(w.end + start_sec, 3),
                    }
                )
        return result
    except Exception as exc:
        print(f"  [WARN] Whisper transcription failed: {exc}")
        return {"text": "", "words": []}
    finally:
        os.unlink(seg_path)


def burn_captions_on_video(
    video_bytes: bytes,
    words: list[dict],
    clip_start_sec: float,
    fps: float,
) -> bytes:
    """Burn word-level captions onto video frames. Returns re-encoded H264 bytes.

    *words* have absolute timestamps (original video timeline).
    *clip_start_sec* is the absolute start time of this clip.
    """
    # Decode video from bytes
    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(video_bytes)
    tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out.name, fourcc, vid_fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        abs_time = clip_start_sec + frame_idx / vid_fps
        caption = _get_active_caption(words, abs_time)
        if caption:
            _draw_caption(frame, caption)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    os.unlink(tmp_in.name)

    # Re-encode to H264
    h264_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h264_tmp.close()
    _ffmpeg_reencode_h264(tmp_out.name, h264_tmp.name)
    os.unlink(tmp_out.name)

    with open(h264_tmp.name, "rb") as f:
        result = f.read()
    os.unlink(h264_tmp.name)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  VLM INTERACTION
# ══════════════════════════════════════════════════════════════════════════════


def call_gemini_with_timing(
    client: genai.Client,
    model_name: str,
    video_bytes: bytes,
    prompt: str,
    video_fps: int = 2,
    has_audio: bool = False,
) -> tuple[Any, str, float]:
    """Call Gemini, return (response, raw_text, wall_clock_seconds).

    Same config as eval_vlm_baseline.call_gemini, but when *has_audio* is
    True the VideoMetadata is omitted so Gemini processes the audio track.
    """
    from google.genai import types as _types

    if has_audio:
        # No VideoMetadata → Gemini reads the audio track from the mp4
        video_part = _types.Part(
            inline_data=_types.Blob(data=video_bytes, mime_type="video/mp4"),
        )
    else:
        video_part = _types.Part(
            inline_data=_types.Blob(data=video_bytes, mime_type="video/mp4"),
            video_metadata=_types.VideoMetadata(fps=video_fps),
        )

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=_types.Content(parts=[video_part, _types.Part(text=prompt)]),
        config=_types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=build_thinking_config(
                model_name,
                -1,
            ),
            response_mime_type="application/json",
        ),
    )
    latency = time.time() - t0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_text = response.text

    return response, raw_text, round(latency, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE PARSERS
# ══════════════════════════════════════════════════════════════════════════════


def parse_init_response(raw_text: str) -> dict | None:
    """Parse initialization response.

    Expected JSON keys: reasoning, human_instruction_summary,
    keyframe_queue (Step 1), action_queue (Step 2).
    Returns None on failure.
    """
    json_str = _extract_json_string(raw_text)
    try:
        payload = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    action_queue = payload.get("action_queue")
    if not isinstance(action_queue, list) or len(action_queue) == 0:
        return None

    return {
        "reasoning": payload.get("reasoning", ""),
        "human_instruction_summary": payload.get("human_instruction_summary", ""),
        "keyframe_queue": payload.get("keyframe_queue", []),  # Step 1 raw keyframes
        "action_queue": action_queue,  # Step 2 recomposed actions
    }


def parse_update_response(raw_text: str) -> dict | None:
    """Parse update response.

    Expected JSON keys: reasoning, action, target_idx, new_elements.
    Returns None on failure.
    """
    json_str = _extract_json_string(raw_text)
    try:
        payload = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None

    action = payload.get("action", "")
    if action not in ("NoOp", "Append", "Replace", "Remove"):
        return None

    completion_evidence = payload.get("completion_evidence", [])
    parsed_completion_evidence: list[dict] = []
    evidence_indices: set[int] = set()
    if isinstance(completion_evidence, list):
        for item in completion_evidence:
            if not isinstance(item, dict):
                continue
            try:
                queue_idx = int(item.get("queue_idx"))
            except (TypeError, ValueError):
                continue
            evidence = str(item.get("evidence", "")).strip()
            if not evidence:
                continue
            parsed_completion_evidence.append(
                {
                    "queue_idx": queue_idx,
                    "evidence": evidence,
                }
            )
            evidence_indices.add(queue_idx)

    completed_queue_indices = payload.get("completed_queue_indices", [])
    if not isinstance(completed_queue_indices, list):
        completed_queue_indices = []
    coerced_completed: list[int] = []
    for value in completed_queue_indices:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx in evidence_indices:
            coerced_completed.append(idx)

    observed_events = payload.get("observed_events", [])
    if not isinstance(observed_events, list):
        observed_events = []
    observed_events = [str(x) for x in observed_events if str(x).strip()]

    return {
        "reasoning": payload.get("reasoning", ""),
        "clip_summary": payload.get("clip_summary", ""),
        "observed_events": observed_events,
        "completion_evidence": parsed_completion_evidence,
        "completed_queue_indices": coerced_completed,
        "action": action,
        "target_idx": payload.get("target_idx"),
        "new_elements": payload.get("new_elements", []),
    }


def apply_queue_update(
    queue: KeyframeQueue,
    parsed: dict,
    clip_bytes: bytes,
    keyframe_dir: str,
    step: int,
    clip_start_sec: float,
) -> str:
    """Apply a parsed update action to the queue. Returns action name."""
    action = parsed["action"]
    completed_indices = parsed.get("completed_queue_indices", [])
    target_idx = parsed.get("target_idx")
    built_new_elements: list[KeyframeQueueElement] = []

    if action == "NoOp":
        queue.apply_progress_and_update(
            completed_indices=completed_indices,
            action=action,
            target_idx=None,
            new_elements=[],
        )
    elif action == "Append":
        built_new_elements = build_queue_elements(
            parsed["new_elements"],
            clip_bytes,
            keyframe_dir,
            prefix=f"step{step}_append",
            clip_start_sec=clip_start_sec,
        )
        queue.apply_progress_and_update(
            completed_indices=completed_indices,
            action=action,
            target_idx=None,
            new_elements=built_new_elements,
        )
    elif action == "Replace":
        elems = parsed.get("new_elements", [])
        if target_idx is not None and len(elems) > 0:
            built_new_elements = build_queue_elements(
                [elems[0]],
                clip_bytes,
                keyframe_dir,
                prefix=f"step{step}_replace",
                clip_start_sec=clip_start_sec,
            )
            queue.apply_progress_and_update(
                completed_indices=completed_indices,
                action=action,
                target_idx=int(target_idx),
                new_elements=built_new_elements,
            )
        else:
            print(f"  [WARN] Replace action missing target_idx or new_elements")
            queue.apply_progress_and_update(
                completed_indices=completed_indices,
                action="NoOp",
                target_idx=None,
                new_elements=[],
            )
    elif action == "Remove":
        if target_idx is not None:
            queue.apply_progress_and_update(
                completed_indices=completed_indices,
                action=action,
                target_idx=int(target_idx),
                new_elements=[],
            )
        else:
            print(f"  [WARN] Remove action missing target_idx")
            queue.apply_progress_and_update(
                completed_indices=completed_indices,
                action="NoOp",
                target_idx=None,
                new_elements=[],
            )
    return action


def _with_display_status(elem: dict, status: str) -> dict:
    item = dict(elem)
    item["_display_status"] = status
    return item


def build_display_queue_snapshot(
    previous_display: list[dict] | None,
    original_queue: list[dict],
    updated_queue: list[dict],
    action: str,
    action_details: dict | None,
) -> list[dict]:
    """Build a visualization-only queue snapshot.

    Completed/removed steps stay visible in the visualization instead of
    disappearing from the rendered queue. The actual execution queue still uses
    `updated_queue`.
    """
    if previous_display is None:
        return [_with_display_status(elem, "pending") for elem in updated_queue]

    display = [dict(item) for item in previous_display]
    pending_positions = [
        i
        for i, item in enumerate(display)
        if item.get("_display_status") not in {"completed", "removed"}
    ]

    if len(pending_positions) != len(original_queue):
        inactive_items = [
            dict(item)
            for item in display
            if item.get("_display_status") in {"completed", "removed"}
        ]
        return inactive_items + [
            _with_display_status(elem, "pending") for elem in updated_queue
        ]

    details = action_details or {}
    completed_indices = []
    for idx in details.get("completed_queue_indices", []):
        if isinstance(idx, int) and 0 <= idx < len(pending_positions):
            completed_indices.append(idx)
    completed_set = set(completed_indices)
    target_idx = details.get("target_idx")

    for idx in sorted(completed_set):
        pos = pending_positions[idx]
        display[pos]["_display_status"] = "completed"

    if action == "Remove" and isinstance(target_idx, int) and 0 <= target_idx < len(pending_positions):
        pos = pending_positions[target_idx]
        if display[pos].get("_display_status") != "completed":
            display[pos]["_display_status"] = "removed"
    elif action == "Replace" and isinstance(target_idx, int) and 0 <= target_idx < len(pending_positions):
        pos = pending_positions[target_idx]
        if display[pos].get("_display_status") != "completed":
            num_completed_before = sum(1 for idx in completed_set if idx < target_idx)
            updated_idx = target_idx - num_completed_before
            if 0 <= updated_idx < len(updated_queue):
                display[pos] = _with_display_status(updated_queue[updated_idx], "replaced")
    elif action == "Append":
        old_remaining_len = len(original_queue) - len(completed_set)
        append_count = len(updated_queue) - old_remaining_len
        if append_count > 0:
            appended = updated_queue[-append_count:]
            display.extend(_with_display_status(elem, "pending") for elem in appended)

    return display


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════


def _prepare_segment_for_vlm(
    video_bytes: bytes,
    audio_path: str | None,
    audio_mode: str,
    clip_start_sec: float,
    clip_end_sec: float,
    fps: float,
) -> tuple[bytes, str, dict]:
    """Prepare a video segment for VLM based on audio_mode.

    Returns (vlm_video_bytes, transcript_section_text, transcript_data).
    transcript_data contains {"text": ..., "words": [...]} or empty.
    """
    transcript_data = {"text": "", "words": []}
    transcript_sections: list[str] = []

    if audio_mode_has_whisper(audio_mode) and audio_path:
        # Option 1: Whisper → captions overlay on the video only
        print(f"  Whisper transcribing {clip_start_sec:.1f}s-{clip_end_sec:.1f}s ...")
        transcript_data = whisper_transcribe_segment(
            audio_path, clip_start_sec, clip_end_sec
        )
        if transcript_data["text"]:
            print(f"  Transcript: \"{transcript_data['text'][:100]}\"")
            # Burn captions on video
            video_bytes = burn_captions_on_video(
                video_bytes,
                transcript_data["words"],
                clip_start_sec,
                fps,
            )
        else:
            print(f"  Transcript: (no speech detected)")

    if audio_mode_has_native_audio(audio_mode) and audio_path:
        # Option 2: merge audio into video → Gemini processes audio natively
        print(
            f"  Merging audio track for {clip_start_sec:.1f}s-{clip_end_sec:.1f}s ..."
        )
        audio_seg_path = cut_audio_segment(audio_path, clip_start_sec, clip_end_sec)
        try:
            video_bytes = merge_video_audio(video_bytes, audio_seg_path)
        finally:
            os.unlink(audio_seg_path)
        transcript_sections.append(
            "\nThis video clip contains an audio track with the human's spoken "
            "instruction. Listen carefully to understand what the human is saying.\n"
        )

    transcript_section = "".join(transcript_sections)
    return video_bytes, transcript_section, transcript_data


def save_step_debug_logs(
    *,
    debug_dir: str,
    step: int,
    phase: str,
    prompt_text: str,
    raw_response: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    """Persist prompt/response/debug payload for one VLM step."""
    os.makedirs(debug_dir, exist_ok=True)
    stem = f"step_{step:03d}_{phase}"

    prompt_path = os.path.join(debug_dir, f"{stem}_prompt.txt")
    response_path = os.path.join(debug_dir, f"{stem}_response.txt")
    payload_path = os.path.join(debug_dir, f"{stem}_payload.json")

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(raw_response)
    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "prompt_path": prompt_path,
        "response_path": response_path,
        "payload_path": payload_path,
    }


def run_simulation(
    *,
    episode_name: str,
    video_path: str,
    audio_path: str | None,
    save_audio_path: str | None,
    audio_mode: str,
    model: str,
    client: genai.Client,
    output_dir: str,
    task_vocabulary_section: str,
    video_fps_hint: int = 2,
    target_resolution: tuple[int, int] | None = None,
    simulation_timing: str = SIMULATION_TIMING_FIXED,
    gaze_points: np.ndarray | None = None,
    gaze_crop_zoom: float = 1.0,
    verbose: bool = False,
) -> tuple[list[dict], KeyframeQueue]:
    """Run the full asynchronous simulation for one episode.

    Returns (log_entries, final_queue).
    """
    total_frames, fps = get_video_meta(video_path)
    video_duration = total_frames / fps
    seg_dir = os.path.join(output_dir, "segmented_videos")
    debug_dir = os.path.join(output_dir, "debug_logs")
    os.makedirs(seg_dir, exist_ok=True)

    queue = KeyframeQueue()
    display_queue_state: list[dict] | None = None
    log_entries: list[dict] = []

    print(f"\n{'='*70}")
    print(f"Episode: {episode_name}")
    print(f"Video: {video_duration:.1f}s  ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Audio mode: {audio_mode}")
    print(f"Simulation timing: {simulation_timing}")
    if gaze_points is not None and gaze_crop_zoom > 1.0:
        print(f"Gaze crop zoom: {gaze_crop_zoom:.2f}x")
    print(f"{'='*70}")

    # ── Phase 1: Initialization ──────────────────────────────────────────
    print(f"\n[Phase 1] Initializing queue from first {INIT_SEGMENT_SEC:.0f}s ...")
    init_end = min(INIT_SEGMENT_SEC, video_duration)
    seg_path = os.path.join(seg_dir, f"segment_0_{0.0:.1f}s_{init_end:.1f}s.mp4")
    init_bytes = cut_video_segment(
        video_path,
        0,
        init_end,
        output_path=seg_path,
        target_resolution=target_resolution,
        target_fps=video_fps_hint,
        gaze_points=gaze_points,
        gaze_crop_zoom=gaze_crop_zoom,
    )

    # Merge audio into saved segment file
    if save_audio_path:
        audio_seg = cut_audio_segment(save_audio_path, 0, init_end)
        _ffmpeg_merge_audio(seg_path, audio_seg, seg_path)
        os.unlink(audio_seg)

    # The clip was subsampled to video_fps_hint fps
    vlm_clip_fps = min(video_fps_hint, fps)
    video_view_section = ""
    if gaze_points is not None and gaze_crop_zoom > 1.0:
        video_view_section = (
            f"\nThis clip is a gaze-centered zoomed crop. Each frame is cropped around the wearer's gaze point "
            f"and enlarged by approximately {gaze_crop_zoom:.1f}x. No extra marker or overlay has been added. "
            "Use this zoomed RGB view for fine-grained object identification.\n"
        )

    # Prepare audio (whisper or voiced-video)
    vlm_bytes, transcript_section, transcript_data = _prepare_segment_for_vlm(
        init_bytes,
        audio_path,
        audio_mode,
        0,
        init_end,
        vlm_clip_fps,
    )

    # Any mode with Whisper: overwrite saved segment with the final VLM clip.
    # In whisper+voiced mode, vlm_bytes already includes the merged audio track.
    if audio_mode_has_whisper(audio_mode) and transcript_data.get("text"):
        with open(seg_path, "wb") as _f:
            _f.write(vlm_bytes)
        # Whisper-only mode needs the saved segment's audio track restored.
        if save_audio_path and not audio_mode_has_native_audio(audio_mode):
            audio_seg = cut_audio_segment(save_audio_path, 0, init_end)
            _ffmpeg_merge_audio(seg_path, audio_seg, seg_path)
            os.unlink(audio_seg)

    prompt = PROMPT_INIT.format(
        clip_duration=init_end,
        fps=vlm_clip_fps,
        task_vocabulary_section=task_vocabulary_section,
        transcript_section=transcript_section,
        video_view_section=video_view_section,
    )

    print(
        f"  VLM input: {len(vlm_bytes)/1024:.1f} KB  |  "
        f"audio_in_video={audio_mode_has_native_audio(audio_mode)}  |  "
        f"clip_fps={vlm_clip_fps}"
    )
    if verbose:
        print(f"  Prompt:\n{prompt[:400]}...")

    response, raw_text, latency = call_gemini_with_timing(
        client,
        model,
        vlm_bytes,
        prompt,
        video_fps=vlm_clip_fps,
        has_audio=audio_mode_has_native_audio(audio_mode),
    )
    cost = extract_cost(response, model)

    parsed = parse_init_response(raw_text)
    if parsed is None:
        debug_paths = save_step_debug_logs(
            debug_dir=debug_dir,
            step=0,
            phase="init",
            prompt_text=prompt,
            raw_response=raw_text,
            payload={
                "episode_name": episode_name,
                "phase": "init",
                "step": 0,
                "simulation_timing": simulation_timing,
                "model": model,
                "audio_mode": audio_mode,
                "input_segment_path": seg_path,
                "input_video_path": video_path,
                "input_video_start_sec": 0.0,
                "input_video_end_sec": round(init_end, 3),
                "queue_before_prompt": "(queue is empty)",
                "queue_before_snapshot": [],
                "queue_after_snapshot": [],
                "parsed_action": "InitFailure",
                "parsed_action_details": None,
                "response_latency": latency,
                "cost": cost,
                "transcript_text": transcript_data.get("text", ""),
                "transcript_section": transcript_section,
                "input_size_bytes": len(vlm_bytes),
                "gaze_crop_zoom": gaze_crop_zoom,
            },
        )
        print(f"  [ERROR] Failed to parse init response. Raw:\n{raw_text[:500]}")
        log_entries.append(
            {
                "step": 0,
                "global_timestamp": round(
                    init_end if simulation_timing == SIMULATION_TIMING_FIXED else init_end + latency,
                    3,
                ),
                "model_response": raw_text,
                "parsed_action": "InitFailure",
                "parsed_action_details": None,
                "original_keyframe_queue": [],
                "updated_keyframe_queue": [],
                "response_latency": latency,
                "input_video_start_sec": 0.0,
                "input_video_end_sec": round(init_end, 3),
                "input_segment_path": seg_path,
                "prompt_text": prompt,
                "cost": cost,
                "debug_prompt_path": debug_paths["prompt_path"],
                "debug_response_path": debug_paths["response_path"],
                "debug_payload_path": debug_paths["payload_path"],
            }
        )
        return log_entries, queue

    # Build queue elements
    kf_dir = os.path.join(output_dir, "keyframes")
    elements = build_queue_elements(
        parsed["action_queue"],
        init_bytes,
        kf_dir,
        prefix="init",
    )
    queue_before = queue.snapshot()
    queue.initialize(elements)
    display_queue_state = build_display_queue_snapshot(
        None,
        queue_before,
        queue.snapshot(),
        "Initialize",
        None,
    )
    debug_paths = save_step_debug_logs(
        debug_dir=debug_dir,
        step=0,
        phase="init",
        prompt_text=prompt,
        raw_response=raw_text,
        payload={
            "episode_name": episode_name,
            "phase": "init",
            "step": 0,
            "simulation_timing": simulation_timing,
            "model": model,
            "audio_mode": audio_mode,
            "input_segment_path": seg_path,
            "input_video_path": video_path,
            "input_video_start_sec": 0.0,
            "input_video_end_sec": round(init_end, 3),
            "queue_before_prompt": "(queue is empty)",
            "queue_before_snapshot": queue_before,
            "queue_after_snapshot": queue.snapshot(),
            "display_queue_snapshot": display_queue_state,
            "parsed_action": "Initialize",
            "parsed_action_details": {
                "human_instruction_summary": parsed.get(
                    "human_instruction_summary", ""
                ),
                "reasoning": parsed.get("reasoning", ""),
                "num_elements": len(elements),
                "raw_keyframe_queue": parsed.get("keyframe_queue", []),
            },
            "response_latency": latency,
            "cost": cost,
            "transcript_text": transcript_data.get("text", ""),
            "transcript_section": transcript_section,
            "input_size_bytes": len(vlm_bytes),
            "gaze_crop_zoom": gaze_crop_zoom,
        },
    )

    # Global time starts at init_end exactly — robot is idle during init,
    # so VLM latency does NOT advance the global timer for the init call.
    global_time = init_end

    log_entries.append(
        {
            "step": 0,
            "global_timestamp": round(init_end, 3),
            "model_response": raw_text,
            "parsed_action": "Initialize",
            "parsed_action_details": {
                "human_instruction_summary": parsed.get(
                    "human_instruction_summary", ""
                ),
                "reasoning": parsed.get("reasoning", ""),
                "num_elements": len(elements),
                "raw_keyframe_queue": parsed.get("keyframe_queue", []),
            },
            "original_keyframe_queue": queue_before,
            "updated_keyframe_queue": queue.snapshot(),
            "display_keyframe_queue": display_queue_state,
            "response_latency": latency,
            "input_video_start_sec": 0.0,
            "input_video_end_sec": round(init_end, 3),
            "input_segment_path": seg_path,
            "prompt_text": prompt,
            "cost": cost,
            "debug_prompt_path": debug_paths["prompt_path"],
            "debug_response_path": debug_paths["response_path"],
            "debug_payload_path": debug_paths["payload_path"],
        }
    )
    print(f"  Latency: {latency:.1f}s  |  Cost: ${cost['cost_usd']:.4f}")
    print(f"  Summary: {parsed.get('human_instruction_summary', 'N/A')}")
    print(f"  Queue initialized with {len(elements)} elements:")
    for i, e in enumerate(elements):
        print(f"    [{i}] {e.subtask_instruction}  |  {e.keyframe_description}")
    print(f"  Global time: {global_time:.1f}s / {video_duration:.1f}s")

    # ── Phase 2: Periodic Updates ────────────────────────────────────────
    def _run_update_step(step: int, seg_start: float, seg_end: float) -> tuple[float, float]:
        nonlocal display_queue_state

        seg_name = f"segment_{step}_{seg_start:.1f}s_{seg_end:.1f}s.mp4"
        seg_path = os.path.join(seg_dir, seg_name)

        print(
            f"\n[Step {step}] Update call @ scheduled_time={seg_end:.1f}s  "
            f"(clip: {seg_start:.1f}s - {seg_end:.1f}s)"
        )

        seg_bytes = cut_video_segment(
            video_path,
            seg_start,
            seg_end,
            output_path=seg_path,
            target_resolution=target_resolution,
            target_fps=video_fps_hint,
            gaze_points=gaze_points,
            gaze_crop_zoom=gaze_crop_zoom,
        )

        if audio_path and os.path.exists(audio_path):
            audio_seg = cut_audio_segment(save_audio_path, seg_start, seg_end)
            _ffmpeg_merge_audio(seg_path, audio_seg, seg_path)
            os.unlink(audio_seg)

        vlm_bytes, transcript_section, transcript_data = _prepare_segment_for_vlm(
            seg_bytes,
            audio_path,
            audio_mode,
            seg_start,
            seg_end,
            vlm_clip_fps,
        )

        if audio_mode_has_whisper(audio_mode) and transcript_data.get("text"):
            with open(seg_path, "wb") as _f:
                _f.write(vlm_bytes)
            if save_audio_path and not audio_mode_has_native_audio(audio_mode):
                audio_seg = cut_audio_segment(save_audio_path, seg_start, seg_end)
                _ffmpeg_merge_audio(seg_path, audio_seg, seg_path)
                os.unlink(audio_seg)

        prompt = PROMPT_UPDATE.format(
            clip_duration=seg_end - seg_start,
            clip_start_sec=seg_start,
            clip_end_sec=seg_end,
            fps=vlm_clip_fps,
            queue_status=queue.format_for_prompt(),
            task_vocabulary_section=task_vocabulary_section,
            transcript_section=transcript_section,
            video_view_section=video_view_section,
        )

        queue_before_prompt = queue.format_for_prompt()
        queue_before = queue.snapshot()

        print(
            f"  VLM input: {len(vlm_bytes)/1024:.1f} KB  |  "
            f"audio_in_video={audio_mode_has_native_audio(audio_mode)}  |  "
            f"clip_fps={vlm_clip_fps}"
        )

        response, raw_text, latency = call_gemini_with_timing(
            client,
            model,
            vlm_bytes,
            prompt,
            video_fps=vlm_clip_fps,
            has_audio=audio_mode_has_native_audio(audio_mode),
        )
        cost = extract_cost(response, model)

        result_visible_time = (
            min(seg_end + latency, video_duration)
            if simulation_timing == SIMULATION_TIMING_REAL
            else seg_end
        )

        parsed = parse_update_response(raw_text)
        if parsed is not None:
            completed_indices = []
            for idx in parsed.get("completed_queue_indices", []):
                if isinstance(idx, int) and 0 <= idx < len(queue_before):
                    completed_indices.append(idx)
            completed_elements = [queue_before[idx] for idx in completed_indices]

            action = apply_queue_update(
                queue,
                parsed,
                clip_bytes=seg_bytes,
                keyframe_dir=kf_dir,
                step=step,
                clip_start_sec=seg_start,
            )
            action_details = {
                "reasoning": parsed.get("reasoning", ""),
                "clip_summary": parsed.get("clip_summary", ""),
                "observed_events": parsed.get("observed_events", []),
                "completion_evidence": parsed.get("completion_evidence", []),
                "completed_queue_indices": completed_indices,
                "completed_elements": completed_elements,
                "action": action,
                "target_idx": parsed.get("target_idx"),
                "new_elements": parsed.get("new_elements", []),
            }
            display_queue_state = build_display_queue_snapshot(
                display_queue_state,
                queue_before,
                queue.snapshot(),
                action,
                action_details,
            )
        else:
            action = "ParseFailure"
            action_details = None
            print(f"  [ERROR] Failed to parse update response. Raw:\n{raw_text[:300]}")

        debug_paths = save_step_debug_logs(
            debug_dir=debug_dir,
            step=step,
            phase="update",
            prompt_text=prompt,
            raw_response=raw_text,
            payload={
                "episode_name": episode_name,
                "phase": "update",
                "step": step,
                "simulation_timing": simulation_timing,
                "model": model,
                "audio_mode": audio_mode,
                "input_segment_path": seg_path,
                "input_video_path": video_path,
                "input_video_start_sec": round(seg_start, 3),
                "input_video_end_sec": round(seg_end, 3),
                "result_visible_time": round(result_visible_time, 3),
                "queue_before_prompt": queue_before_prompt,
                "queue_before_snapshot": queue_before,
                "queue_after_snapshot": queue.snapshot(),
                "display_queue_snapshot": display_queue_state,
                "parsed_action": action,
                "parsed_action_details": action_details,
                "response_latency": latency,
                "cost": cost,
                "transcript_text": transcript_data.get("text", ""),
                "transcript_section": transcript_section,
                "input_size_bytes": len(vlm_bytes),
                "gaze_crop_zoom": gaze_crop_zoom,
            },
        )

        log_entries.append(
            {
                "step": step,
                "global_timestamp": round(result_visible_time, 3),
                "model_response": raw_text,
                "parsed_action": action,
                "parsed_action_details": action_details,
                "original_keyframe_queue": queue_before,
                "updated_keyframe_queue": queue.snapshot(),
                "display_keyframe_queue": display_queue_state,
                "response_latency": latency,
                "input_video_start_sec": round(seg_start, 3),
                "input_video_end_sec": round(seg_end, 3),
                "input_segment_path": seg_path,
                "prompt_text": prompt,
                "cost": cost,
                "debug_prompt_path": debug_paths["prompt_path"],
                "debug_response_path": debug_paths["response_path"],
                "debug_payload_path": debug_paths["payload_path"],
            }
        )

        print(f"  Latency: {latency:.1f}s  |  Cost: ${cost['cost_usd']:.4f}")
        print(f"  Action: {action}")
        print(f"  Result visible at simulation t={result_visible_time:.1f}s")
        if action_details and action_details.get("clip_summary"):
            print(f"  Clip summary: {action_details['clip_summary']}")
        if action_details and action_details.get("completed_queue_indices"):
            print(
                "  Completed queue indices: "
                + ", ".join(str(i) for i in action_details["completed_queue_indices"])
            )
        if action_details and action_details.get("completion_evidence"):
            print("  Completion evidence:")
            for item in action_details["completion_evidence"][:5]:
                print(f"    - [{item.get('queue_idx')}] {item.get('evidence', '')}")
        if action_details and action_details.get("observed_events"):
            print("  Observed events:")
            for event in action_details["observed_events"][:5]:
                print(f"    - {event}")
        if parsed and action != "NoOp":
            print(f"  Reasoning: {parsed.get('reasoning', '')[:150]}")
        print(f"  Queue ({len(queue)} elements):")
        for i in range(len(queue)):
            print(
                f"    [{i}] {queue[i].subtask_instruction}  |  {queue[i].keyframe_description}"
            )

        return latency, result_visible_time

    step = 1
    if simulation_timing == SIMULATION_TIMING_FIXED:
        while True:
            seg_end = min(INIT_SEGMENT_SEC + step * SEGMENT_DURATION_SEC, video_duration)
            if seg_end <= global_time + 1e-6:
                break
            seg_start = max(0, seg_end - SEGMENT_DURATION_SEC)
            _latency, _visible_time = _run_update_step(step, seg_start, seg_end)
            global_time = seg_end
            print(f"  Global time: {global_time:.1f}s / {video_duration:.1f}s")
            if seg_end >= video_duration:
                break
            step += 1
    else:
        local_timer = 0.0  # fresh start; init latency was not counted
        while global_time < video_duration:
            wait = SEGMENT_DURATION_SEC - local_timer
            if wait < 0:
                wait = 0
            global_time += wait
            global_time = min(global_time, video_duration)

            if global_time >= video_duration:
                break

            seg_end = global_time
            seg_start = max(0, seg_end - SEGMENT_DURATION_SEC)
            latency, visible_time = _run_update_step(step, seg_start, seg_end)
            global_time = min(visible_time, video_duration)
            local_timer = latency
            print(f"  Global time: {global_time:.1f}s / {video_duration:.1f}s")
            step += 1

    print(f"\n{'─'*70}")
    print(f"Simulation complete. {step} VLM calls total.")
    total_cost = sum(e["cost"]["cost_usd"] for e in log_entries)
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Final queue ({len(queue)} elements):")
    for i in range(len(queue)):
        print(
            f"  [{i}] {queue[i].subtask_instruction}  |  {queue[i].keyframe_description}"
        )

    return log_entries, queue


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION — offline_simulation.mp4
# ══════════════════════════════════════════════════════════════════════════════

# Layout constants — large canvas with video in top-right corner
VIS_CANVAS_W = 4096
VIS_CANVAS_H = 3250
VIS_VIDEO_SIZE = 512  # original video thumbnail (top-right)
QUEUE_THUMB_SIZE = (512, 512)  # keyframe thumbnail size
QUEUE_ROW_HEIGHT = 620  # px per queue row (512 thumb + padding)
QUEUE_ROWS_PER_COL = 5  # 5 rows, 2 columns
QUEUE_NUM_COLS = 2
QUEUE_COL_WIDTH = 1950  # px per column (thumb + text)
QUEUE_LEFT_MARGIN = 40
QUEUE_TOP_START = 60  # where queue rows begin (below title)


def lookup_queue_at_time(
    t: float,
    log_entries: list[dict],
) -> list[dict] | None:
    """Return queue state at simulated time t.

    None means queue not yet initialized.
    """
    result = None
    for entry in log_entries:
        if entry["global_timestamp"] <= t:
            result = entry.get("display_keyframe_queue", entry["updated_keyframe_queue"])
        else:
            break
    return result


def lookup_latest_entry_at_time(t: float, log_entries: list[dict]) -> dict | None:
    """Return the latest applied inference entry at simulated time t."""
    latest = None
    for entry in log_entries:
        if entry["global_timestamp"] <= t:
            latest = entry
        else:
            break
    return latest


def lookup_next_entry_after_time(t: float, log_entries: list[dict]) -> dict | None:
    """Return the next inference entry that becomes visible after time t."""
    for entry in log_entries:
        if entry["global_timestamp"] > t:
            return entry
    return None


def _action_color(action: str) -> tuple[int, int, int]:
    """Return a stable BGR color for an inference action label."""
    if action == "NoOp":
        return (90, 200, 255)
    if action in {"Append", "Initialize"}:
        return (100, 255, 100)
    if action == "Replace":
        return (255, 210, 90)
    if action == "Remove":
        return (110, 160, 255)
    if action == "ParseFailure":
        return (90, 90, 255)
    if action == "InitFailure":
        return (90, 90, 255)
    return (220, 220, 220)


def _render_canvas(
    video_frame: np.ndarray | None,
    queue_elements: list[dict] | None,
    current_time: float,
    frame_idx: int,
    total_frames: int,
    latest_entry: dict | None,
    next_entry: dict | None,
    image_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Render one frame of the offline simulation.

    Layout: 4096 x 3250 canvas.
      - Top-right corner: 512x512 original video frame
      - Rest of canvas: keyframe queue (top-to-bottom, left-aligned)
      - Each queue element: 512x512 thumb + text to the right
    """
    W, H = VIS_CANVAS_W, VIS_CANVAS_H
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)  # dark background

    font = cv2.FONT_HERSHEY_SIMPLEX
    vs = VIS_VIDEO_SIZE

    # ── Original video (top-right) ──
    vx = W - vs - 30
    vy = 30
    if video_frame is not None:
        thumb = cv2.resize(video_frame, (vs, vs), interpolation=cv2.INTER_LANCZOS4)
        canvas[vy : vy + vs, vx : vx + vs] = thumb
        cv2.rectangle(
            canvas, (vx - 2, vy - 2), (vx + vs + 1, vy + vs + 1), (180, 180, 180), 2
        )
    # Time label under video
    cv2.putText(
        canvas,
        f"t = {current_time:.1f}s",
        (vx, vy + vs + 35),
        font,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"frame = {frame_idx + 1}/{total_frames}",
        (vx, vy + vs + 72),
        font,
        0.9,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    # ── Inference status panel (under video) ──
    panel_x = vx - 10
    panel_y = vy + vs + 100
    panel_w = min(900, W - panel_x - 30)
    panel_h = 300
    cv2.rectangle(
        canvas,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (36, 36, 36),
        -1,
    )
    cv2.rectangle(
        canvas,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (95, 95, 95),
        2,
    )
    cv2.putText(
        canvas,
        "Inference Status",
        (panel_x + 18, panel_y + 36),
        font,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if latest_entry is not None:
        action = latest_entry.get("parsed_action", "(unknown)")
        clip_start = float(latest_entry.get("input_video_start_sec", 0.0))
        clip_end = float(latest_entry.get("input_video_end_sec", 0.0))
        visible_at = float(latest_entry.get("global_timestamp", 0.0))
        latency = float(latest_entry.get("response_latency", 0.0))
        reason = ""
        clip_summary = ""
        completed_txt = ""
        details = latest_entry.get("parsed_action_details")
        if isinstance(details, dict):
            reason = str(details.get("reasoning", "")).strip()
            clip_summary = str(details.get("clip_summary", "")).strip()
            completed = details.get("completed_queue_indices", [])
            if isinstance(completed, list) and completed:
                completed_txt = ", ".join(str(x) for x in completed[:6])

        cv2.putText(
            canvas,
            f"latest result: step {latest_entry.get('step', '?')}  {action}",
            (panel_x + 18, panel_y + 78),
            font,
            0.95,
            _action_color(action),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"inference clip: {clip_start:.1f}s - {clip_end:.1f}s",
            (panel_x + 18, panel_y + 118),
            font,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"result became active at t={visible_at:.1f}s  |  latency={latency:.1f}s",
            (panel_x + 18, panel_y + 154),
            font,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        if completed_txt:
            cv2.putText(
                canvas,
                f"completed queue idx: {completed_txt}",
                (panel_x + 18, panel_y + 188),
                font,
                0.72,
                (200, 220, 200),
                2,
                cv2.LINE_AA,
            )
        summary_text = clip_summary or reason
        if summary_text:
            summary_text = "summary: " + " ".join(summary_text.split())
            max_chars = 78
            summary_lines = [
                summary_text[i : i + max_chars]
                for i in range(0, len(summary_text), max_chars)
            ]
            start_y = panel_y + (216 if completed_txt else 192)
            for li, line in enumerate(summary_lines[:2]):
                cv2.putText(
                    canvas,
                    line,
                    (panel_x + 18, start_y + li * 28),
                    font,
                    0.65,
                    (170, 170, 170),
                    1,
                    cv2.LINE_AA,
                )
    elif next_entry is not None:
        clip_start = float(next_entry.get("input_video_start_sec", 0.0))
        clip_end = float(next_entry.get("input_video_end_sec", 0.0))
        visible_at = float(next_entry.get("global_timestamp", 0.0))
        cv2.putText(
            canvas,
            "latest result: pending",
            (panel_x + 18, panel_y + 78),
            font,
            0.95,
            (120, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"pending clip: {clip_start:.1f}s - {clip_end:.1f}s",
            (panel_x + 18, panel_y + 118),
            font,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"will become active at simulated t={visible_at:.1f}s",
            (panel_x + 18, panel_y + 154),
            font,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "latest result: none",
            (panel_x + 18, panel_y + 78),
            font,
            0.95,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )

    # ── Queue section (left side) ──
    thumb_w, thumb_h = QUEUE_THUMB_SIZE

    # Title
    cv2.putText(
        canvas,
        "Keyframe Action Queue",
        (QUEUE_LEFT_MARGIN, 45),
        font,
        1.5,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    if queue_elements is None:
        cv2.putText(
            canvas,
            "Initializing...",
            (QUEUE_LEFT_MARGIN, 150),
            font,
            1.2,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        return canvas

    if len(queue_elements) == 0:
        cv2.putText(
            canvas,
            "(queue is empty)",
            (QUEUE_LEFT_MARGIN, 150),
            font,
            1.0,
            (120, 120, 120),
            2,
            cv2.LINE_AA,
        )
        return canvas

    # Draw queue elements in a 5×2 grid, filled row-major:
    # row 0 -> [0, 1], row 1 -> [2, 3], ...
    for i, elem in enumerate(queue_elements):
        col = i % QUEUE_NUM_COLS
        row = i // QUEUE_NUM_COLS
        display_status = str(elem.get("_display_status", "pending"))
        is_completed = display_status == "completed"
        is_removed = display_status == "removed"
        is_replaced = display_status == "replaced"

        # Thumbnail — load from saved PNG
        img_path = elem.get("keyframe_image_path", "")
        if img_path and img_path not in image_cache:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(
                    img, (thumb_w, thumb_h), interpolation=cv2.INTER_LANCZOS4
                )
            else:
                img = np.zeros((thumb_h, thumb_w, 3), np.uint8)
            image_cache[img_path] = img
        thumb = image_cache.get(img_path, np.zeros((thumb_h, thumb_w, 3), np.uint8))

        tx = QUEUE_LEFT_MARGIN + col * QUEUE_COL_WIDTH
        ty = QUEUE_TOP_START + row * QUEUE_ROW_HEIGHT + 8
        # Clamp to canvas bounds
        if ty + thumb_h > H or tx + thumb_w > W:
            continue
        thumb_to_draw = thumb.copy()
        if is_completed:
            thumb_to_draw = cv2.addWeighted(
                thumb_to_draw,
                0.45,
                np.full_like(thumb_to_draw, 32),
                0.55,
                0,
            )
        elif is_removed:
            red_tint = np.full_like(thumb_to_draw, (20, 20, 90))
            thumb_to_draw = cv2.addWeighted(thumb_to_draw, 0.45, red_tint, 0.55, 0)
        elif is_replaced:
            orange_tint = np.full_like(thumb_to_draw, (40, 120, 200))
            thumb_to_draw = cv2.addWeighted(thumb_to_draw, 0.5, orange_tint, 0.5, 0)
        canvas[ty : ty + thumb_h, tx : tx + thumb_w] = thumb_to_draw
        cv2.rectangle(
            canvas,
            (tx - 2, ty - 2),
            (tx + thumb_w + 1, ty + thumb_h + 1),
            (80, 180, 80)
            if is_completed
            else ((90, 90, 220) if is_removed else ((90, 180, 255) if is_replaced else (100, 100, 100))),
            2,
        )

        # Text right of thumbnail
        text_x = tx + thumb_w + 25

        # Index + instruction (large, color-coded)
        instr = elem.get("subtask_instruction", "")
        if is_completed:
            color = (140, 220, 140)
        elif is_removed:
            color = (120, 120, 255)
        elif is_replaced:
            color = (120, 200, 255)
        else:
            color = (100, 255, 100) if instr.startswith("pick up") else (100, 200, 255)

        box_x = text_x
        box_y = ty + 12
        box_size = 22
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_size, box_y + box_size),
            (220, 220, 220) if (is_removed or is_replaced) else (180, 180, 180),
            2,
        )
        if is_completed:
            cv2.line(
                canvas,
                (box_x + 4, box_y + 12),
                (box_x + 10, box_y + 18),
                (80, 220, 80),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (box_x + 10, box_y + 18),
                (box_x + 18, box_y + 4),
                (80, 220, 80),
                3,
                cv2.LINE_AA,
            )
        elif is_removed:
            cv2.line(
                canvas,
                (box_x + 4, box_y + 4),
                (box_x + box_size - 4, box_y + box_size - 4),
                (90, 90, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (box_x + box_size - 4, box_y + 4),
                (box_x + 4, box_y + box_size - 4),
                (90, 90, 255),
                3,
                cv2.LINE_AA,
            )
        elif is_replaced:
            cv2.line(
                canvas,
                (box_x + 4, box_y + box_size // 2),
                (box_x + box_size - 4, box_y + box_size // 2),
                (80, 180, 255),
                3,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"[{i}] {instr}",
            (text_x + 34, ty + 40),
            font,
            1.1,
            color,
            2,
            cv2.LINE_AA,
        )
        if is_completed:
            cv2.putText(
                canvas,
                "DONE",
                (text_x + 34, ty + 76),
                font,
                0.72,
                (120, 220, 120),
                2,
                cv2.LINE_AA,
            )
        elif is_removed:
            cv2.putText(
                canvas,
                "REMOVED",
                (text_x + 34, ty + 76),
                font,
                0.72,
                (120, 120, 255),
                2,
                cv2.LINE_AA,
            )
        elif is_replaced:
            cv2.putText(
                canvas,
                "REPLACED",
                (text_x + 34, ty + 76),
                font,
                0.72,
                (120, 200, 255),
                2,
                cv2.LINE_AA,
            )

        # Interval
        s_sec = elem.get("start_sec", 0.0)
        e_sec = elem.get("end_sec", 0.0)
        peak_sec = elem.get("pointing_peak_sec")
        peak_txt = f"  |  peak: {peak_sec:.1f}s" if peak_sec is not None else ""
        cv2.putText(
            canvas,
            f"interval: {s_sec:.1f}s - {e_sec:.1f}s{peak_txt}",
            (text_x, ty + (112 if (is_completed or is_removed or is_replaced) else 80)),
            font,
            0.8,
            (170, 185, 195)
            if is_replaced
            else ((160, 160, 160) if (is_completed or is_removed) else (200, 200, 200)),
            1,
            cv2.LINE_AA,
        )

        # Description (multi-line, wrapped)
        desc = elem.get("keyframe_description", "")
        max_chars_per_col = (
            60 if col == 1 else 90
        )  # narrower for col 1 (video in top-right)
        desc_lines = [
            desc[j : j + max_chars_per_col]
            for j in range(0, len(desc), max_chars_per_col)
        ]
        for li, line in enumerate(desc_lines[:5]):
            cv2.putText(
                canvas,
                line,
                (text_x, ty + (152 if (is_completed or is_removed or is_replaced) else 120) + li * 30),
                font,
                0.7,
                (135, 170, 190)
                if is_replaced
                else ((120, 140, 120) if is_completed else ((135, 120, 150) if is_removed else (150, 150, 150))),
                1,
                cv2.LINE_AA,
            )

    return canvas


def _ffmpeg_merge_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Merge audio into a video file using ffmpeg. Overwrites output_path.

    Downmixes to mono to ensure compatibility (source may be multi-channel).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-ac",
            "1",  # downmix to mono
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            tmp.name,
        ],
        check=True,
    )
    os.replace(tmp.name, output_path)


def render_offline_simulation(
    video_path: str,
    log_entries: list[dict],
    output_path: str,
    audio_path: str | None = None,
) -> None:
    """Render the offline simulation video with queue overlay.

    Canvas: 4096 x 3250.  Original video 512x512 in top-right.
    Keyframe queue fills left portion.
    """
    if not log_entries:
        print("  No log entries, skipping visualization.")
        return

    total_frames, fps = get_video_meta(video_path)

    W, H = VIS_CANVAS_W, VIS_CANVAS_H
    print(
        f"\nRendering offline_simulation.mp4  ({W}x{H} @ {fps:.0f}fps, {total_frames} frames) ..."
    )

    # Temp output
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (W, H))

    image_cache: dict[str, np.ndarray] = {}

    cap = cv2.VideoCapture(video_path)
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        queue_state = lookup_queue_at_time(t, log_entries)
        latest_entry = lookup_latest_entry_at_time(t, log_entries)
        next_entry = lookup_next_entry_after_time(t, log_entries)

        canvas = _render_canvas(
            video_frame=frame,
            queue_elements=queue_state,
            current_time=t,
            frame_idx=frame_idx,
            total_frames=total_frames,
            latest_entry=latest_entry,
            next_entry=next_entry,
            image_cache=image_cache,
        )
        writer.write(canvas)

        if frame_idx % (int(fps) * 5) == 0:
            print(f"  Frame {frame_idx}/{total_frames}  (t={t:.1f}s)")

    writer.release()
    cap.release()

    # Re-encode to H264
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _ffmpeg_reencode_h264(tmp.name, output_path)
    os.unlink(tmp.name)

    # Merge audio if available
    if audio_path and os.path.exists(audio_path):
        print(f"  Merging audio track into offline_simulation.mp4 ...")
        _ffmpeg_merge_audio(output_path, audio_path, output_path)

    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def print_simulation_summary(
    log_entries: list[dict],
    queue: KeyframeQueue,
    episode_name: str,
) -> None:
    """Print a concise summary of the simulation."""
    print(f"\n{'='*70}")
    print(f"SIMULATION SUMMARY — {episode_name}")
    print(f"{'='*70}")

    total_cost = sum(e["cost"]["cost_usd"] for e in log_entries)
    total_latency = sum(e["response_latency"] for e in log_entries)
    actions = [e["parsed_action"] for e in log_entries]

    print(f"  VLM calls: {len(log_entries)}")
    print(f"  Total latency: {total_latency:.1f}s")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"  Actions: {', '.join(actions)}")
    print(f"  Final queue size: {len(queue)}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / MAIN
# ══════════════════════════════════════════════════════════════════════════════


def parse_resolution(s: str) -> tuple[int, int] | None:
    """Parse resolution string like '512' or '512x384'."""
    if not s:
        return None
    if "x" in s:
        parts = s.split("x")
        return (int(parts[0]), int(parts[1]))
    v = int(s)
    return (v, v)


def process_episode(
    *,
    episode_name: str,
    dataset_dir: str,
    output_root_dir: str,
    use_gaze_video: bool,
    audio_mode: str,
    model: str,
    api_key: str,
    task_vocabulary_section: str,
    video_fps_hint: int,
    target_resolution: tuple[int, int] | None,
    simulation_timing: str,
    gaze_crop_zoom: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run one episode end-to-end and return a compact status dict."""
    try:
        files = resolve_episode_files(dataset_dir, episode_name)
        video_path = files["rgb_with_gaze"] if use_gaze_video else files["rgb"]
        if not os.path.exists(video_path):
            return {
                "episode": episode_name,
                "ok": False,
                "error": f"Video not found: {video_path}",
            }

        ep_output_dir = os.path.join(output_root_dir, episode_name)
        os.makedirs(ep_output_dir, exist_ok=True)

        episode_audio_mode = audio_mode
        vlm_audio_path = files["audio"] if episode_audio_mode != AUDIO_MODE_NONE else None
        if vlm_audio_path and not os.path.exists(vlm_audio_path):
            print(
                f"WARNING [{episode_name}]: Audio file not found: {vlm_audio_path}, "
                "falling back to no audio"
            )
            vlm_audio_path = None
            episode_audio_mode = AUDIO_MODE_NONE

        save_audio_path = files["audio"] if os.path.exists(files["audio"]) else None
        gaze_points = None
        if gaze_crop_zoom > 1.0:
            gaze_points = load_gaze_points(files["gaze"])
            if gaze_points is None:
                print(
                    f"WARNING [{episode_name}]: gaze crop requested but gaze data not found/invalid; "
                    "falling back to uncropped video"
                )
        client = genai.Client(api_key=api_key)

        log_entries, final_queue = run_simulation(
            episode_name=episode_name,
            video_path=video_path,
            audio_path=vlm_audio_path,
            save_audio_path=save_audio_path,
            audio_mode=episode_audio_mode,
            model=model,
            client=client,
            output_dir=ep_output_dir,
            task_vocabulary_section=task_vocabulary_section,
            video_fps_hint=video_fps_hint,
            target_resolution=target_resolution,
            simulation_timing=simulation_timing,
            gaze_points=gaze_points,
            gaze_crop_zoom=gaze_crop_zoom,
            verbose=verbose,
        )

        log_path = os.path.join(ep_output_dir, "keyframe_queue_status.json")
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved queue log: {log_path}")

        sim_path = os.path.join(ep_output_dir, "offline_simulation.mp4")
        render_offline_simulation(
            video_path=video_path,
            log_entries=log_entries,
            output_path=sim_path,
            audio_path=files["audio"] if os.path.exists(files["audio"]) else None,
        )

        print_simulation_summary(log_entries, final_queue, episode_name)

        return {
            "episode": episode_name,
            "ok": True,
            "output_dir": ep_output_dir,
            "sim_path": sim_path,
            "log_path": log_path,
            "vlm_calls": len(log_entries),
            "total_cost_usd": round(
                sum(entry["cost"]["cost_usd"] for entry in log_entries), 4
            ),
        }
    except Exception as exc:
        return {
            "episode": episode_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asynchronous keyframe queue simulation with periodic VLM calls",
    )
    parser.add_argument(
        "--episode", help="Episode name (e.g., 2f_store_..._append_2)"
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Process all episodes discovered under --dataset-dir",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of episodes to process in parallel when using --all-episodes",
    )
    parser.add_argument(
        "--dataset-dir",
        default="./processed_file_intervention",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODEL_CHOICES,
        help="Gemini model to use",
    )
    parser.add_argument(
        "--output-dir", default="./eval_results_async", help="Output root directory"
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=4,
        help="FPS hint sent to Gemini (sampling rate)",
    )
    parser.add_argument(
        "--target-resolution",
        default="256",
        help="Resize video segments before VLM (e.g., '256' or '256x256')",
    )
    parser.add_argument(
        "--simulation-timing",
        default=SIMULATION_TIMING_FIXED,
        choices=SIMULATION_TIMING_CHOICES,
        help="How inference results are placed on the offline simulation timeline",
    )
    parser.add_argument(
        "--gaze-crop-zoom",
        type=float,
        default=1.0,
        help="If >1.0, crop each VLM input frame around the gaze point and zoom by this factor",
    )
    parser.add_argument(
        "--task-vocab-file",
        default=None,
        help="Optional text file with scene-specific canonical pick/place strings to inject into the prompt",
    )
    parser.add_argument(
        "--use-gaze-video",
        action="store_true",
        help="Use rgb_with_gaze.mp4 instead of rgb.mp4",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key (optional; falls back to env vars or .env)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for --whisper (optional; falls back to env vars or .env)",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--whisper",
        action="store_true",
        help="Enable Whisper transcription per segment → caption overlay + prompt text "
        "(requires OPENAI_API_KEY). Can be combined with --voiced-video.",
    )
    parser.add_argument(
        "--voiced-video",
        action="store_true",
        help="Merge mp4+wav → send audio to Gemini natively. Can be combined with --whisper.",
    )

    args = parser.parse_args()

    if args.all_episodes and args.episode:
        parser.error("Use either --episode or --all-episodes, not both.")
    if not args.all_episodes and not args.episode:
        parser.error("Provide --episode or use --all-episodes.")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1.")
    if args.gaze_crop_zoom < 1.0:
        parser.error("--gaze-crop-zoom must be >= 1.0.")

    # Resolve API key
    api_key = resolve_gemini_api_key(args.api_key)
    if not api_key:
        print(
            "ERROR: No Gemini API key found. Set GEMINI_API_KEY env var or use --api-key."
        )
        return

    target_res = parse_resolution(args.target_resolution)
    task_vocabulary_section = load_task_vocab_hints(args.task_vocab_file)

    # Determine audio mode
    if args.whisper and args.voiced_video:
        audio_mode = AUDIO_MODE_WHISPER_VOICED
        openai_api_key = resolve_openai_api_key(args.openai_api_key)
        if not openai_api_key:
            print(
                "ERROR: --whisper requires an OpenAI API key. "
                "Set OPENAI_API_KEY, add it to .env, or use --openai-api-key."
            )
            return
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif args.whisper:
        audio_mode = AUDIO_MODE_WHISPER
        openai_api_key = resolve_openai_api_key(args.openai_api_key)
        if not openai_api_key:
            print(
                "ERROR: --whisper requires an OpenAI API key. "
                "Set OPENAI_API_KEY, add it to .env, or use --openai-api-key."
            )
            return
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif args.voiced_video:
        audio_mode = AUDIO_MODE_VOICED
    else:
        audio_mode = AUDIO_MODE_NONE

    episodes = [args.episode]
    if args.all_episodes:
        episodes = list_episode_names(args.dataset_dir)
        if not episodes:
            print(f"ERROR: No episodes found in {args.dataset_dir}")
            return
        print(
            f"Discovered {len(episodes)} episode(s) in {args.dataset_dir}. "
            f"Running with jobs={args.jobs}."
        )
    else:
        # Resolve episode files
        files = resolve_episode_files(args.dataset_dir, args.episode)
        video_path = files["rgb_with_gaze"] if args.use_gaze_video else files["rgb"]
        if not os.path.exists(video_path):
            print(f"ERROR: Video not found: {video_path}")
            print(f"Available episodes:")
            for name in list_episode_names(args.dataset_dir):
                print(f"  {name}")
            return

    worker_kwargs = {
        "dataset_dir": args.dataset_dir,
        "output_root_dir": args.output_dir,
        "use_gaze_video": args.use_gaze_video,
        "audio_mode": audio_mode,
        "model": args.model,
        "api_key": api_key,
        "task_vocabulary_section": task_vocabulary_section,
        "video_fps_hint": args.video_fps,
        "target_resolution": target_res,
        "simulation_timing": args.simulation_timing,
        "gaze_crop_zoom": args.gaze_crop_zoom,
        "verbose": args.verbose,
    }

    if len(episodes) == 1:
        result = process_episode(episode_name=episodes[0], **worker_kwargs)
        if not result["ok"]:
            print(f"ERROR [{result['episode']}]: {result['error']}")
        return

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.jobs, len(episodes))
    ) as executor:
        future_map = {
            executor.submit(process_episode, episode_name=episode, **worker_kwargs): episode
            for episode in episodes
        }
        for future in concurrent.futures.as_completed(future_map):
            episode = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "episode": episode,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            if result["ok"]:
                print(
                    f"[done] {episode}  "
                    f"calls={result['vlm_calls']}  "
                    f"cost=${result['total_cost_usd']:.4f}  "
                    f"video={result['sim_path']}"
                )
            else:
                print(f"[failed] {episode}  {result['error']}")

    ok_results = [result for result in results if result["ok"]]
    failed_results = [result for result in results if not result["ok"]]
    print(f"\nBatch finished: {len(ok_results)} succeeded, {len(failed_results)} failed.")
    if failed_results:
        print("Failed episodes:")
        for result in failed_results:
            print(f"  {result['episode']}: {result['error']}")


if __name__ == "__main__":
    main()
