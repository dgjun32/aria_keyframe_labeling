#!/usr/bin/env python3
"""
Keyframe Labeler + Dataset Exporter
====================================
Interactive Tkinter GUI for:
  - RGB video playback with live gaze overlay
  - Whisper transcript overlay (word-level)
  - Angular velocity plot (scrolling line with cursor)
  - Unified label annotation (1-frame + range)
  - Automatic dataset export on save

Usage:
  python labeler.py
  python labeler.py --prefix 2f_store_dongjun_setup_single_object

Controls:
  Right / D / L           : +2 frames
  Left  / A / H           : -2 frames
  Shift+Right / Shift+D   : +20 frames
  Shift+Left  / Shift+A   : -20 frames
  Ctrl+Right              : +200 frames
  Ctrl+Left               : -200 frames
  Plot drag               : Add one label range
  [                       : Mark label start
  ]                       : Mark label end (add range)
  Ctrl+Z                  : Undo last label range
  Space                   : Toggle 1-frame label
  Delete / Backspace      : Remove selected label
  P                       : Play / Pause
  , / .                   : Prev / Next task
  S                       : Save annotations + export dataset
  Q / Escape              : Quit (saves first)
"""

from __future__ import annotations

import argparse
import glob
import os
import json
import re
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_loader import TaskData, discover_tasks

# ── config ────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROC_DIR = os.path.join(PROJECT_DIR, "preproc_files")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
SAVE_PATH = os.path.join(PREPROC_DIR, "keyframe_annotations.json")
LABEL_SAVE_PATH = os.path.join(PREPROC_DIR, "label_annotations.json")
LEGACY_SEGMENT_SAVE_PATH = os.path.join(PREPROC_DIR, "segment_annotations.json")
EXPORT_MAP_PATH = os.path.join(PREPROC_DIR, "export_mapping.json")
EXPORT_SETTINGS_PATH = os.path.join(PREPROC_DIR, "export_settings.json")
GAZE_OFFSET_PATH = os.path.join(PREPROC_DIR, "gaze_offsets.json")
DISPLAY_W = 640  # display width (aspect ratio preserved)
TASK_META_PATTERN = re.compile(
    r"^(?P<setting>[^_]+_[^_]+)_(?P<user>[^_]+)_(?P<option>[^_]+)"
    r"_(?P<setup>[^_]+_[^_]+)_(?P<task>.+?)(?:_(?P<episode>\d+))?$"
)


# ── Whisper helpers (inlined from whisper_cache.py) ───────────────────────
def _convert_to_mono(wav_path: str) -> str:
    """Convert multi-channel WAV to 16kHz mono temp file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", wav_path,
         "-ac", "1", "-ar", "16000", tmp.name],
        check=True,
    )
    return tmp.name


def _transcribe_task(client, audio_path: str) -> dict:
    """Transcribe a single audio file via Whisper API."""
    mono_path = _convert_to_mono(audio_path)
    try:
        with open(mono_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=f, language="en",
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        result = {"text": response.text, "words": []}
        if hasattr(response, "words") and response.words:
            for w in response.words:
                result["words"].append(
                    {"word": w.word, "start": w.start, "end": w.end}
                )
        return result
    finally:
        os.unlink(mono_path)


# ══════════════════════════════════════════════════════════════════════════
class LabelerApp:
    def __init__(self, root: tk.Tk, prefix: str | None = None):
        self.root = root
        self.root.title("Keyframe Labeler")
        self.root.configure(bg="#1e1e1e")

        # ── state ──
        self.prefix = prefix or ""
        self.naming_mode = "task_name"  # task_name | prefix_index
        self.export_dir = DATASET_DIR
        self.video_scale = 60  # percent
        self.task_names = self._sort_task_names(discover_tasks(PREPROC_DIR))
        self.task_idx = 0
        self.task_data: TaskData | None = None
        self.annotations: dict[str, list[dict]] = {}  # {task: [{frame: int}]}
        self.segments: dict[str, list[dict]] = {}     # {task: [{start: int, end: int}]}
        self.segment_colors = [
            "#5dade2", "#58d68d", "#f5b041", "#af7ac5",
            "#ec7063", "#48c9b0", "#f4d03f",
        ]
        self.playing = False
        self._play_thread: threading.Thread | None = None
        self._audio_proc: subprocess.Popen | None = None
        self._ffplay_path: str | None = None
        self._warned_no_ffplay = False
        self._photo = None
        self.cur_frame = 0
        self.disp_w = DISPLAY_W
        self.disp_h = DISPLAY_W  # updated on task load
        self._drag_start_frame: int | None = None
        self._drag_preview = None
        self._drag_mode: str | None = None  # create | resize_start | resize_end
        self._drag_target_idx: int | None = None
        self._drag_fixed_frame: int | None = None
        self._segment_start: int | None = None  # for [ ] segment marking
        self._selected_segment_idx: int | None = None  # selected label range
        self.gaze_offsets: dict[str, dict[str, float]] = {}  # task-specific overrides
        self.gaze_user_offsets: dict[str, dict[str, float]] = {}  # user defaults
        self.last_gaze_offset = {"x": 0.0, "y": 0.0}
        self._updating_offset_ui = False
        self._updating_export_ui = False
        self._updating_scale_ui = False
        self._updating_label_list = False
        self._switching_task = False

        # ── task→episode mapping (persisted across runs) ──
        self.export_map: dict[str, str] = {}  # {task_name: episode_name}
        self._load_export_map()
        self._load_export_settings(cli_prefix=prefix)
        self._load_gaze_offsets()

        # ── load saved annotations ──
        self._load_annotations()
        self._load_segments()
        self._merge_legacy_keyframes_into_segments()

        # ── ensure Whisper transcripts ──
        self._ensure_transcripts()

        # ── build UI ──
        self._build_ui()
        self._init_plot()

        # ── load first task ──
        if self.task_names:
            self._load_task(0)

        # ── key bindings ──
        self._bind_keys()

    # ═══════════════════════════════════════════════════════════════════════
    #  PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════
    def _load_export_map(self):
        """Load task→episode mapping from previous sessions."""
        if os.path.exists(EXPORT_MAP_PATH):
            try:
                with open(EXPORT_MAP_PATH) as f:
                    self.export_map = json.load(f)
            except Exception as e:
                print(f"[export_map load error] {e}")

    def _save_export_map(self):
        os.makedirs(os.path.dirname(EXPORT_MAP_PATH), exist_ok=True)
        with open(EXPORT_MAP_PATH, "w") as f:
            json.dump(self.export_map, f, indent=2)

    def _infer_default_prefix(self, task_name: str | None = None) -> str:
        candidate = task_name
        if candidate is None and self.task_names:
            candidate = self.task_names[0]
        if candidate:
            meta = self._parse_task_metadata(candidate)
            if meta is not None:
                return f"{meta['setting']}_{meta['user']}_{meta['setup']}_{meta['task']}"
        return "episode"

    def _load_export_settings(self, cli_prefix: str | None = None):
        if os.path.exists(EXPORT_SETTINGS_PATH):
            try:
                with open(EXPORT_SETTINGS_PATH) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.export_dir = raw.get("export_dir", self.export_dir) or self.export_dir
                    self.naming_mode = raw.get("naming_mode", self.naming_mode) or self.naming_mode
                    self.video_scale = int(raw.get("video_scale", self.video_scale))
                    file_prefix = raw.get("prefix", "")
                    if isinstance(file_prefix, str):
                        self.prefix = file_prefix or self.prefix
            except Exception as e:
                print(f"[export_settings load error] {e}")

        if cli_prefix:
            self.prefix = cli_prefix
        if self.naming_mode not in ("task_name", "prefix_index"):
            self.naming_mode = "task_name"
        if not self.prefix:
            self.prefix = self._infer_default_prefix()
        self.video_scale = max(40, min(80, int(self.video_scale)))

    def _save_export_settings(self):
        os.makedirs(os.path.dirname(EXPORT_SETTINGS_PATH), exist_ok=True)
        with open(EXPORT_SETTINGS_PATH, "w") as f:
            json.dump(
                {
                    "export_dir": self.export_dir,
                    "naming_mode": self.naming_mode,
                    "prefix": self.prefix,
                    "video_scale": self.video_scale,
                },
                f,
                indent=2,
            )

    def _task_user_name(self, task_name: str) -> str:
        meta = self._parse_task_metadata(task_name)
        if meta is None:
            return "__default__"
        return meta.get("user", "__default__") or "__default__"

    def _load_gaze_offsets(self):
        """Load user defaults + per-task gaze overrides."""
        if not os.path.exists(GAZE_OFFSET_PATH):
            return
        try:
            with open(GAZE_OFFSET_PATH) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[gaze_offsets load error] {e}")
            return

        # New format:
        # {
        #   "user_offsets": {"juheon": {"x": 0.0, "y": 0.0}, ...},
        #   "task_offsets": {"task_a": {"x": 0.0, "y": 0.0}, ...},
        #   "last_used": {"x": 0.0, "y": 0.0}
        # }
        # Legacy fallback: {"task_a": {"x": ..., "y": ...}, ...}
        task_offsets = raw.get("task_offsets", raw) if isinstance(raw, dict) else {}
        user_offsets = raw.get("user_offsets", {}) if isinstance(raw, dict) else {}

        if isinstance(user_offsets, dict):
            for user_name, off in user_offsets.items():
                if not isinstance(off, dict):
                    continue
                x = float(off.get("x", 0.0))
                y = float(off.get("y", 0.0))
                self.gaze_user_offsets[user_name] = {"x": x, "y": y}

        if isinstance(task_offsets, dict):
            for task, off in task_offsets.items():
                if not isinstance(off, dict):
                    continue
                x = float(off.get("x", 0.0))
                y = float(off.get("y", 0.0))
                self.gaze_offsets[task] = {"x": x, "y": y}

        if isinstance(raw, dict) and isinstance(raw.get("last_used"), dict):
            self.last_gaze_offset = {
                "x": float(raw["last_used"].get("x", 0.0)),
                "y": float(raw["last_used"].get("y", 0.0)),
            }

    def _save_gaze_offsets(self):
        os.makedirs(os.path.dirname(GAZE_OFFSET_PATH), exist_ok=True)
        with open(GAZE_OFFSET_PATH, "w") as f:
            json.dump(
                {
                    "user_offsets": self.gaze_user_offsets,
                    "task_offsets": self.gaze_offsets,
                    "last_used": self.last_gaze_offset,
                },
                f,
                indent=2,
            )

    def _ensure_user_gaze_offset(self, task_name: str):
        """Ensure the task's user has a default offset."""
        user_name = self._task_user_name(task_name)
        if user_name not in self.gaze_user_offsets:
            seed = self.gaze_offsets.get(task_name, self.last_gaze_offset)
            self.gaze_user_offsets[user_name] = {
                "x": float(seed.get("x", 0.0)),
                "y": float(seed.get("y", 0.0)),
            }
        off = self.gaze_user_offsets[user_name]
        self.last_gaze_offset = {"x": float(off["x"]), "y": float(off["y"])}

    def _get_user_gaze_offset(self, task_name: str) -> dict[str, float]:
        self._ensure_user_gaze_offset(task_name)
        user_name = self._task_user_name(task_name)
        return self.gaze_user_offsets.get(user_name, {"x": 0.0, "y": 0.0})

    def _has_task_gaze_override(self, task_name: str) -> bool:
        return task_name in self.gaze_offsets

    def _get_effective_gaze_offset(self, task_name: str) -> dict[str, float]:
        if self._has_task_gaze_override(task_name):
            return self.gaze_offsets[task_name]
        return self._get_user_gaze_offset(task_name)

    def _enable_task_gaze_override(self, task_name: str):
        if task_name not in self.gaze_offsets:
            base = self._get_effective_gaze_offset(task_name)
            self.gaze_offsets[task_name] = {
                "x": float(base.get("x", 0.0)),
                "y": float(base.get("y", 0.0)),
            }

    def _disable_task_gaze_override(self, task_name: str):
        self.gaze_offsets.pop(task_name, None)

    def _load_annotations(self):
        if os.path.exists(SAVE_PATH):
            try:
                with open(SAVE_PATH) as f:
                    raw = json.load(f)
                # Migrate any old format to current {frame: int} dicts
                for task, val in raw.items():
                    if val and isinstance(val[0], int):
                        # Old format: plain list of ints
                        raw[task] = [{"frame": v} for v in val]
                    elif val and isinstance(val[0], dict):
                        # Strip labels from old labeled format
                        raw[task] = [{"frame": kf["frame"]} for kf in val]
                self.annotations = raw
            except Exception as e:
                print(f"[load error] {e}")

    def _load_segments(self):
        load_path = None
        if os.path.exists(LABEL_SAVE_PATH):
            load_path = LABEL_SAVE_PATH
        elif os.path.exists(LEGACY_SEGMENT_SAVE_PATH):
            load_path = LEGACY_SEGMENT_SAVE_PATH
        if load_path is None:
            return

        try:
            with open(load_path) as f:
                raw = json.load(f)
            cleaned = {}
            for task, segs in raw.items():
                if not isinstance(segs, list):
                    continue
                valid = []
                for seg in segs:
                    if not isinstance(seg, dict):
                        continue
                    if "start" not in seg or "end" not in seg:
                        continue
                    s = int(seg["start"])
                    e = int(seg["end"])
                    if e < s:
                        s, e = e, s
                    valid.append({"start": s, "end": e})
                cleaned[task] = valid
            self.segments = cleaned
        except Exception as e:
            print(f"[label load error] {e}")

    def _save_segments(self):
        os.makedirs(os.path.dirname(LABEL_SAVE_PATH), exist_ok=True)
        with open(LABEL_SAVE_PATH, "w") as f:
            json.dump(self.segments, f, indent=2)

    def _sync_annotations_from_segments(self, task_name: str | None = None):
        """Keep legacy keyframe file in sync from point labels (start == end)."""
        if task_name is None:
            tasks = set(self.task_names) | set(self.segments.keys()) | set(self.annotations.keys())
        else:
            tasks = {task_name}

        for task in tasks:
            segs = self.segments.get(task, [])
            points = []
            for seg in segs:
                s = int(seg.get("start", -1))
                e = int(seg.get("end", -1))
                if e < s:
                    s, e = e, s
                if s == e and s >= 0:
                    points.append({"frame": s})
            points.sort(key=lambda x: x["frame"])
            self.annotations[task] = points

    def _merge_legacy_keyframes_into_segments(self):
        """Migrate old keyframe-only data into unified label ranges."""
        for task, kfs in self.annotations.items():
            segs = self.segments.setdefault(task, [])
            existing = {
                (int(min(seg.get("start", -1), seg.get("end", -1))),
                 int(max(seg.get("start", -1), seg.get("end", -1))))
                for seg in segs
            }
            for kf in kfs:
                frame = int(kf.get("frame", -1))
                if frame < 0:
                    continue
                key = (frame, frame)
                if key not in existing:
                    segs.append({"start": frame, "end": frame})
                    existing.add(key)
            segs.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))

        self._sync_annotations_from_segments()

    def _save_annotations(self):
        self._sync_annotations_from_segments()
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as f:
            json.dump(self.annotations, f, indent=2)
        self._export_dataset()
        self._save_segments()
        self._save_gaze_offsets()
        self._save_export_settings()
        self._status(
            f"Saved + exported to {self.export_dir} "
            f"(mode: {self.naming_mode}, prefix: {self.prefix})"
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  WHISPER TRANSCRIPT CACHING
    # ═══════════════════════════════════════════════════════════════════════
    def _ensure_transcripts(self):
        """Run Whisper on tasks missing transcripts (requires OPENAI_API_KEY)."""
        transcript_dir = os.path.join(PREPROC_DIR, "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[info] OPENAI_API_KEY not set -- skipping transcription")
            return

        missing = []
        for task in self.task_names:
            tp = os.path.join(transcript_dir, f"{task}_transcript.json")
            if not os.path.exists(tp):
                missing.append(task)

        if not missing:
            return

        print(f"[whisper] Transcribing {len(missing)} task(s)...")
        from openai import OpenAI
        client = OpenAI()

        for task in missing:
            audio_path = os.path.join(PREPROC_DIR, f"{task}_audio.wav")
            if not os.path.exists(audio_path):
                continue
            try:
                result = _transcribe_task(client, audio_path)
                out_path = os.path.join(transcript_dir, f"{task}_transcript.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  {task}: {len(result['words'])} words")
            except Exception as e:
                print(f"  {task}: ERROR {e}")

        print("[whisper] Done")

    # ═══════════════════════════════════════════════════════════════════════
    #  DATASET EXPORT
    # ═══════════════════════════════════════════════════════════════════════
    def _next_episode_number(self, base_dir: str, prefix: str) -> int:
        """Find next available suffix number in base_dir for given prefix."""
        os.makedirs(base_dir, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        max_num = 0
        for name in os.listdir(base_dir):
            m = pattern.match(name)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return max_num + 1

    def _export_dataset(self):
        """Export labeled episodes using current export settings."""
        base_dir = self.export_dir or DATASET_DIR
        os.makedirs(base_dir, exist_ok=True)

        exported = 0
        for task_name in self.task_names:
            segs = self.segments.get(task_name, [])
            if not segs:
                continue

            if self.naming_mode == "task_name":
                episode_name = task_name
            else:
                if task_name in self.export_map:
                    episode_name = self.export_map[task_name]
                else:
                    prefix = (self.prefix or "").strip() or self._infer_default_prefix(task_name)
                    next_num = self._next_episode_number(base_dir, prefix)
                    episode_name = f"{prefix}_{next_num}"
                    self.export_map[task_name] = episode_name

            episode_dir = os.path.join(base_dir, episode_name)
            os.makedirs(episode_dir, exist_ok=True)

            # 1. video.mp4
            src_video = os.path.join(PREPROC_DIR, f"{task_name}_rgb.mp4")
            dst_video = os.path.join(episode_dir, "video.mp4")
            if os.path.exists(src_video):
                shutil.copy2(src_video, dst_video)

            # 2. transcript.json
            src_transcript = os.path.join(
                PREPROC_DIR, "transcripts", f"{task_name}_transcript.json"
            )
            dst_transcript = os.path.join(episode_dir, "transcript.json")
            if os.path.exists(src_transcript):
                shutil.copy2(src_transcript, dst_transcript)

            # 3. gaze.json
            self._export_gaze_json(task_name, episode_dir)

            # 4. annotations.json (+ legacy segments.json)
            self._export_annotations_json(task_name, episode_dir, segs)

            # 5. labels.npy
            self._export_labels_npy(task_name, episode_dir, segs)

            exported += 1

        # Persist the mapping so next run remembers assignments
        self._save_export_map()
        print(
            f"[export] {exported} episode(s) exported "
            f"(dir: {base_dir}, mode: {self.naming_mode})"
        )

    def _export_gaze_json(self, task_name: str, episode_dir: str):
        """Convert gaze.npz + pitch_yaw.npz to gaze.json."""
        gaze_path = os.path.join(PREPROC_DIR, f"{task_name}_gaze.npz")
        py_path = os.path.join(PREPROC_DIR, f"{task_name}_pitch_yaw.npz")
        offset = self._get_effective_gaze_offset(task_name)
        offset_x = float(offset.get("x", 0.0))
        offset_y = float(offset.get("y", 0.0))

        gaze_data = None
        if os.path.exists(gaze_path):
            gaze_data = np.load(gaze_path)["gaze"]  # (T, 2)

        py_data = None
        if os.path.exists(py_path):
            py_data = np.load(py_path)["pitch_yaw"]  # (T, 6)

        T = 0
        if gaze_data is not None:
            T = len(gaze_data)
        elif py_data is not None:
            T = len(py_data)

        frames = []
        for i in range(T):
            entry = {"frame_idx": i}
            if gaze_data is not None:
                entry["gaze_x"] = float(gaze_data[i, 0] + offset_x)
                entry["gaze_y"] = float(gaze_data[i, 1] + offset_y)
            if py_data is not None:
                entry["pitch"] = float(py_data[i, 0])
                entry["pitch_lower"] = float(py_data[i, 1])
                entry["pitch_upper"] = float(py_data[i, 2])
                entry["yaw"] = float(py_data[i, 3])
                entry["yaw_lower"] = float(py_data[i, 4])
                entry["yaw_upper"] = float(py_data[i, 5])
            frames.append(entry)

        with open(os.path.join(episode_dir, "gaze.json"), "w") as f:
            json.dump({"frames": frames}, f, indent=2)

    def _export_annotations_json(self, task_name: str, episode_dir: str, segs: list[dict]):
        video_path = os.path.join(PREPROC_DIR, f"{task_name}_rgb.mp4")
        fps = 15.0
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            fps_from_video = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps_from_video > 0:
                fps = float(fps_from_video)

        annotations = []
        for seg in segs:
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            if s < 0:
                continue
            annotations.append(
                {
                    "start_frame": s,
                    "end_frame": e,
                    "start_sec": float(s / fps),
                    "end_sec": float((e + 1) / fps),
                    "duration_sec": float((e - s + 1) / fps),
                    "kind": "label",
                }
            )

        with open(os.path.join(episode_dir, "annotations.json"), "w") as f:
            json.dump({"fps": fps, "annotations": annotations}, f, indent=2)

        # Backward compatibility for older consumers.
        with open(os.path.join(episode_dir, "segments.json"), "w") as f:
            json.dump({"segments": segs}, f, indent=2)

    def _export_labels_npy(self, task_name: str, episode_dir: str, segs: list[dict]):
        """Create T-length binary label array from point+range labels."""
        video_path = os.path.join(PREPROC_DIR, f"{task_name}_rgb.mp4")
        if not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        labels = np.zeros(T, dtype=np.int8)

        for seg in segs:
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            s = max(0, min(T - 1, s))
            e = max(0, min(T - 1, e))
            if e >= s:
                labels[s:e + 1] = 1

        np.save(os.path.join(episode_dir, "labels.npy"), labels)

    # ═══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.task_var = tk.StringVar()
        self._updating_task_list = False

        # ── split layout: left task list + right workspace ──
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_frame, bg="#1a1a1a", width=420)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Tasks", bg="#1a1a1a", fg="#cccccc",
                 font=("Helvetica", 12, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        self.task_tree = ttk.Treeview(
            left_panel,
            columns=("done", "task", "setup", "user", "ep"),
            show="headings",
            style="Task.Treeview",
            selectmode="browse",
        )
        self.task_tree.heading("done", text="Done")
        self.task_tree.heading("task", text="Task")
        self.task_tree.heading("setup", text="Setup")
        self.task_tree.heading("user", text="User")
        self.task_tree.heading("ep", text="Ep")
        self.task_tree.column("done", width=46, minwidth=46, stretch=False, anchor=tk.CENTER)
        self.task_tree.column("task", width=96, minwidth=84, stretch=True, anchor=tk.W)
        self.task_tree.column("setup", width=112, minwidth=96, stretch=True, anchor=tk.W)
        self.task_tree.column("user", width=72, minwidth=64, stretch=False, anchor=tk.W)
        self.task_tree.column("ep", width=40, minwidth=40, stretch=False, anchor=tk.CENTER)
        self.task_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        self.task_tree.bind("<<TreeviewSelect>>", self._on_task_tree_select)
        self.task_tree.bind("<Button-1>", self._on_task_tree_click, add="+")
        self.task_tree.bind("<Enter>", self._on_task_tree_enter, add="+")
        self.task_tree.tag_configure("done", foreground="#9ad67d")
        self.task_tree.tag_configure("todo", foreground="#dddddd")

        task_scroll = ttk.Scrollbar(left_panel, orient=tk.VERTICAL,
                                    command=self.task_tree.yview)
        task_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=(0, 8))
        self.task_tree.configure(yscrollcommand=task_scroll.set)

        right_panel = tk.Frame(main_frame, bg="#1e1e1e")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # ── top bar ──
        top = tk.Frame(right_panel, bg="#1e1e1e")
        top.pack(fill=tk.X, pady=(0, 2))

        tk.Label(top, text="Current Task:", bg="#1e1e1e", fg="#cccccc",
                 font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.task_name_label = tk.Label(top, text="-", bg="#1e1e1e", fg="#ffffff",
                                        font=("Helvetica", 12, "bold"))
        self.task_name_label.pack(side=tk.LEFT, padx=8)

        self.prog_label = tk.Label(top, text="", bg="#1e1e1e", fg="#888888",
                                   font=("Helvetica", 11))
        self.prog_label.pack(side=tk.LEFT, padx=10)

        self.done_label = tk.Label(top, text="", bg="#1e1e1e", fg="#4ec94e",
                                   font=("Helvetica", 11, "bold"))
        self.done_label.pack(side=tk.RIGHT, padx=6)

        # ── body split: center media + right metadata/settings ──
        body = tk.Frame(right_panel, bg="#1e1e1e")
        body.pack(fill=tk.BOTH, expand=True)

        center_panel = tk.Frame(body, bg="#1e1e1e")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side_panel = tk.Frame(body, bg="#202124", width=260)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        side_panel.pack_propagate(False)

        # ── parsed metadata panel (always visible) ──
        self.meta_frame = tk.Frame(side_panel, bg="#252526", bd=1, relief=tk.FLAT)
        self.meta_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(self.meta_frame, text="Parsed Metadata", bg="#252526", fg="#dcdcdc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))

        meta_grid = tk.Frame(self.meta_frame, bg="#252526")
        meta_grid.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.meta_value_labels = {}
        meta_fields = [
            ("setting", "setting"),
            ("user", "user"),
            ("option", "option"),
            ("setup", "setup"),
            ("task", "task"),
            ("episode", "episode"),
        ]
        for i, (key, title) in enumerate(meta_fields):
            tk.Label(meta_grid, text=f"{title}:", bg="#252526", fg="#8fb8ff",
                     font=("Courier", 10)).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=1)
            value = tk.Label(meta_grid, text="-", bg="#252526", fg="#e6e6e6",
                             font=("Courier", 10))
            value.grid(row=i, column=1, sticky="w", pady=1)
            self.meta_value_labels[key] = value

        # ── gaze calibration offsets (compact, right side) ──
        offset_card = tk.Frame(side_panel, bg="#252526", bd=1, relief=tk.FLAT)
        offset_card.pack(fill=tk.X, pady=(0, 8))
        tk.Label(offset_card, text="Gaze Offset (px)", bg="#252526", fg="#dcdcdc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 4))

        offset_grid = tk.Frame(offset_card, bg="#252526")
        offset_grid.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(offset_grid, text="X", bg="#252526", fg="#aaaaaa",
                 font=("Helvetica", 10)).grid(row=0, column=0, sticky="w")
        tk.Label(offset_grid, text="Y", bg="#252526", fg="#aaaaaa",
                 font=("Helvetica", 10)).grid(row=1, column=0, sticky="w")

        self.gaze_offset_x_var = tk.DoubleVar(value=0.0)
        self.gaze_offset_y_var = tk.DoubleVar(value=0.0)

        scale_cfg = dict(
            from_=-200, to=200, resolution=1, orient=tk.HORIZONTAL, length=130,
            bg="#252526", fg="#cccccc", troughcolor="#444444",
            highlightthickness=0, showvalue=False,
        )
        self.gaze_offset_x_scale = tk.Scale(
            offset_grid, variable=self.gaze_offset_x_var,
            command=self._on_gaze_offset_change, **scale_cfg
        )
        self.gaze_offset_y_scale = tk.Scale(
            offset_grid, variable=self.gaze_offset_y_var,
            command=self._on_gaze_offset_change, **scale_cfg
        )
        self.gaze_offset_x_scale.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.gaze_offset_y_scale.grid(row=1, column=1, sticky="ew", padx=(6, 6))

        self.gaze_offset_value_label = tk.Label(
            offset_card, text="X +0 px | Y +0 px",
            bg="#252526", fg="#cccccc", font=("Courier", 10),
        )
        self.gaze_offset_value_label.pack(anchor="w", padx=8, pady=(0, 6))

        self.gaze_offset_scope_label = tk.Label(
            offset_card, text="Applies to: user default",
            bg="#252526", fg="#8fb8ff", font=("Helvetica", 9),
        )
        self.gaze_offset_scope_label.pack(anchor="w", padx=8, pady=(0, 4))

        self.gaze_task_override_var = tk.BooleanVar(value=False)
        self.gaze_override_check = tk.Checkbutton(
            offset_card,
            text="Custom for this task",
            variable=self.gaze_task_override_var,
            command=self._on_gaze_override_toggle,
            bg="#252526",
            fg="#d8d8d8",
            activebackground="#252526",
            activeforeground="#ffffff",
            selectcolor="#2d2d2d",
            highlightthickness=0,
            borderwidth=0,
            font=("Helvetica", 9),
        )
        self.gaze_override_check.pack(anchor="w", padx=8, pady=(0, 6))

        self.gaze_reset_btn = ttk.Button(
            offset_card, text="Reset offset", command=self._reset_gaze_offset,
        )
        self.gaze_reset_btn.pack(anchor="w", padx=8, pady=(0, 8))
        offset_grid.grid_columnconfigure(1, weight=1)

        # ── view settings ──
        view_card = tk.Frame(side_panel, bg="#252526", bd=1, relief=tk.FLAT)
        view_card.pack(fill=tk.X, pady=(0, 8))
        tk.Label(view_card, text="View Settings", bg="#252526", fg="#dcdcdc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 4))
        self.video_scale_var = tk.IntVar(value=int(self.video_scale))
        scale_row = tk.Frame(view_card, bg="#252526")
        scale_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(scale_row, text="Video size", bg="#252526", fg="#aaaaaa",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.video_scale_label = tk.Label(
            scale_row, text=f"{int(self.video_scale)}%", bg="#252526",
            fg="#e6e6e6", font=("Courier", 10)
        )
        self.video_scale_label.pack(side=tk.RIGHT)
        self.video_scale_slider = tk.Scale(
            view_card, from_=40, to=80, resolution=5, orient=tk.HORIZONTAL, length=170,
            variable=self.video_scale_var, command=self._on_video_scale_change,
            bg="#252526", fg="#cccccc", troughcolor="#444444",
            highlightthickness=0, showvalue=False,
        )
        self.video_scale_slider.pack(anchor="w", padx=8, pady=(0, 8))

        # ── export settings ──
        export_card = tk.Frame(side_panel, bg="#252526", bd=1, relief=tk.FLAT)
        export_card.pack(fill=tk.X, pady=(0, 8))
        tk.Label(export_card, text="Export Settings", bg="#252526", fg="#dcdcdc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 4))

        self.export_dir_var = tk.StringVar(value=self.export_dir)
        self.naming_mode_var = tk.StringVar(value=self.naming_mode)
        self.prefix_var = tk.StringVar(value=self.prefix)

        tk.Label(export_card, text="Export dir", bg="#252526", fg="#8fb8ff",
                 font=("Courier", 10)).pack(anchor="w", padx=8)
        self.export_dir_entry = tk.Entry(
            export_card, textvariable=self.export_dir_var,
            bg="#2b2b2b", fg="#e6e6e6", insertbackground="#e6e6e6",
            relief=tk.FLAT, highlightthickness=1, highlightbackground="#3a3a3a",
        )
        self.export_dir_entry.pack(fill=tk.X, padx=8, pady=(2, 6))
        self.export_dir_entry.bind("<FocusOut>", self._on_export_setting_change)
        self.export_dir_entry.bind("<Return>", self._on_export_setting_change)

        tk.Label(export_card, text="Naming", bg="#252526", fg="#8fb8ff",
                 font=("Courier", 10)).pack(anchor="w", padx=8)
        self.naming_mode_combo = ttk.Combobox(
            export_card,
            textvariable=self.naming_mode_var,
            values=["task_name", "prefix_index"],
            state="readonly",
            width=20,
        )
        self.naming_mode_combo.pack(anchor="w", padx=8, pady=(2, 6))
        self.naming_mode_combo.bind("<<ComboboxSelected>>", self._on_export_setting_change)

        tk.Label(export_card, text="Prefix (prefix_index mode)", bg="#252526", fg="#8fb8ff",
                 font=("Courier", 10)).pack(anchor="w", padx=8)
        self.prefix_entry = tk.Entry(
            export_card, textvariable=self.prefix_var,
            bg="#2b2b2b", fg="#e6e6e6", insertbackground="#e6e6e6",
            relief=tk.FLAT, highlightthickness=1, highlightbackground="#3a3a3a",
        )
        self.prefix_entry.pack(fill=tk.X, padx=8, pady=(2, 8))
        self.prefix_entry.bind("<FocusOut>", self._on_export_setting_change)
        self.prefix_entry.bind("<Return>", self._on_export_setting_change)

        # ── label list (point + range, unified) ──
        label_card = tk.Frame(side_panel, bg="#252526", bd=1, relief=tk.FLAT)
        label_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        tk.Label(label_card, text="Labels", bg="#252526", fg="#dcdcdc",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 4))

        self.kf_listbox = tk.Listbox(
            label_card, height=12, bg="#2d2d2d", fg="#f0c040",
            selectbackground="#4b5870", font=("Courier", 10),
            relief=tk.FLAT, borderwidth=0, activestyle="none",
        )
        self.kf_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        self.kf_listbox.bind("<<ListboxSelect>>", self._on_kf_select)

        self._label_list_seg_indices: list[int] = []
        lbl_scroll = ttk.Scrollbar(label_card, orient=tk.VERTICAL,
                                   command=self.kf_listbox.yview)
        lbl_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 8), pady=(0, 8))
        self.kf_listbox.configure(yscrollcommand=lbl_scroll.set)

        # ── video canvas ──
        self.canvas = tk.Canvas(center_panel, bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=0, pady=4)

        # ── seek bar ──
        seek_frame = tk.Frame(center_panel, bg="#1e1e1e")
        seek_frame.pack(fill=tk.X, pady=2)

        self.frame_label = tk.Label(seek_frame, text="0 / 0", bg="#1e1e1e",
                                    fg="#cccccc", font=("Courier", 11), width=14)
        self.frame_label.pack(side=tk.LEFT)

        self.seek_var = tk.IntVar()
        self.seek_bar = ttk.Scale(
            seek_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.seek_var, command=self._on_seek,
        )
        self.seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.time_label = tk.Label(seek_frame, text="0.00 s", bg="#1e1e1e",
                                   fg="#cccccc", font=("Courier", 11), width=10)
        self.time_label.pack(side=tk.LEFT)

        # ── plot frame (filled by _init_plot) ──
        self.plot_frame = tk.Frame(center_panel, bg="#1e1e1e")
        self.plot_frame.pack(fill=tk.X, pady=4)

        # ── shortcut help ──
        help_frame = tk.Frame(center_panel, bg="#1e1e1e")
        help_frame.pack(fill=tk.X, pady=(0, 2))
        tk.Label(
            help_frame,
            text="Drag=create/resize label  [ ]=label range  Space=toggle 1-frame label  Del=remove  S=save",
            bg="#1e1e1e", fg="#666666", font=("Courier", 9),
        ).pack(side=tk.LEFT)

        # ── button row ──
        btn_frame = tk.Frame(center_panel, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, pady=(2, 0))

        btn_cfg = dict(
            bg="#333333", fg="#ffffff", relief=tk.FLAT,
            font=("Helvetica", 11), padx=10, pady=4,
            activebackground="#555555", activeforeground="#ffffff",
            cursor="hand2",
        )

        tk.Button(btn_frame, text="Play/Pause P", command=self._toggle_play,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Label start [",
                  command=self._mark_segment_start,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Toggle Label Space",
                  command=self._toggle_keyframe,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Remove Del", command=self._remove_selected_label_or_keyframe,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Save+Export S",
                  bg="#1a4a1a", fg="#4ec94e", relief=tk.FLAT,
                  font=("Helvetica", 11, "bold"), padx=10, pady=4,
                  activebackground="#2a6a2a", activeforeground="#4ec94e",
                  cursor="hand2", command=self._save_annotations).pack(
            side=tk.RIGHT, padx=3)

        self._refresh_task_sidebar()
        self._refresh_export_settings_ui()

        # ── status bar ──
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#111111", fg="#888888", font=("Helvetica", 10),
                 anchor=tk.W, padx=6).pack(fill=tk.X, side=tk.BOTTOM)

    # ═══════════════════════════════════════════════════════════════════════
    #  MATPLOTLIB PLOT
    # ═══════════════════════════════════════════════════════════════════════
    def _init_plot(self):
        self.fig = Figure(figsize=(6.4, 2.6), dpi=100, facecolor="#1e1e1e")
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
        self.ax_vel = self.fig.add_subplot(gs[0, 0])
        self.ax_audio = self.fig.add_subplot(gs[1, 0], sharex=self.ax_vel)

        for ax in (self.ax_vel, self.ax_audio):
            ax.set_facecolor("#2d2d2d")
            ax.tick_params(colors="#888888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#555555")

        self.ax_vel.set_ylabel("Ang Vel (deg/s)", color="#aaaaaa", fontsize=9)
        self.ax_audio.set_ylabel("Audio", color="#aaaaaa", fontsize=9)
        self.ax_audio.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)

        self.vel_line, = self.ax_vel.plot([], [], color="#4ec9ff", linewidth=1.0)
        self.audio_line, = self.ax_audio.plot([], [], color="#f39c12", linewidth=1.0, alpha=0.9)
        self.cursor_line_vel = self.ax_vel.axvline(x=0, color="#ff4444", linewidth=1.2)
        self.cursor_line_audio = self.ax_audio.axvline(x=0, color="#ff4444", linewidth=1.2)
        self.kf_plot_lines: list = []
        self.segment_plot_artists: list = []

        self.fig.tight_layout(pad=0.5)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.X, expand=False)
        self.plot_canvas.mpl_connect("button_press_event", self._on_plot_mouse_down)
        self.plot_canvas.mpl_connect("motion_notify_event", self._on_plot_mouse_move)
        self.plot_canvas.mpl_connect("button_release_event", self._on_plot_mouse_up)

    def _update_plot_data(self):
        td = self.task_data
        if td is None:
            return

        if td.time_axis is None or len(td.time_axis) == 0:
            return
        time_axis = td.time_axis

        if td.angular_velocity_deg is not None and len(td.angular_velocity_deg) == len(time_axis):
            vel = td.angular_velocity_deg
        else:
            vel = np.zeros_like(time_axis)
        self.vel_line.set_data(time_axis, vel)

        if td.audio_energy is not None and len(td.audio_energy) == len(time_axis):
            audio = td.audio_energy
        else:
            audio = np.zeros_like(time_axis)
        self.audio_line.set_data(time_axis, audio)

        t_end = (len(time_axis) / td.fps) if len(time_axis) else 1.0
        self.ax_vel.set_xlim(0, max(t_end, 1.0))
        self.ax_audio.set_xlim(0, max(t_end, 1.0))
        self.ax_vel.set_ylim(0, max(float(np.max(vel)) * 1.1, 1.0))
        self.ax_audio.set_ylim(0, max(float(np.max(audio)) * 1.2, 1.0))

        # Remove old exact-frame guide markers
        for line in self.kf_plot_lines:
            line.remove()
        self.kf_plot_lines.clear()

        # Remove old label-range highlights
        for art in self.segment_plot_artists:
            art.remove()
        self.segment_plot_artists.clear()

        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        if self._selected_segment_idx is not None:
            if not (0 <= self._selected_segment_idx < len(segs)):
                self._selected_segment_idx = None

        # Add colored label ranges (supports multiple ranges per task)
        for i, seg in enumerate(segs):
            s = int(seg.get("start", 0))
            e = int(seg.get("end", 0))
            if e < s:
                s, e = e, s
            t0, t1 = self._label_time_bounds(s, e, td.fps)
            color = self.segment_colors[i % len(self.segment_colors)]
            selected = (i == self._selected_segment_idx)
            vel_alpha = 0.32 if selected else 0.17
            audio_alpha = 0.22 if selected else 0.12
            edge = "#ffffff" if selected else color
            vel_lw = 2.0 if selected else 0.8
            audio_lw = 1.3 if selected else 0.6
            self.segment_plot_artists.append(
                self.ax_vel.axvspan(
                    t0, t1, facecolor=color, alpha=vel_alpha,
                    edgecolor=edge, linewidth=vel_lw
                )
            )
            self.segment_plot_artists.append(
                self.ax_audio.axvspan(
                    t0, t1, facecolor=color, alpha=audio_alpha,
                    edgecolor=edge, linewidth=audio_lw
                )
            )

        self.cursor_line_vel.set_xdata([0, 0])
        self.cursor_line_audio.set_xdata([0, 0])
        self.plot_canvas.draw_idle()

    def _update_plot_cursor(self, frame_idx: int):
        if self.task_data is None:
            return
        t = frame_idx / self.task_data.fps
        self.cursor_line_vel.set_xdata([t, t])
        self.cursor_line_audio.set_xdata([t, t])
        self.plot_canvas.draw_idle()

    def _label_time_bounds(self, start_frame: int, end_frame: int, fps: float) -> tuple[float, float]:
        s = int(min(start_frame, end_frame))
        e = int(max(start_frame, end_frame))
        return s / fps, (e + 1) / fps

    def _plot_time_from_event(self, event) -> float | None:
        td = self.task_data
        if td is None or td.fps <= 0:
            return None
        if event.xdata is None:
            return None
        max_t = (td.total_frames - 1) / td.fps if td.total_frames > 0 else 0.0
        return float(np.clip(event.xdata, 0.0, max_t))

    def _plot_frame_from_event(self, event) -> int | None:
        td = self.task_data
        if td is None or td.total_frames <= 0:
            return None
        t = self._plot_time_from_event(event)
        if t is None:
            return None
        frame = int(round(t * td.fps))
        return max(0, min(td.total_frames - 1, frame))

    def _set_drag_preview_frames(self, start_frame: int, end_frame: int):
        td = self.task_data
        if td is None or td.fps <= 0:
            return
        t0, t1 = self._label_time_bounds(start_frame, end_frame, td.fps)
        if self._drag_preview is not None:
            self._drag_preview.remove()
        self._drag_preview = self.ax_vel.axvspan(
            t0, t1, facecolor="#ffffff", alpha=0.12, edgecolor="none"
        )
        self.plot_canvas.draw_idle()

    def _find_resize_handle(self, frame: int) -> tuple[int, str] | None:
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        idx = self._selected_segment_idx
        if idx is None or not (0 <= idx < len(segs)):
            return None

        seg = segs[idx]
        s = int(seg.get("start", -1))
        e = int(seg.get("end", -1))
        if e < s:
            s, e = e, s
        threshold = 2
        near_start = abs(frame - s) <= threshold
        near_end = abs(frame - e) <= threshold
        if not near_start and not near_end:
            return None
        if near_start and near_end:
            if abs(frame - s) <= abs(frame - e):
                return idx, "resize_start"
            return idx, "resize_end"
        if near_start:
            return idx, "resize_start"
        return idx, "resize_end"

    def _update_segment_range(self, seg_idx: int, start_frame: int, end_frame: int):
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        if not (0 <= seg_idx < len(segs)):
            return
        seg = segs[seg_idx]
        new_start = int(min(start_frame, end_frame))
        new_end = int(max(start_frame, end_frame))
        for other in segs:
            if other is seg:
                continue
            s = int(other.get("start", -1))
            e = int(other.get("end", -1))
            if e < s:
                s, e = e, s
            if s == new_start and e == new_end:
                self._status(f"Label [{new_start} — {new_end}] already exists")
                self._after_kf_change()
                return

        seg["start"] = new_start
        seg["end"] = new_end
        segs.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))
        self._selected_segment_idx = segs.index(seg)
        s = int(seg["start"])
        e = int(seg["end"])
        self._status(f"Updated label [{s} — {e}]")
        self._after_kf_change()

    def _on_plot_mouse_down(self, event):
        if event.button != 1 or event.inaxes not in (self.ax_vel, self.ax_audio):
            return
        frame = self._plot_frame_from_event(event)
        if frame is None:
            return
        resize_handle = self._find_resize_handle(frame)
        if resize_handle is not None:
            seg_idx, mode = resize_handle
            seg = self.segments[self.task_names[self.task_idx]][seg_idx]
            s = int(seg.get("start", frame))
            e = int(seg.get("end", frame))
            if e < s:
                s, e = e, s
            self._drag_mode = mode
            self._drag_target_idx = seg_idx
            self._drag_fixed_frame = e if mode == "resize_start" else s
            self._drag_start_frame = frame
            self._set_drag_preview_frames(s, e)
            return

        self._drag_mode = "create"
        self._drag_target_idx = None
        self._drag_fixed_frame = None
        self._drag_start_frame = frame
        self._set_drag_preview_frames(frame, frame)

    def _on_plot_mouse_move(self, event):
        if self._drag_start_frame is None:
            return
        frame = self._plot_frame_from_event(event)
        if frame is None:
            return
        if self._drag_mode == "create":
            self._set_drag_preview_frames(self._drag_start_frame, frame)
        elif self._drag_mode == "resize_start" and self._drag_fixed_frame is not None:
            self._set_drag_preview_frames(frame, self._drag_fixed_frame)
        elif self._drag_mode == "resize_end" and self._drag_fixed_frame is not None:
            self._set_drag_preview_frames(self._drag_fixed_frame, frame)

    def _on_plot_mouse_up(self, event):
        if self._drag_start_frame is None:
            return
        td = self.task_data
        if td is None:
            self._drag_start_frame = None
            self._drag_mode = None
            self._drag_target_idx = None
            self._drag_fixed_frame = None
            return

        start_frame = self._drag_start_frame
        end_frame = self._plot_frame_from_event(event)
        if end_frame is None:
            end_frame = start_frame
        drag_mode = self._drag_mode
        drag_target_idx = self._drag_target_idx
        drag_fixed_frame = self._drag_fixed_frame

        self._drag_start_frame = None
        self._drag_mode = None
        self._drag_target_idx = None
        self._drag_fixed_frame = None
        if self._drag_preview is not None:
            self._drag_preview.remove()
            self._drag_preview = None

        if drag_mode == "resize_start" and drag_target_idx is not None and drag_fixed_frame is not None:
            self._update_segment_range(drag_target_idx, end_frame, drag_fixed_frame)
            self._seek_to(end_frame)
            return
        if drag_mode == "resize_end" and drag_target_idx is not None and drag_fixed_frame is not None:
            self._update_segment_range(drag_target_idx, drag_fixed_frame, end_frame)
            self._seek_to(end_frame)
            return

        # Near-click means selection; drag means create one label range.
        if abs(end_frame - start_frame) < 1:
            self._select_segment_at_frame(end_frame)
            self._seek_to(end_frame)
            self.plot_canvas.draw_idle()
            return

        s = max(0, min(td.total_frames - 1, min(start_frame, end_frame)))
        e = max(0, min(td.total_frames - 1, max(start_frame, end_frame)))
        self._add_segment(s, e)

    # ═══════════════════════════════════════════════════════════════════════
    #  KEY BINDINGS
    # ═══════════════════════════════════════════════════════════════════════
    def _bind_keys(self):
        r = self.root
        r.bind("<Right>", lambda e: self._step(2))
        r.bind("<Left>", lambda e: self._step(-2))
        r.bind("d", lambda e: self._step(2))
        r.bind("a", lambda e: self._step(-2))
        r.bind("l", lambda e: self._step(2))
        r.bind("h", lambda e: self._step(-2))
        r.bind("<Shift-Right>", lambda e: self._step(20))
        r.bind("<Shift-Left>", lambda e: self._step(-20))
        r.bind("<Shift-D>", lambda e: self._step(20))
        r.bind("<Shift-A>", lambda e: self._step(-20))
        r.bind("<Control-Right>", lambda e: self._step(200))
        r.bind("<Control-Left>", lambda e: self._step(-200))
        r.bind("[", lambda e: self._mark_segment_start())
        r.bind("]", lambda e: self._mark_segment_end())
        r.bind("<space>", lambda e: self._toggle_keyframe())
        r.bind("<Delete>", lambda e: self._remove_selected_label_or_keyframe())
        r.bind("<BackSpace>", lambda e: self._remove_selected_label_or_keyframe())
        r.bind("<Control-z>", lambda e: self._undo_last_segment())
        r.bind("p", lambda e: self._toggle_play())
        r.bind("P", lambda e: self._toggle_play())
        r.bind(",", lambda e: self._prev_task())
        r.bind(".", lambda e: self._next_task())
        r.bind("s", lambda e: self._save_annotations())
        r.bind("S", lambda e: self._save_annotations())
        r.bind("q", lambda e: self._quit())
        r.bind("<Escape>", lambda e: self._quit())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    # ═══════════════════════════════════════════════════════════════════════
    #  TASK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    def _sort_task_names(self, task_names: list[str]) -> list[str]:
        """Sort tasks by parsed metadata (task-centric), then by episode."""
        def sort_key(name: str):
            meta = self._parse_task_metadata(name)
            if meta is None:
                return (1, name)

            try:
                ep = int(meta.get("episode", "1"))
            except (TypeError, ValueError):
                ep = 1

            return (
                0,
                meta["task"],
                meta["setup"],
                meta["user"],
                meta["setting"],
                meta["option"],
                ep,
                name,
            )

        return sorted(task_names, key=sort_key)

    def _parse_task_metadata(self, task_name: str) -> dict[str, str] | None:
        m = TASK_META_PATTERN.match(task_name)
        if m is None:
            return None
        meta = m.groupdict()
        meta["episode"] = meta.get("episode") or "1"
        return {
            "setting": meta["setting"],
            "user": meta["user"],
            "option": meta["option"],
            "setup": meta["setup"],
            "task": meta["task"],
            "episode": meta["episode"],
        }

    def _update_task_metadata_panel(self, task_name: str):
        meta = self._parse_task_metadata(task_name)
        values = {
            "setting": "-",
            "user": "-",
            "option": "-",
            "setup": "-",
            "task": "-",
            "episode": "-",
        }
        if meta is not None:
            values.update(meta)
        for k, lbl in self.meta_value_labels.items():
            lbl.config(text=values.get(k, "-"))

    def _refresh_task_sidebar(self):
        if not hasattr(self, "task_tree"):
            return

        self._updating_task_list = True
        try:
            for iid in self.task_tree.get_children():
                self.task_tree.delete(iid)

            for idx, task in enumerate(self.task_names):
                done = bool(self.segments.get(task))
                mark = "[x]" if done else "[ ]"
                meta = self._parse_task_metadata(task)
                if meta is None:
                    values = (mark, task, "-", "-", "-")
                else:
                    values = (
                        mark,
                        meta["task"],
                        meta["setup"],
                        meta["user"],
                        meta["episode"],
                    )
                tag = "done" if done else "todo"
                self.task_tree.insert("", tk.END, iid=str(idx), values=values, tags=(tag,))

            if self.task_names:
                iid = str(self.task_idx)
                self.task_tree.selection_set(iid)
                self.task_tree.focus(iid)
                self.task_tree.see(iid)
        finally:
            self._updating_task_list = False

    def _switch_task(self, new_idx: int):
        if self._switching_task or new_idx == self.task_idx:
            return
        self._switching_task = True
        try:
            self._save_annotations()
            self._load_task(new_idx)
        finally:
            self._switching_task = False

    def _on_task_tree_enter(self, _event=None):
        self.task_tree.focus_set()

    def _on_task_tree_click(self, event):
        try:
            self.root.focus_force()
        except tk.TclError:
            pass
        self.task_tree.focus_set()
        row = self.task_tree.identify_row(event.y)
        if not row:
            return
        new_idx = int(row)
        self.task_tree.selection_set(row)
        self.task_tree.focus(row)
        self.task_tree.see(row)
        if new_idx != self.task_idx:
            # Schedule once to avoid races with tree internal selection callbacks.
            self.root.after(0, lambda idx=new_idx: self._switch_task(idx))

    def _on_task_tree_select(self, _event=None):
        if self._updating_task_list:
            return
        sel = self.task_tree.selection()
        if not sel:
            return
        new_idx = int(sel[0])
        self._switch_task(new_idx)

    def _refresh_export_settings_ui(self):
        prefix_enabled = self.naming_mode_var.get() == "prefix_index"
        state = "normal" if prefix_enabled else "disabled"
        self.prefix_entry.config(state=state)
        self.video_scale_label.config(text=f"{int(self.video_scale_var.get())}%")

    def _on_export_setting_change(self, _event=None):
        if self._updating_export_ui:
            return
        self.export_dir = (self.export_dir_var.get() or DATASET_DIR).strip()
        self.naming_mode = self.naming_mode_var.get() or "task_name"
        self.prefix = (self.prefix_var.get() or "").strip()
        if not self.prefix:
            self.prefix = self._infer_default_prefix()
            self.prefix_var.set(self.prefix)
        self._refresh_export_settings_ui()
        self._save_export_settings()

    def _on_video_scale_change(self, _value=None):
        if self._updating_scale_ui:
            return
        self.video_scale = int(self.video_scale_var.get())
        self.video_scale = max(40, min(80, self.video_scale))
        self.video_scale_var.set(self.video_scale)
        self.video_scale_label.config(text=f"{self.video_scale}%")
        self._save_export_settings()
        if self.task_data is not None:
            self._update_video_display_size()
            self._seek_to(self.cur_frame)

    def _update_video_display_size(self):
        td = self.task_data
        if td is None or td.frame_w <= 0 or td.frame_h <= 0:
            return
        base_w = min(DISPLAY_W, td.frame_w)
        scale = self.video_scale / 100.0
        self.disp_w = max(220, int(base_w * scale))
        self.disp_h = int(self.disp_w * td.frame_h / td.frame_w)
        self.canvas.config(width=self.disp_w, height=self.disp_h)

    def _load_task(self, idx: int):
        self._stop_play()

        if self.task_data is not None:
            self.task_data.release()

        self.task_idx = idx
        task = self.task_names[idx]
        self.task_var.set(task)
        self.task_name_label.config(text=task)
        self.prog_label.config(text=f"{idx + 1} / {len(self.task_names)}")
        self._refresh_task_sidebar()
        self._update_task_metadata_panel(task)

        td = TaskData(PREPROC_DIR, task)
        td.load()
        self.task_data = td

        self._update_video_display_size()

        self.seek_bar.config(to=max(0, td.total_frames - 1))

        if task not in self.segments:
            self.segments[task] = []
        self._sync_annotations_from_segments(task)
        self._ensure_user_gaze_offset(task)
        self._set_gaze_offset_ui(task)
        self._segment_start = None
        self._selected_segment_idx = None

        self.cur_frame = 0
        self._update_plot_data()
        self._seek_to(0)
        self._update_kf_list()
        self._update_done_label()
        self._status(
            f"Loaded: {task}  ({td.total_frames} frames, {td.fps:.0f} fps)"
        )

    def _next_task(self):
        if self.task_idx < len(self.task_names) - 1:
            self._switch_task(self.task_idx + 1)

    def _prev_task(self):
        if self.task_idx > 0:
            self._switch_task(self.task_idx - 1)

    # ═══════════════════════════════════════════════════════════════════════
    #  PLAYBACK
    # ═══════════════════════════════════════════════════════════════════════
    def _get_ffplay_path(self) -> str | None:
        if self._ffplay_path is None:
            self._ffplay_path = shutil.which("ffplay") or ""
        return self._ffplay_path or None

    def _start_audio_playback(self, start_sec: float):
        ffplay = self._get_ffplay_path()
        if ffplay is None:
            if not self._warned_no_ffplay:
                self._warned_no_ffplay = True
                self._status("ffplay not found: video will play without audio")
            return
        if not self.task_names:
            return

        task = self.task_names[self.task_idx]
        audio_path = os.path.join(PREPROC_DIR, f"{task}_audio.wav")
        if not os.path.exists(audio_path):
            return

        self._stop_audio_playback()
        start_sec = max(0.0, float(start_sec))
        try:
            self._audio_proc = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-ss",
                    f"{start_sec:.3f}",
                    "-i",
                    audio_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._audio_proc = None
            self._status(f"Audio playback error: {e}")

    def _stop_audio_playback(self):
        if self._audio_proc is None:
            return
        try:
            if self._audio_proc.poll() is None:
                self._audio_proc.terminate()
                try:
                    self._audio_proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    self._audio_proc.kill()
                    self._audio_proc.wait(timeout=0.2)
        except Exception:
            pass
        finally:
            self._audio_proc = None

    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if self.playing:
            return
        td = self.task_data
        if td is None:
            return
        self.playing = True
        self._start_audio_playback(self.cur_frame / td.fps if td.fps > 0 else 0.0)
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _stop_play(self):
        self.playing = False
        self._stop_audio_playback()
        if self._play_thread:
            self._play_thread.join(timeout=0.5)
            self._play_thread = None

    def _play_loop(self):
        td = self.task_data
        if td is None:
            return
        interval = 1.0 / td.fps
        while self.playing:
            t0 = time.perf_counter()
            if self.cur_frame >= td.total_frames - 1:
                self.playing = False
                break
            self.root.after(0, self._step_play)
            elapsed = time.perf_counter() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
        try:
            self.root.after(0, self._stop_audio_playback)
        except tk.TclError:
            self._stop_audio_playback()

    def _step_play(self):
        td = self.task_data
        if td and self.cur_frame < td.total_frames - 1:
            self._seek_to(self.cur_frame + 1)

    # ═══════════════════════════════════════════════════════════════════════
    #  FRAME SEEKING / RENDERING
    # ═══════════════════════════════════════════════════════════════════════
    def _step(self, delta: int):
        self._stop_play()
        td = self.task_data
        if td is None:
            return
        new = max(0, min(td.total_frames - 1, self.cur_frame + delta))
        self._seek_to(new)

    def _on_seek(self, value):
        frame = int(float(value))
        if frame != self.cur_frame:
            self._seek_to(frame)
            if self.playing:
                td = self.task_data
                if td is not None and td.fps > 0:
                    self._start_audio_playback(frame / td.fps)

    def _seek_to(self, frame: int):
        td = self.task_data
        if td is None:
            return
        self.cur_frame = frame

        # Read and render frame
        img = td.read_frame(frame)
        if img is None:
            return

        img = cv2.resize(img, (self.disp_w, self.disp_h))

        # ── gaze overlay ──
        if td.gaze is not None and frame < len(td.gaze):
            gx, gy = td.gaze[frame]
            task = self.task_names[self.task_idx]
            off = self._get_effective_gaze_offset(task)
            gx += float(off.get("x", 0.0))
            gy += float(off.get("y", 0.0))
            sx = self.disp_w / td.frame_w if td.frame_w else 1
            sy = self.disp_h / td.frame_h if td.frame_h else 1
            dx = int(np.clip(gx * sx, 0, self.disp_w - 1))
            dy = int(np.clip(gy * sy, 0, self.disp_h - 1))

            # Crosshair
            cross_len = 12
            cv2.line(img, (dx - cross_len, dy), (dx + cross_len, dy),
                     (0, 0, 0), 3)
            cv2.line(img, (dx, dy - cross_len), (dx, dy + cross_len),
                     (0, 0, 0), 3)
            cv2.line(img, (dx - cross_len, dy), (dx + cross_len, dy),
                     (0, 255, 0), 2)
            cv2.line(img, (dx, dy - cross_len), (dx, dy + cross_len),
                     (0, 255, 0), 2)
            cv2.circle(img, (dx, dy), 4, (0, 255, 0), -1)
            cv2.circle(img, (dx, dy), 5, (0, 0, 0), 1)

        # ── transcript overlay ──
        current_text = self._get_current_words(frame / td.fps)
        if current_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(
                current_text, font, font_scale, thickness
            )
            tx = (self.disp_w - tw) // 2
            ty = 30  # near top

            overlay = img.copy()
            cv2.rectangle(overlay,
                          (tx - 8, ty - th - 8),
                          (tx + tw + 8, ty + baseline + 8),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.putText(img, current_text, (tx, ty),
                        font, font_scale, (255, 255, 255), thickness,
                        cv2.LINE_AA)

        # ── label border (color by label index) ──
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        for i, seg in enumerate(segs):
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            if s <= frame <= e:
                hex_color = self.segment_colors[i % len(self.segment_colors)].lstrip("#")
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                cv2.rectangle(img, (8, 8),
                              (self.disp_w - 9, self.disp_h - 9),
                              (b, g, r), 3)
                break

        # ── display ──
        pil_img = Image.fromarray(img)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # ── update widgets ──
        self.seek_var.set(frame)
        self.frame_label.config(text=f"{frame:5d} / {td.total_frames - 1}")
        t = frame / td.fps
        self.time_label.config(text=f"{t:6.2f} s")

        # ── update plot cursor ──
        self._update_plot_cursor(frame)

        # ── highlight keyframe in list ──
        self._highlight_kf_in_list(frame)

    def _get_current_words(self, time_sec: float) -> str:
        td = self.task_data
        if td is None or td.transcript is None:
            return ""
        active = [
            w["word"]
            for w in td.transcript
            if w["start"] <= time_sec <= w["end"]
        ]
        return " ".join(active)

    # ═══════════════════════════════════════════════════════════════════════
    #  LABEL ANNOTATION
    # ═══════════════════════════════════════════════════════════════════════
    def _mark_segment_start(self):
        """Mark label start with [ key."""
        self._segment_start = self.cur_frame
        self._status(f"Label start: frame {self.cur_frame}  (press ] to set end)")

    def _mark_segment_end(self):
        """Mark label end with ] key — creates one range [start, end]."""
        if self._segment_start is None:
            self._status("No label start set — press [ first")
            return

        start = min(self._segment_start, self.cur_frame)
        end = max(self._segment_start, self.cur_frame)
        self._segment_start = None
        self._add_segment(start, end)

    def _add_segment(self, start: int, end: int):
        task = self.task_names[self.task_idx]
        segs = self.segments.setdefault(task, [])
        s = int(min(start, end))
        e = int(max(start, end))
        candidate = {"start": s, "end": e}
        if candidate in segs:
            self._selected_segment_idx = segs.index(candidate)
            self._status(f"Label [{s} — {e}] already exists (selected)")
            self._after_kf_change()
            return
        segs.append(candidate)
        segs.sort(key=lambda x: (x["start"], x["end"]))
        self._selected_segment_idx = segs.index(candidate)
        self._status(
            f"Added label [{s} — {e}] "
            f"(total labels: {len(segs)})"
        )
        self._after_kf_change()

    def _find_segment_idx_containing_frame(self, frame: int) -> int | None:
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        for i, seg in enumerate(segs):
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            if s <= frame <= e:
                return i
        return None

    def _find_point_label_idx(self, frame: int) -> int | None:
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        for i, seg in enumerate(segs):
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            if s == e == frame:
                return i
        return None

    def _select_segment_at_frame(self, frame: int) -> bool:
        idx = self._find_segment_idx_containing_frame(frame)
        if idx is None:
            self._selected_segment_idx = None
            self._status("No label at clicked position")
            self._update_plot_data()
            return False

        task = self.task_names[self.task_idx]
        seg = self.segments[task][idx]
        s = int(seg.get("start", frame))
        e = int(seg.get("end", frame))
        if e < s:
            s, e = e, s
        self._selected_segment_idx = idx
        self._status(f"Selected label [{s} — {e}]  (Del to remove)")
        self._update_plot_data()
        return True

    def _remove_selected_segment(self) -> bool:
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        if not segs:
            self._selected_segment_idx = None
            return False

        idx = self._selected_segment_idx
        if idx is None or not (0 <= idx < len(segs)):
            self._selected_segment_idx = None
            return False

        removed = segs.pop(idx)
        if segs:
            self._selected_segment_idx = min(idx, len(segs) - 1)
        else:
            self._selected_segment_idx = None
        rs = int(removed.get("start", -1))
        re = int(removed.get("end", -1))
        if re < rs:
            rs, re = re, rs
        self._status(f"Removed label [{rs} — {re}]")
        self._after_kf_change()
        return True

    def _remove_selected_label_or_keyframe(self):
        if self._remove_selected_segment():
            return
        if self._remove_keyframe():
            return
        self._status("No removable label at current frame")

    def _toggle_keyframe(self):
        """Toggle a 1-frame label on/off at current frame."""
        task = self.task_names[self.task_idx]
        segs = self.segments.setdefault(task, [])

        idx = self._find_point_label_idx(self.cur_frame)
        if idx is not None:
            removed = segs.pop(idx)
            if segs:
                self._selected_segment_idx = min(idx, len(segs) - 1)
            else:
                self._selected_segment_idx = None
            self._status(f"Removed label [{removed['start']} — {removed['end']}]")
            self._after_kf_change()
            return

        self._add_segment(self.cur_frame, self.cur_frame)

    def _remove_keyframe(self):
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        idx = self._find_point_label_idx(self.cur_frame)
        if idx is not None:
            removed = segs.pop(idx)
            if segs:
                self._selected_segment_idx = min(idx, len(segs) - 1)
            else:
                self._selected_segment_idx = None
            self._status(f"Removed label [{removed['start']} — {removed['end']}]")
            self._after_kf_change()
            return True
        return False

    def _undo_last_segment(self):
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        if not segs:
            return
        removed = segs.pop()
        if segs:
            self._selected_segment_idx = min(
                self._selected_segment_idx or 0,
                len(segs) - 1,
            )
        else:
            self._selected_segment_idx = None
        rs = int(removed.get("start", -1))
        re = int(removed.get("end", -1))
        if re < rs:
            rs, re = re, rs
        self._status(f"Removed label [{rs} — {re}]")
        self._after_kf_change()

    def _after_kf_change(self):
        """Refresh UI after label change."""
        task = self.task_names[self.task_idx]
        self._sync_annotations_from_segments(task)
        self._seek_to(self.cur_frame)
        self._update_kf_list()
        self._update_done_label()
        self._update_plot_data()

    def _update_kf_list(self):
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        segs.sort(key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0))))
        self.kf_listbox.delete(0, tk.END)
        self._label_list_seg_indices = []
        td = self.task_data
        fps = td.fps if td else 15.0
        for seg_idx, seg in enumerate(segs):
            s = int(seg.get("start", -1))
            e = int(seg.get("end", -1))
            if e < s:
                s, e = e, s
            if s < 0:
                continue
            t0 = s / fps
            t1 = (e + 1) / fps
            text = f"  label  {s:5d}-{e:5d}   ({t0:.2f}-{t1:.2f}s)"
            self.kf_listbox.insert(
                tk.END,
                text
            )
            self._label_list_seg_indices.append(seg_idx)
        self._highlight_kf_in_list(self.cur_frame)

    def _highlight_kf_in_list(self, frame: int):
        self._updating_label_list = True
        try:
            self.kf_listbox.selection_clear(0, tk.END)
            if not self._label_list_seg_indices:
                return

            seg_idx = self._selected_segment_idx
            if seg_idx is None:
                seg_idx = self._find_segment_idx_containing_frame(frame)
            if seg_idx is None:
                return

            if seg_idx in self._label_list_seg_indices:
                list_idx = self._label_list_seg_indices.index(seg_idx)
                self.kf_listbox.selection_set(list_idx)
                self.kf_listbox.see(list_idx)
        finally:
            self._updating_label_list = False

    def _on_kf_select(self, _event=None):
        if self._updating_label_list:
            return
        sel = self.kf_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._label_list_seg_indices):
            return

        seg_idx = self._label_list_seg_indices[idx]
        task = self.task_names[self.task_idx]
        segs = self.segments.get(task, [])
        if not (0 <= seg_idx < len(segs)):
            return

        seg = segs[seg_idx]
        s = int(seg.get("start", 0))
        e = int(seg.get("end", 0))
        if e < s:
            s, e = e, s
        self._selected_segment_idx = seg_idx
        self._stop_play()
        self._seek_to(s)
        self._update_plot_data()

    def _update_done_label(self):
        done = sum(1 for t in self.task_names if self.segments.get(t))
        self.done_label.config(
            text=f"done {done}/{len(self.task_names)} tasks"
        )
        self._refresh_task_sidebar()

    # ═══════════════════════════════════════════════════════════════════════
    #  MISC
    # ═══════════════════════════════════════════════════════════════════════
    def _set_gaze_offset_ui(self, task_name: str):
        off = self._get_effective_gaze_offset(task_name)
        x = float(off.get("x", 0.0))
        y = float(off.get("y", 0.0))
        user_name = self._task_user_name(task_name)
        using_task_override = self._has_task_gaze_override(task_name)
        scope_text = (
            "Applies to: this task only"
            if using_task_override
            else f"Applies to: user default ({user_name})"
        )
        self._updating_offset_ui = True
        try:
            self.gaze_offset_x_var.set(x)
            self.gaze_offset_y_var.set(y)
            self.gaze_task_override_var.set(using_task_override)
        finally:
            self._updating_offset_ui = False
        self.gaze_offset_value_label.config(text=f"X {x:+.0f} px | Y {y:+.0f} px")
        self.gaze_offset_scope_label.config(text=scope_text)

    def _on_gaze_override_toggle(self):
        if self._updating_offset_ui or not self.task_names:
            return
        task = self.task_names[self.task_idx]
        if self.gaze_task_override_var.get():
            self._enable_task_gaze_override(task)
        else:
            self._disable_task_gaze_override(task)
        self._save_gaze_offsets()
        self._set_gaze_offset_ui(task)

        if self.task_data is not None:
            self._seek_to(self.cur_frame)

    def _on_gaze_offset_change(self, _value=None):
        if self._updating_offset_ui or not self.task_names:
            return
        task = self.task_names[self.task_idx]
        x = float(self.gaze_offset_x_var.get())
        y = float(self.gaze_offset_y_var.get())
        if self.gaze_task_override_var.get():
            self._enable_task_gaze_override(task)
            self.gaze_offsets[task] = {"x": x, "y": y}
            scope_text = "Applies to: this task only"
        else:
            user_name = self._task_user_name(task)
            self.gaze_user_offsets[user_name] = {"x": x, "y": y}
            scope_text = f"Applies to: user default ({user_name})"
        self.last_gaze_offset = {"x": x, "y": y}
        self.gaze_offset_value_label.config(text=f"X {x:+.0f} px | Y {y:+.0f} px")
        self.gaze_offset_scope_label.config(text=scope_text)
        self._save_gaze_offsets()

        # Re-render current frame immediately so user sees the effect.
        if self.task_data is not None:
            self._seek_to(self.cur_frame)

    def _reset_gaze_offset(self):
        if not self.task_names:
            return
        task = self.task_names[self.task_idx]
        if self.gaze_task_override_var.get():
            self._enable_task_gaze_override(task)
            self.gaze_offsets[task] = {"x": 0.0, "y": 0.0}
        else:
            user_name = self._task_user_name(task)
            self.gaze_user_offsets[user_name] = {"x": 0.0, "y": 0.0}
        self.last_gaze_offset = {"x": 0.0, "y": 0.0}
        self._save_gaze_offsets()
        self._set_gaze_offset_ui(task)
        if self.task_data is not None:
            self._seek_to(self.cur_frame)

    def _status(self, msg: str):
        self.status_var.set(msg)

    def _quit(self):
        self._stop_play()
        self._save_annotations()
        if self.task_data is not None:
            self.task_data.release()
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Keyframe Labeler + Dataset Exporter")
    parser.add_argument(
        "--prefix", default=None,
        help=(
            "Optional export prefix when naming mode is prefix_index. "
            "If omitted, a default is inferred from task metadata."
        ),
    )
    args = parser.parse_args()

    root = tk.Tk()

    # Dark ttk theme
    style = ttk.Style(root)
    for t in ("clam", "alt", "default"):
        if t in style.theme_names():
            style.theme_use(t)
            break
    style.configure("TCombobox", fieldbackground="#2d2d2d",
                    background="#2d2d2d", foreground="#ffffff")
    style.configure("TScale", background="#1e1e1e", troughcolor="#444444")
    style.configure("Vertical.TScrollbar", background="#333333",
                    troughcolor="#1e1e1e", arrowcolor="#888888")
    style.configure(
        "Task.Treeview",
        background="#2b2b2b",
        fieldbackground="#2b2b2b",
        foreground="#dddddd",
        borderwidth=0,
        rowheight=24,
        font=("Courier", 10),
    )
    style.map(
        "Task.Treeview",
        background=[("selected", "#3a506b")],
        foreground=[("selected", "#ffffff")],
    )
    style.configure(
        "Task.Treeview.Heading",
        background="#1f2a38",
        foreground="#d9e7ff",
        relief="flat",
        font=("Helvetica", 10, "bold"),
    )

    LabelerApp(root, prefix=args.prefix)
    root.mainloop()


if __name__ == "__main__":
    main()
