#!/usr/bin/env python3
"""
Keyframe Labeler + Dataset Exporter
====================================
Interactive Tkinter GUI for:
  - RGB video playback with live gaze overlay
  - Whisper transcript overlay (word-level)
  - Angular velocity plot (scrolling line with cursor)
  - Binary keyframe annotation (0/1)
  - Automatic dataset export on save

Controls:
  Right / D / L           : Next frame
  Left  / A / H           : Prev frame
  Shift+Right / Shift+D   : +10 frames
  Shift+Left  / Shift+A   : -10 frames
  Ctrl+Right              : +100 frames
  Ctrl+Left               : -100 frames
  Space                   : Toggle keyframe (on/off)
  Delete / Backspace      : Remove keyframe
  P                       : Play / Pause
  [ / ]                   : Prev / Next task
  S                       : Save annotations + export dataset
  Q / Escape              : Quit (saves first)
"""

from __future__ import annotations

import os
import json
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
DISPLAY_W = 640  # display width (aspect ratio preserved)


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
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Keyframe Labeler")
        self.root.configure(bg="#1e1e1e")

        # ── state ──
        self.task_names = discover_tasks(PREPROC_DIR)
        self.task_idx = 0
        self.task_data: TaskData | None = None
        self.annotations: dict[str, list[dict]] = {}  # {task: [{frame: int}]}
        self.playing = False
        self._play_thread: threading.Thread | None = None
        self._photo = None
        self.cur_frame = 0
        self.disp_w = DISPLAY_W
        self.disp_h = DISPLAY_W  # updated on task load

        # ── load saved annotations ──
        self._load_annotations()

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

    def _save_annotations(self):
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as f:
            json.dump(self.annotations, f, indent=2)
        self._export_dataset()
        self._status(f"Saved + exported to {DATASET_DIR}")

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
    def _export_dataset(self):
        """Export labeled episodes to dataset/ directory."""
        os.makedirs(DATASET_DIR, exist_ok=True)

        for task_name, kfs in self.annotations.items():
            episode_dir = os.path.join(DATASET_DIR, task_name)
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

            # 4. labels.npy
            self._export_labels_npy(task_name, episode_dir, kfs)

    def _export_gaze_json(self, task_name: str, episode_dir: str):
        """Convert gaze.npz + pitch_yaw.npz to gaze.json."""
        gaze_path = os.path.join(PREPROC_DIR, f"{task_name}_gaze.npz")
        py_path = os.path.join(PREPROC_DIR, f"{task_name}_pitch_yaw.npz")

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
                entry["gaze_x"] = float(gaze_data[i, 0])
                entry["gaze_y"] = float(gaze_data[i, 1])
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

    def _export_labels_npy(self, task_name: str, episode_dir: str,
                           kfs: list[dict]):
        """Create T-length binary label array from keyframe annotations."""
        video_path = os.path.join(PREPROC_DIR, f"{task_name}_rgb.mp4")
        if not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        labels = np.zeros(T, dtype=np.int8)
        for kf in kfs:
            idx = kf["frame"]
            if 0 <= idx < T:
                labels[idx] = 1

        np.save(os.path.join(episode_dir, "labels.npy"), labels)

    # ═══════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── top bar ──
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(fill=tk.X, padx=8, pady=(8, 2))

        tk.Label(top, text="Task:", bg="#1e1e1e", fg="#cccccc",
                 font=("Helvetica", 12)).pack(side=tk.LEFT)

        self.task_var = tk.StringVar()
        self.task_combo = ttk.Combobox(
            top, textvariable=self.task_var,
            values=self.task_names, state="readonly", width=32,
            font=("Helvetica", 12),
        )
        self.task_combo.pack(side=tk.LEFT, padx=6)
        self.task_combo.bind("<<ComboboxSelected>>", self._on_combo_select)

        self.prog_label = tk.Label(top, text="", bg="#1e1e1e", fg="#888888",
                                   font=("Helvetica", 11))
        self.prog_label.pack(side=tk.LEFT, padx=10)

        self.done_label = tk.Label(top, text="", bg="#1e1e1e", fg="#4ec94e",
                                   font=("Helvetica", 11, "bold"))
        self.done_label.pack(side=tk.RIGHT, padx=6)

        # ── video canvas ──
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=8, pady=4)

        # ── seek bar ──
        seek_frame = tk.Frame(self.root, bg="#1e1e1e")
        seek_frame.pack(fill=tk.X, padx=8, pady=2)

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
        self.plot_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.plot_frame.pack(fill=tk.X, padx=8, pady=4)

        # ── keyframe listbox ──
        kf_frame = tk.Frame(self.root, bg="#1e1e1e")
        kf_frame.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(kf_frame, text="Keyframes:", bg="#1e1e1e", fg="#aaaaaa",
                 font=("Helvetica", 11)).pack(side=tk.LEFT)

        self.kf_listbox = tk.Listbox(
            kf_frame, height=3, bg="#2d2d2d", fg="#f0c040",
            selectbackground="#555555", font=("Courier", 11),
            relief=tk.FLAT, borderwidth=0, activestyle="none",
        )
        self.kf_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.kf_listbox.bind("<<ListboxSelect>>", self._on_kf_select)

        scrollbar = ttk.Scrollbar(kf_frame, orient=tk.VERTICAL,
                                  command=self.kf_listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.kf_listbox.configure(yscrollcommand=scrollbar.set)

        # ── shortcut help ──
        help_frame = tk.Frame(self.root, bg="#1e1e1e")
        help_frame.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(
            help_frame,
            text="Space=toggle KF  Del=remove  P=play  [/]=tasks  S=save+export",
            bg="#1e1e1e", fg="#666666", font=("Courier", 9),
        ).pack(side=tk.LEFT)

        # ── button row ──
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(fill=tk.X, padx=8, pady=(2, 8))

        btn_cfg = dict(
            bg="#333333", fg="#ffffff", relief=tk.FLAT,
            font=("Helvetica", 11), padx=10, pady=4,
            activebackground="#555555", activeforeground="#ffffff",
            cursor="hand2",
        )

        tk.Button(btn_frame, text="<< Prev [", command=self._prev_task,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text=">> Next ]", command=self._next_task,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Play/Pause P", command=self._toggle_play,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Toggle KF Space",
                  command=self._toggle_keyframe,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Remove Del", command=self._remove_keyframe,
                  **btn_cfg).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Save+Export S",
                  bg="#1a4a1a", fg="#4ec94e", relief=tk.FLAT,
                  font=("Helvetica", 11, "bold"), padx=10, pady=4,
                  activebackground="#2a6a2a", activeforeground="#4ec94e",
                  cursor="hand2", command=self._save_annotations).pack(
            side=tk.RIGHT, padx=3)

        # ── status bar ──
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#111111", fg="#888888", font=("Helvetica", 10),
                 anchor=tk.W, padx=6).pack(fill=tk.X, side=tk.BOTTOM)

    # ═══════════════════════════════════════════════════════════════════════
    #  MATPLOTLIB PLOT
    # ═══════════════════════════════════════════════════════════════════════
    def _init_plot(self):
        self.fig = Figure(figsize=(6.4, 1.5), dpi=100, facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#2d2d2d")
        self.ax.tick_params(colors="#888888", labelsize=8)
        self.ax.set_ylabel("Ang Vel (deg/s)", color="#aaaaaa", fontsize=9)
        self.ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#555555")

        self.vel_line, = self.ax.plot([], [], color="#4ec9ff", linewidth=1.0)
        self.cursor_line = self.ax.axvline(x=0, color="#ff4444", linewidth=1.5)
        self.kf_plot_lines: list = []

        self.fig.tight_layout(pad=0.5)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.X, expand=False)
        self._plot_bg = None  # for blitting

    def _update_plot_data(self):
        td = self.task_data
        if td is None or td.angular_velocity_deg is None:
            return

        self.vel_line.set_data(td.time_axis, td.angular_velocity_deg)
        self.ax.set_xlim(0, td.time_axis[-1] if len(td.time_axis) else 1)
        max_vel = td.angular_velocity_deg.max()
        self.ax.set_ylim(0, max(max_vel * 1.1, 1.0))

        # Remove old keyframe markers
        for line in self.kf_plot_lines:
            line.remove()
        self.kf_plot_lines.clear()

        # Add keyframe markers
        task = self.task_names[self.task_idx]
        for kf in self.annotations.get(task, []):
            t = kf["frame"] / td.fps
            vl = self.ax.axvline(x=t, color="#f0c040", linewidth=0.8,
                                 linestyle="--", alpha=0.7)
            self.kf_plot_lines.append(vl)

        self.cursor_line.set_xdata([0, 0])
        self.plot_canvas.draw()

        # Cache background for blitting
        self._plot_bg = self.plot_canvas.copy_from_bbox(self.ax.bbox)

    def _update_plot_cursor(self, frame_idx: int):
        if self.task_data is None:
            return
        t = frame_idx / self.task_data.fps

        if self._plot_bg is not None:
            self.plot_canvas.restore_region(self._plot_bg)
            self.cursor_line.set_xdata([t, t])
            self.ax.draw_artist(self.cursor_line)
            self.plot_canvas.blit(self.ax.bbox)
        else:
            self.cursor_line.set_xdata([t, t])
            self.plot_canvas.draw_idle()

    # ═══════════════════════════════════════════════════════════════════════
    #  KEY BINDINGS
    # ═══════════════════════════════════════════════════════════════════════
    def _bind_keys(self):
        r = self.root
        r.bind("<Right>", lambda e: self._step(1))
        r.bind("<Left>", lambda e: self._step(-1))
        r.bind("d", lambda e: self._step(1))
        r.bind("a", lambda e: self._step(-1))
        r.bind("l", lambda e: self._step(1))
        r.bind("h", lambda e: self._step(-1))
        r.bind("<Shift-Right>", lambda e: self._step(10))
        r.bind("<Shift-Left>", lambda e: self._step(-10))
        r.bind("<Shift-D>", lambda e: self._step(10))
        r.bind("<Shift-A>", lambda e: self._step(-10))
        r.bind("<Control-Right>", lambda e: self._step(100))
        r.bind("<Control-Left>", lambda e: self._step(-100))
        r.bind("<space>", lambda e: self._toggle_keyframe())
        r.bind("<Delete>", lambda e: self._remove_keyframe())
        r.bind("<BackSpace>", lambda e: self._remove_keyframe())
        r.bind("p", lambda e: self._toggle_play())
        r.bind("P", lambda e: self._toggle_play())
        r.bind("]", lambda e: self._next_task())
        r.bind("[", lambda e: self._prev_task())
        r.bind("s", lambda e: self._save_annotations())
        r.bind("S", lambda e: self._save_annotations())
        r.bind("q", lambda e: self._quit())
        r.bind("<Escape>", lambda e: self._quit())
        r.protocol("WM_DELETE_WINDOW", self._quit)

    # ═══════════════════════════════════════════════════════════════════════
    #  TASK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    def _load_task(self, idx: int):
        self._stop_play()

        if self.task_data is not None:
            self.task_data.release()

        self.task_idx = idx
        task = self.task_names[idx]
        self.task_var.set(task)
        self.prog_label.config(text=f"{idx + 1} / {len(self.task_names)}")

        td = TaskData(PREPROC_DIR, task)
        td.load()
        self.task_data = td

        # Compute display dimensions (keep aspect ratio)
        if td.frame_w > 0 and td.frame_h > 0:
            self.disp_w = min(DISPLAY_W, td.frame_w)
            self.disp_h = int(self.disp_w * td.frame_h / td.frame_w)
        self.canvas.config(width=self.disp_w, height=self.disp_h)

        self.seek_bar.config(to=max(0, td.total_frames - 1))

        if task not in self.annotations:
            self.annotations[task] = []

        self.cur_frame = 0
        self._update_plot_data()
        self._seek_to(0)
        self._update_kf_list()
        self._update_done_label()
        self._status(
            f"Loaded: {task}  ({td.total_frames} frames, {td.fps:.0f} fps)"
        )

    def _on_combo_select(self, _event=None):
        new_task = self.task_var.get()
        if new_task in self.task_names:
            new_idx = self.task_names.index(new_task)
            if new_idx != self.task_idx:
                self._save_annotations()
                self._load_task(new_idx)

    def _next_task(self):
        if self.task_idx < len(self.task_names) - 1:
            self._save_annotations()
            self._load_task(self.task_idx + 1)

    def _prev_task(self):
        if self.task_idx > 0:
            self._save_annotations()
            self._load_task(self.task_idx - 1)

    # ═══════════════════════════════════════════════════════════════════════
    #  PLAYBACK
    # ═══════════════════════════════════════════════════════════════════════
    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.playing = True
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._play_thread.start()

    def _stop_play(self):
        self.playing = False
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

        # ── keyframe border ──
        task = self.task_names[self.task_idx]
        kf_frames = [kf["frame"] for kf in self.annotations.get(task, [])]
        if frame in kf_frames:
            cv2.rectangle(img, (0, 0),
                          (self.disp_w - 1, self.disp_h - 1),
                          (255, 220, 0), 6)

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
    #  KEYFRAME ANNOTATION (BINARY)
    # ═══════════════════════════════════════════════════════════════════════
    def _toggle_keyframe(self):
        """Toggle keyframe on/off at current frame."""
        task = self.task_names[self.task_idx]
        kfs = self.annotations.setdefault(task, [])

        existing = [kf for kf in kfs if kf["frame"] == self.cur_frame]
        if existing:
            kfs.remove(existing[0])
            self._status(f"Removed keyframe at frame {self.cur_frame}")
        else:
            kfs.append({"frame": self.cur_frame})
            kfs.sort(key=lambda x: x["frame"])
            self._status(
                f"Added keyframe at frame {self.cur_frame}  "
                f"(total: {len(kfs)})"
            )
        self._after_kf_change()

    def _remove_keyframe(self):
        task = self.task_names[self.task_idx]
        kfs = self.annotations.get(task, [])
        existing = [kf for kf in kfs if kf["frame"] == self.cur_frame]
        if existing:
            kfs.remove(existing[0])
            self._status(f"Removed keyframe at frame {self.cur_frame}")
            self._after_kf_change()

    def _after_kf_change(self):
        """Refresh UI after keyframe change."""
        self._seek_to(self.cur_frame)
        self._update_kf_list()
        self._update_done_label()
        self._update_plot_data()

    def _update_kf_list(self):
        task = self.task_names[self.task_idx]
        kfs = self.annotations.get(task, [])
        self.kf_listbox.delete(0, tk.END)
        td = self.task_data
        fps = td.fps if td else 15.0
        for kf in kfs:
            t = kf["frame"] / fps
            self.kf_listbox.insert(
                tk.END,
                f"  frame {kf['frame']:5d}   ({t:.2f}s)"
            )
        self._highlight_kf_in_list(self.cur_frame)

    def _highlight_kf_in_list(self, frame: int):
        task = self.task_names[self.task_idx]
        kfs = self.annotations.get(task, [])
        frames = [kf["frame"] for kf in kfs]
        if frame in frames:
            idx = frames.index(frame)
            self.kf_listbox.selection_clear(0, tk.END)
            self.kf_listbox.selection_set(idx)
            self.kf_listbox.see(idx)

    def _on_kf_select(self, _event=None):
        sel = self.kf_listbox.curselection()
        if not sel:
            return
        task = self.task_names[self.task_idx]
        kfs = self.annotations.get(task, [])
        idx = sel[0]
        if idx < len(kfs):
            self._stop_play()
            self._seek_to(kfs[idx]["frame"])

    def _update_done_label(self):
        done = sum(1 for t in self.task_names if self.annotations.get(t))
        self.done_label.config(
            text=f"done {done}/{len(self.task_names)} tasks"
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  MISC
    # ═══════════════════════════════════════════════════════════════════════
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

    LabelerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
