"""
Data loading utilities for the visualization dashboard.
Handles video, gaze, pitch/yaw, angular velocity, and cached transcripts.
"""

from __future__ import annotations

import os
import json
import re
import cv2
import numpy as np
import soundfile as sf
from scipy.signal import savgol_filter
from pathlib import Path


def compute_angular_velocity(pitch_yaw: np.ndarray, fps: float,
                             savgol_window: int = 11,
                             savgol_poly: int = 2) -> np.ndarray:
    """Compute frame-aligned gaze angular velocity (deg/s) from pitch/yaw.

    Args:
        pitch_yaw: (T, 6) array. Column 0 = pitch, column 3 = yaw (radians).
        fps: frames per second.
    Returns:
        (T,) array of angular velocity in deg/s aligned to each frame.
    """
    pitch = pitch_yaw[:, 0]
    yaw = pitch_yaw[:, 3]

    T = len(pitch)
    if T <= 1 or fps <= 0:
        return np.zeros(T, dtype=np.float64)

    if T < savgol_window:
        savgol_window = max(T | 1, 3)  # ensure odd and >= 3
    if savgol_poly >= savgol_window:
        savgol_poly = savgol_window - 1

    pitch_s = savgol_filter(pitch, savgol_window, savgol_poly)
    yaw_s = savgol_filter(yaw, savgol_window, savgol_poly)

    # Convert to 3D unit vectors on sphere
    x = np.cos(pitch_s) * np.cos(yaw_s)
    y = np.cos(pitch_s) * np.sin(yaw_s)
    z = np.sin(pitch_s)

    # Use centered differences so velocity[i] is aligned to frame i.
    step_dot = np.clip(
        x[:-1] * x[1:] + y[:-1] * y[1:] + z[:-1] * z[1:],
        -1.0,
        1.0,
    )
    step_velocity = np.arccos(step_dot) * fps  # rad/s over 1 frame

    angular_velocity = np.zeros(T, dtype=np.float64)
    angular_velocity[0] = step_velocity[0]
    angular_velocity[-1] = step_velocity[-1]

    if T > 2:
        center_dot = np.clip(
            x[:-2] * x[2:] + y[:-2] * y[2:] + z[:-2] * z[2:],
            -1.0,
            1.0,
        )
        angular_velocity[1:-1] = np.arccos(center_dot) * (fps / 2.0)

    return np.rad2deg(angular_velocity)


def discover_tasks(base_dir: str) -> list[str]:
    """Scan base_dir for *_rgb.mp4 and return sorted task names."""
    tasks = []
    for f in os.listdir(base_dir):
        if f.endswith("_rgb.mp4"):
            task_name = f[: -len("_rgb.mp4")]
            tasks.append(task_name)

    # Natural sort: unsuffixed first, then by numeric suffix
    def sort_key(name: str):
        m = re.search(r"_(\d+)$", name)
        return int(m.group(1)) if m else 0

    tasks.sort(key=sort_key)
    return tasks


class TaskData:
    """All data for a single task, loaded eagerly."""

    def __init__(self, base_dir: str, task_name: str, fps: float = 15.0):
        self.base_dir = base_dir
        self.task_name = task_name
        self.fps = fps

        self.cap: cv2.VideoCapture | None = None
        self.total_frames: int = 0
        self.frame_w: int = 0
        self.frame_h: int = 0

        self.gaze: np.ndarray | None = None          # (T, 2)
        self.pitch_yaw: np.ndarray | None = None      # (T, 6)
        self.angular_velocity_deg: np.ndarray | None = None  # (T,)
        self.time_axis: np.ndarray | None = None       # (T,)
        self.transcript: list[dict] | None = None      # word-level
        self.audio_energy: np.ndarray | None = None    # (T,)

    def _path(self, suffix: str) -> str:
        return os.path.join(self.base_dir, f"{self.task_name}{suffix}")

    def load(self):
        """Load all data for this task."""
        # Video
        video_path = self._path("_rgb.mp4")
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps_from_video = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_from_video > 0:
            self.fps = fps_from_video

        # Gaze
        gaze_path = self._path("_gaze.npz")
        if os.path.exists(gaze_path):
            self.gaze = np.load(gaze_path)["gaze"]  # (T, 2)

        # Pitch / yaw
        py_path = self._path("_pitch_yaw.npz")
        if os.path.exists(py_path):
            self.pitch_yaw = np.load(py_path)["pitch_yaw"]  # (T, 6)
            self.angular_velocity_deg = compute_angular_velocity(
                self.pitch_yaw, self.fps
            )

        # Time axis
        T = self.total_frames
        self.time_axis = np.arange(T) / self.fps

        # Transcript (cached JSON)
        transcript_path = os.path.join(
            self.base_dir, "transcripts", f"{self.task_name}_transcript.json"
        )
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                data = json.load(f)
            self.transcript = data.get("words", [])

        # Audio energy envelope (aligned to frames)
        audio_path = self._path("_audio.wav")
        if os.path.exists(audio_path) and self.total_frames > 0 and self.fps > 0:
            try:
                audio, sr = sf.read(audio_path)
                if audio.ndim == 2:
                    # Mix down channels for a single envelope.
                    audio = np.mean(audio, axis=1)
                audio = np.abs(audio.astype(np.float32))

                samples_per_frame = sr / self.fps
                envelope = np.zeros(self.total_frames, dtype=np.float32)
                n = len(audio)
                for i in range(self.total_frames):
                    s = int(i * samples_per_frame)
                    e = int((i + 1) * samples_per_frame)
                    if s >= n:
                        break
                    e = min(e, n)
                    if e > s:
                        envelope[i] = float(np.mean(audio[s:e]))

                max_v = float(envelope.max()) if envelope.size else 0.0
                if max_v > 1e-8:
                    envelope /= max_v
                self.audio_energy = envelope
            except Exception:
                self.audio_energy = None

    def read_frame(self, idx: int) -> np.ndarray | None:
        """Read a specific frame (returns RGB numpy array or None)."""
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
