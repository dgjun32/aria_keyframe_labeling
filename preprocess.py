#!/usr/bin/env python3
"""
preprocess.py -- Convert VRS files to preprocessed modalities.

Usage:
    python preprocess.py [--vrs-dir VRS_DIR] [--vrs-file VRS_FILE] [--out-dir OUT_DIR]

Processes all .vrs files in VRS_DIR (default: ./vrs_files/).
If --vrs-file is provided, process only that single VRS file.
Outputs go to OUT_DIR (default: ./preproc_files/).
Already-processed files are skipped (idempotent).

Outputs per VRS file:
    {name}_rgb.mp4           256x256 @ 15fps
    {name}_rgb_with_gaze.mp4 256x256 @ 15fps with gaze overlay
    {name}_gaze.npz          (T, 2) gaze pixel coordinates
    {name}_pitch_yaw.npz     (T, 6) [pitch, pitch_lower, pitch_upper, yaw, yaw_lower, yaw_upper]
    {name}_audio.wav         multi-channel 48kHz audio
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

from eye_gaze import EyeGazeInferenceBatch, AriaGazeProjector

# ── Constants ──────────────────────────────────────────────────────────────
RGB_STREAM = StreamId("214-1")
EYE_STREAM = StreamId("211-1")
TARGET_SIZE = 1408
OUTPUT_FPS = 15
OUTPUT_SUFFIXES = [
    "_rgb.mp4", "_rgb_with_gaze.mp4", "_gaze.npz", "_pitch_yaw.npz", "_audio.wav"
]


# ── Helpers ────────────────────────────────────────────────────────────────
def rotate_coordinate(coordinate, image_width, image_height, rotation="cw_90"):
    """Rotate a 2D coordinate according to image rotation."""
    x, y = coordinate
    if rotation == "cw_90":
        x_new = image_height - 1 - y
        y_new = x
    elif rotation == "cw_180":
        x_new = image_width - 1 - x
        y_new = image_height - 1 - y
    elif rotation in ("cw_270", "ccw_90"):
        x_new = y
        y_new = image_width - 1 - x
    else:
        x_new, y_new = x, y
    return np.array([x_new, y_new])


def scale_coordinate(coordinate, scale_factor):
    """Scale a 2D coordinate by a uniform factor."""
    return coordinate * scale_factor


def is_already_processed(basename: str, out_dir: str) -> bool:
    """Check if all 5 output files exist for this basename."""
    return all(
        os.path.exists(os.path.join(out_dir, f"{basename}{s}"))
        for s in OUTPUT_SUFFIXES
    )


def _extract_audio(provider, out_path: str) -> None:
    """Extract multi-channel audio from VRS and save as WAV."""
    try:
        stream_id = provider.get_stream_id_from_label("mic")
        num_blocks = provider.get_num_data(stream_id)
        all_channels = None

        for index in range(num_blocks):
            audio_record, _ = provider.get_audio_data_by_index(stream_id, index)
            audio_block = np.array(audio_record.data, dtype=np.int32)

            if len(audio_block) % 7 == 0:
                num_channels = 7
            elif len(audio_block) % 2 == 0:
                num_channels = 2
            else:
                raise RuntimeError("Cannot determine number of channels")

            if all_channels is None:
                all_channels = [[] for _ in range(num_channels)]

            for c in range(num_channels):
                all_channels[c].extend(audio_block[c::num_channels])

        audio_array = np.stack(
            [np.array(ch, dtype=np.int32) for ch in all_channels], axis=1
        )
        sf.write(out_path, audio_array, samplerate=48000)
        print(f"  [audio] {audio_array.shape[0]} samples, {audio_array.shape[1]} channels")
    except Exception as e:
        print(f"  [WARNING] Audio extraction failed: {e}")


# ── Main processing ───────────────────────────────────────────────────────
def process_vrs(
    vrs_path: str, out_dir: str, gaze_model: EyeGazeInferenceBatch
) -> None:
    """Process a single VRS file into preprocessed outputs."""
    basename = os.path.splitext(os.path.basename(vrs_path))[0]

    if is_already_processed(basename, out_dir):
        print(f"[SKIP] {basename} -- already processed")
        return

    print(f"\n{'=' * 50}")
    print(f"[PROCESSING] {basename}")
    print(f"{'=' * 50}")

    provider = data_provider.create_vrs_data_provider(vrs_path)

    num_rgb = provider.get_num_data(RGB_STREAM)
    num_eye = provider.get_num_data(EYE_STREAM)
    num_frames = min(num_rgb, num_eye)

    print(f"  RGB frames: {num_rgb}, Eye frames: {num_eye}")

    gaze_projector = AriaGazeProjector(vrs_path=vrs_path)

    # Get original frame dimensions (after rotation)
    first_rgb = provider.get_image_data_by_index(RGB_STREAM, 0)[0].to_numpy_array()
    first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
    first_rgb = cv2.rotate(first_rgb, cv2.ROTATE_90_CLOCKWISE)
    height, width = first_rgb.shape[:2]

    # Video writers (256x256 @ 15fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    rgb_writer = cv2.VideoWriter(
        os.path.join(out_dir, f"{basename}_rgb.mp4"),
        fourcc, OUTPUT_FPS, (TARGET_SIZE, TARGET_SIZE),
    )
    overlay_writer = cv2.VideoWriter(
        os.path.join(out_dir, f"{basename}_rgb_with_gaze.mp4"),
        fourcc, OUTPUT_FPS, (TARGET_SIZE, TARGET_SIZE),
    )

    gaze_coords = []
    pitch_yaws = []

    scale_factor = TARGET_SIZE / width

    # Frame loop: every 2nd frame for 15fps from 30fps native
    for idx in range(0, num_frames, 2):
        # ── RGB ──
        rgb = provider.get_image_data_by_index(RGB_STREAM, idx)[0].to_numpy_array()
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgb_rot = cv2.rotate(rgb_bgr, cv2.ROTATE_90_CLOCKWISE)
        rgb_resized = cv2.resize(rgb_rot, (TARGET_SIZE, TARGET_SIZE))
        rgb_writer.write(rgb_resized)

        # ── Eye tracking ──
        eye_img = provider.get_image_data_by_index(EYE_STREAM, idx)[0].to_numpy_array()
        results = gaze_model.predict(eye_img[None, ...])

        pitch_yaws.append(np.array([
            results["pitch"][0], results["pitch_lower"][0], results["pitch_upper"][0],
            results["yaw"][0], results["yaw_lower"][0], results["yaw_upper"][0],
        ]))

        # ── Gaze projection ──
        unscaled_coords, _ = gaze_projector.project_with_simple_fallback(
            yaw_array=results["yaw"],
            pitch_array=results["pitch"],
            image_width=width,
            image_height=height,
            depth_m=1.0,
        )

        gx, gy = unscaled_coords[0]
        coord = rotate_coordinate(np.array([int(gx), int(gy)]), width, height)
        coord = scale_coordinate(coord, scale_factor=scale_factor)
        gaze_coords.append([coord[0], coord[1]])

        # ── Overlay frame ──
        overlay = cv2.resize(rgb_rot, (TARGET_SIZE, TARGET_SIZE))
        cv2.circle(overlay, (int(coord[0]), int(coord[1])), 4, (0, 0, 255), -1)
        overlay_writer.write(overlay)

        if idx % 100 == 0:
            print(f"  frame {idx}/{num_frames}")

    rgb_writer.release()
    overlay_writer.release()

    # Save gaze and pitch/yaw
    np.savez(
        os.path.join(out_dir, f"{basename}_gaze.npz"),
        gaze=np.array(gaze_coords),
    )
    np.savez(
        os.path.join(out_dir, f"{basename}_pitch_yaw.npz"),
        pitch_yaw=np.array(pitch_yaws),
    )
    print(f"  [gaze] {len(gaze_coords)} frames saved")

    # Audio
    _extract_audio(provider, os.path.join(out_dir, f"{basename}_audio.wav"))

    print(f"[DONE] {basename}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VRS files into modality-separated outputs."
    )
    parser.add_argument(
        "--vrs-dir", default="./vrs_files",
        help="Directory containing .vrs files (default: ./vrs_files)",
    )
    parser.add_argument(
        "--out-dir", default="./preproc_files",
        help="Output directory (default: ./preproc_files)",
    )
    parser.add_argument(
        "--vrs-file", default=None,
        help=(
            "Process only one VRS file. Accepts: absolute path, relative path, "
            "or filename inside --vrs-dir."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.vrs_file:
        candidate_paths = [
            args.vrs_file,
            os.path.join(args.vrs_dir, args.vrs_file),
        ]
        resolved_vrs = next((p for p in candidate_paths if os.path.exists(p)), None)
        if resolved_vrs is None:
            print(
                f"[ERROR] --vrs-file not found: {args.vrs_file} "
                f"(also checked under {args.vrs_dir})"
            )
            sys.exit(1)
        if not resolved_vrs.endswith(".vrs"):
            print(f"[ERROR] --vrs-file must point to a .vrs file: {resolved_vrs}")
            sys.exit(1)
        vrs_files = [resolved_vrs]
        print(f"Using single VRS file: {resolved_vrs}")
    else:
        vrs_files = sorted(glob.glob(os.path.join(args.vrs_dir, "*.vrs")))
        if not vrs_files:
            print(f"No .vrs files found in {args.vrs_dir}")
            sys.exit(0)
        print(f"Found {len(vrs_files)} VRS file(s) in {args.vrs_dir}")

    # Load gaze model ONCE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading gaze model on {device}...")
    gaze_model = EyeGazeInferenceBatch(device=device)

    for vrs_path in tqdm(vrs_files, desc="Processing VRS"):
        try:
            process_vrs(vrs_path, args.out_dir, gaze_model)
        except Exception as e:
            print(f"[ERROR] Failed to process {vrs_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
