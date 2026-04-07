#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_TASKS = (
    ("double_object", "2f_store_juheon_setup_double_object_pickup_", 10),
    ("pick_place", "2f_store_juheon_setup_single_object_pick_place", 10),
)


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_segments(episode_dir: Path) -> list[Segment]:
    ann_path = episode_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations.json: {ann_path}")

    payload = json.loads(ann_path.read_text())
    annotations = payload.get("annotations", [])
    labels = [ann for ann in annotations if ann.get("kind", "label") == "label"]
    labels.sort(key=lambda ann: (ann["start_sec"], ann["end_sec"]))
    return [Segment(start_sec=ann["start_sec"], end_sec=ann["end_sec"]) for ann in labels]


def load_gaze_frames(episode_dir: Path) -> dict[int, tuple[float, float]]:
    gaze_path = episode_dir / "gaze.json"
    if not gaze_path.exists():
        return {}
    payload = json.loads(gaze_path.read_text())
    frames = payload.get("frames", [])
    return {
        int(entry["frame_idx"]): (float(entry["gaze_x"]), float(entry["gaze_y"]))
        for entry in frames
        if "frame_idx" in entry and "gaze_x" in entry and "gaze_y" in entry
    }


def select_episodes(dataset_dir: Path, prefix: str, limit: int) -> list[Path]:
    candidates: list[Path] = []
    for episode_dir in sorted(dataset_dir.glob(f"{prefix}*"), key=lambda path: natural_key(path.name)):
        if not episode_dir.is_dir():
            continue
        try:
            segments = load_segments(episode_dir)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue
        if len(segments) == 2 and (episode_dir / "video.mp4").exists():
            candidates.append(episode_dir)
        if len(candidates) >= limit:
            break
    return candidates


def load_episodes_from_manifest(dataset_dir: Path, manifest_path: Path) -> list[tuple[str, Path]]:
    manifest = json.loads(manifest_path.read_text())
    episodes: list[tuple[str, Path]] = []
    for item in manifest:
        task_name = item["task"]
        episode_name = item["episode"]
        episode_dir = dataset_dir / episode_name
        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode from manifest not found: {episode_dir}")
        episodes.append((task_name, episode_dir))
    return episodes


def build_filter(segments: list[Segment], fps: int, width: int) -> str:
    chains: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        label = f"v{index}"
        chains.append(
            "[0:v]"
            f"trim=start={segment.start_sec:.6f}:end={segment.end_sec:.6f},"
            "setpts=PTS-STARTPTS,"
            f"fps={fps},"
            f"scale={width}:-1:flags=lanczos[{label}]"
        )
        labels.append(f"[{label}]")
    concat_inputs = "".join(labels)
    chains.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0,split[clip][palette_source]")
    chains.append("[palette_source]palettegen=stats_mode=single[palette]")
    chains.append("[clip][palette]paletteuse=dither=bayer:bayer_scale=3")
    return ";".join(chains)


def build_full_video_filter(fps: int, width: int) -> str:
    return (
        "[0:v]"
        f"fps={fps},"
        f"scale={width}:-1:flags=lanczos,"
        "split[clip][palette_source];"
        "[palette_source]palettegen=stats_mode=single[palette];"
        "[clip][palette]paletteuse=dither=bayer:bayer_scale=3"
    )


def get_video_stream_info(video_path: Path) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    payload = json.loads(subprocess.check_output(cmd))
    stream = payload["streams"][0]
    fps_raw = stream["r_frame_rate"]
    num, den = fps_raw.split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def draw_gaze_array(frame: np.ndarray, gx: float, gy: float) -> None:
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, int(round(gx))))
    y = max(0, min(h - 1, int(round(gy))))
    radius = max(3, int(h / 50))

    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    outline_mask = dist_sq <= (radius + 1) ** 2
    fill_mask = dist_sq <= radius**2
    frame[outline_mask] = (0, 0, 0)
    frame[fill_mask] = (0, 255, 0)


def render_gaze_gif(
    video_path: Path,
    output_path: Path,
    gaze_by_frame: dict[int, tuple[float, float]],
    segments: list[Segment] | None,
    fps: int,
    width: int,
    overwrite: bool,
) -> None:
    src_w, src_h, src_fps = get_video_stream_info(video_path)
    frame_size = src_w * src_h * 3

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{src_w}x{src_h}",
            "-r",
            f"{src_fps:.6f}",
            "-i",
            "-",
            "-filter_complex",
            (
                build_filter(segments=segments, fps=fps, width=width)
                if segments is not None
                else build_full_video_filter(fps=fps, width=width)
            ),
            "-loop",
            "0",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )

    frame_idx = 0
    while True:
        assert decoder.stdout is not None
        chunk = decoder.stdout.read(frame_size)
        if len(chunk) < frame_size:
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
        if frame_idx in gaze_by_frame:
            gx, gy = gaze_by_frame[frame_idx]
            draw_gaze_array(frame, gx, gy)
        assert encoder.stdin is not None
        encoder.stdin.write(frame.tobytes())
        frame_idx += 1

    assert decoder.stdout is not None
    decoder.stdout.close()
    assert encoder.stdin is not None
    encoder.stdin.close()

    decode_rc = decoder.wait()
    encode_rc = encoder.wait()
    if decode_rc != 0:
        raise subprocess.CalledProcessError(decode_rc, decoder.args)
    if encode_rc != 0:
        raise subprocess.CalledProcessError(encode_rc, encoder.args)


def render_labeled_gaze_gif(
    video_path: Path,
    output_path: Path,
    gaze_by_frame: dict[int, tuple[float, float]],
    segments: list[Segment],
    fps: int,
    width: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    src_w, src_h, src_fps = get_video_stream_info(video_path)
    frame_size = src_w * src_h * 3
    resized_height = max(1, int(round(src_h * (width / src_w))))

    segment_ranges = [
        (
            int(round(segment.start_sec * src_fps)),
            int(round(segment.end_sec * src_fps)),
        )
        for segment in segments
    ]
    next_capture_times = [0.0 for _ in segment_ranges]
    captured_frames: list[Image.Image] = []

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    frame_idx = 0
    while True:
        assert decoder.stdout is not None
        chunk = decoder.stdout.read(frame_size)
        if len(chunk) < frame_size:
            break

        for seg_idx, (start_frame, end_frame) in enumerate(segment_ranges):
            if not (start_frame <= frame_idx < end_frame):
                continue

            rel_time = (frame_idx - start_frame) / src_fps
            if rel_time + 1e-9 < next_capture_times[seg_idx]:
                break

            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
            if frame_idx in gaze_by_frame:
                gx, gy = gaze_by_frame[frame_idx]
                draw_gaze_array(frame, gx, gy)

            image = Image.fromarray(frame, mode="RGB")
            resized = image.resize((width, resized_height), Image.Resampling.LANCZOS)
            captured_frames.append(resized)
            next_capture_times[seg_idx] += 1.0 / fps
            break

        frame_idx += 1

    assert decoder.stdout is not None
    decoder.stdout.close()
    decode_rc = decoder.wait()
    if decode_rc != 0:
        raise subprocess.CalledProcessError(decode_rc, decoder.args)
    if not captured_frames:
        raise RuntimeError(f"No frames captured for labeled gaze GIF: {video_path}")

    captured_frames[0].save(
        output_path,
        save_all=True,
        append_images=captured_frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )


def render_gif(
    video_path: Path,
    output_path: Path,
    segments: list[Segment] | None,
    fps: int,
    width: int,
    overwrite: bool,
) -> None:
    filter_complex = (
        build_filter(segments=segments, fps=fps, width=width)
        if segments is not None
        else build_full_video_filter(fps=fps, width=width)
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-filter_complex",
        filter_complex,
        "-loop",
        "0",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GIFs from dataset episodes, either stitched label segments or full videos."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("debug/stitched_labeled_gifs"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--mode", choices=("labeled", "full"), default="labeled")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--gaze-annot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_selected_episodes(args: argparse.Namespace, dataset_dir: Path) -> Iterable[tuple[str, Path]]:
    if args.source_manifest is not None:
        yield from load_episodes_from_manifest(dataset_dir=dataset_dir, manifest_path=args.source_manifest.resolve())
        return

    for task_name, prefix, limit in DEFAULT_TASKS:
        selected = select_episodes(dataset_dir=dataset_dir, prefix=prefix, limit=limit)
        if len(selected) < limit:
            raise RuntimeError(f"Only found {len(selected)} episodes for {task_name}, expected {limit}")
        for episode_dir in selected:
            yield task_name, episode_dir


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for task_name, episode_dir in iter_selected_episodes(args=args, dataset_dir=dataset_dir):
        task_output_dir = output_dir / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)
        segments = load_segments(episode_dir)
        output_path = task_output_dir / f"{episode_dir.name}.gif"
        gaze_by_frame = load_gaze_frames(episode_dir) if args.gaze_annot else {}
        render_input = episode_dir / "video.mp4"
        if args.gaze_annot and args.mode == "labeled" and gaze_by_frame:
            render_labeled_gaze_gif(
                video_path=render_input,
                output_path=output_path,
                gaze_by_frame=gaze_by_frame,
                segments=segments,
                fps=args.fps,
                width=args.width,
                overwrite=args.overwrite,
            )
        elif args.gaze_annot and gaze_by_frame:
            render_gaze_gif(
                video_path=render_input,
                output_path=output_path,
                gaze_by_frame=gaze_by_frame,
                segments=segments if args.mode == "labeled" else None,
                fps=args.fps,
                width=args.width,
                overwrite=args.overwrite,
            )
        else:
            render_gif(
                video_path=render_input,
                output_path=output_path,
                segments=segments if args.mode == "labeled" else None,
                fps=args.fps,
                width=args.width,
                overwrite=args.overwrite,
            )
        manifest.append(
            {
                "task": task_name,
                "episode": episode_dir.name,
                "video_path": str((episode_dir / "video.mp4").resolve()),
                "gif_path": str(output_path),
                "mode": args.mode,
                "gaze_annot": args.gaze_annot,
                "segments": [
                    {"start_sec": segment.start_sec, "end_sec": segment.end_sec}
                    for segment in segments
                ],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Created {len(manifest)} GIFs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
