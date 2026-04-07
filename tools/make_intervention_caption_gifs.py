#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_COLORS = [
    (137, 239, 93),
    (71, 211, 255),
    (255, 179, 71),
    (255, 105, 180),
    (255, 99, 71),
    (186, 104, 200),
]


@dataclass(frozen=True)
class LabelRange:
    label_id: int
    start_frame: int
    end_frame_exclusive: int
    color: tuple[int, int, int]


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def get_video_stream_info(video_path: Path) -> tuple[int, int, float]:
    payload = json.loads(
        subprocess.check_output(
            [
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
        )
    )
    stream = payload["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def load_gaze_frames(episode_dir: Path) -> dict[int, tuple[float, float]]:
    payload = json.loads((episode_dir / "gaze.json").read_text())
    return {
        int(entry["frame_idx"]): (float(entry["gaze_x"]), float(entry["gaze_y"]))
        for entry in payload.get("frames", [])
        if "frame_idx" in entry and "gaze_x" in entry and "gaze_y" in entry
    }


def load_words(episode_dir: Path) -> list[dict]:
    payload = json.loads((episode_dir / "transcript.json").read_text())
    return payload.get("words", [])


def load_label_ranges(episode_dir: Path) -> list[LabelRange]:
    payload = json.loads((episode_dir / "annotations.json").read_text())
    labels = [ann for ann in payload.get("annotations", []) if ann.get("kind", "label") == "label"]
    labels.sort(key=lambda ann: (ann["start_frame"], ann["end_frame"]))
    return [
        LabelRange(
            label_id=index + 1,
            start_frame=int(label["start_frame"]),
            end_frame_exclusive=int(label["end_frame"]) + 1,
            color=LABEL_COLORS[index % len(LABEL_COLORS)],
        )
        for index, label in enumerate(labels)
    ]


def get_active_caption(words: list[dict], time_sec: float) -> str:
    active = [word["word"] for word in words if word["start"] <= time_sec <= word["end"]]
    return " ".join(active)


def get_active_label(label_ranges: list[LabelRange], frame_idx: int) -> LabelRange | None:
    for label_range in label_ranges:
        if label_range.start_frame <= frame_idx < label_range.end_frame_exclusive:
            return label_range
    return None


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_border(draw: ImageDraw.ImageDraw, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    width, height = size
    thickness = max(4, height // 75)
    for offset in range(thickness):
        draw.rectangle(
            (offset, offset, width - 1 - offset, height - 1 - offset),
            outline=color,
            width=1,
        )


def draw_caption(draw: ImageDraw.ImageDraw, size: tuple[int, int], text: str) -> None:
    if not text:
        return
    width, height = size
    font_size = max(18, height // 18)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(8, width // 60)
    pad_y = max(6, height // 90)
    x0 = max(0, (width - text_w) // 2 - pad_x)
    y0 = max(0, height // 28 - pad_y)
    x1 = min(width, x0 + text_w + 2 * pad_x)
    y1 = min(height, y0 + text_h + 2 * pad_y)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=max(6, height // 60), fill=(0, 0, 0, 165))
    draw.text((x0 + pad_x, y0 + pad_y - 1), text, font=font, fill=(255, 255, 255, 255))


def draw_gaze(draw: ImageDraw.ImageDraw, size: tuple[int, int], gaze_xy: tuple[float, float]) -> None:
    width, height = size
    x = max(0, min(width - 1, int(round(gaze_xy[0]))))
    y = max(0, min(height - 1, int(round(gaze_xy[1]))))
    radius = max(4, height // 60)
    arm = radius * 2
    outline = radius + 2

    draw.line((x - arm, y, x + arm, y), fill=(0, 0, 0, 255), width=3)
    draw.line((x, y - arm, x, y + arm), fill=(0, 0, 0, 255), width=3)
    draw.line((x - arm, y, x + arm, y), fill=(0, 255, 0, 255), width=1)
    draw.line((x, y - arm, x, y + arm), fill=(0, 255, 0, 255), width=1)
    draw.ellipse((x - outline, y - outline, x + outline, y + outline), outline=(0, 0, 0, 255), width=2)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0, 255), width=2)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0, 255))


def render_frame(
    frame: np.ndarray,
    output_size: tuple[int, int],
    caption_text: str,
    gaze_xy: tuple[float, float] | None,
    active_label: LabelRange | None,
) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB").resize(output_size, Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", output_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if active_label is not None:
        draw_border(draw, output_size, active_label.color)
    draw_caption(draw, output_size, caption_text)
    if gaze_xy is not None:
        draw_gaze(draw, output_size, gaze_xy)

    return Image.alpha_composite(image, overlay).convert("P", palette=Image.Palette.ADAPTIVE)


def render_episode(
    episode_dir: Path,
    output_path: Path,
    fps: int,
    width: int,
    labels_only: bool,
    overwrite: bool,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    video_path = episode_dir / "video.mp4"
    src_w, src_h, src_fps = get_video_stream_info(video_path)
    output_height = max(1, int(round(src_h * (width / src_w))))
    output_size = (width, output_height)
    gaze_by_frame = load_gaze_frames(episode_dir)
    words = load_words(episode_dir)
    label_ranges = load_label_ranges(episode_dir)
    frame_size = src_w * src_h * 3
    sampled_frames: list[Image.Image] = []

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

    next_capture_time = 0.0
    next_capture_times_by_label = [0.0 for _ in label_ranges]
    frame_idx = 0
    sx = output_size[0] / src_w
    sy = output_size[1] / src_h
    while True:
        assert decoder.stdout is not None
        chunk = decoder.stdout.read(frame_size)
        if len(chunk) < frame_size:
            break

        time_sec = frame_idx / src_fps
        active_label = get_active_label(label_ranges, frame_idx)
        should_capture = False
        if labels_only:
            if active_label is not None:
                label_index = active_label.label_id - 1
                rel_time = (frame_idx - active_label.start_frame) / src_fps
                if rel_time + 1e-9 >= next_capture_times_by_label[label_index]:
                    next_capture_times_by_label[label_index] += 1.0 / fps
                    should_capture = True
        elif time_sec + 1e-9 >= next_capture_time:
            next_capture_time += 1.0 / fps
            should_capture = True

        if should_capture:
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((src_h, src_w, 3)).copy()
            gaze_xy = None
            if frame_idx in gaze_by_frame:
                gx, gy = gaze_by_frame[frame_idx]
                gaze_xy = (gx * sx, gy * sy)
            sampled_frames.append(
                render_frame(
                    frame=frame,
                    output_size=output_size,
                    caption_text=get_active_caption(words, time_sec),
                    gaze_xy=gaze_xy,
                    active_label=active_label,
                )
            )

        frame_idx += 1

    assert decoder.stdout is not None
    decoder.stdout.close()
    decode_rc = decoder.wait()
    if decode_rc != 0:
        raise subprocess.CalledProcessError(decode_rc, decoder.args)
    if not sampled_frames:
        raise RuntimeError(f"No frames were sampled for {episode_dir.name}")

    sampled_frames[0].save(
        output_path,
        save_all=True,
        append_images=sampled_frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )

    return {
        "episode": episode_dir.name,
        "gif_path": str(output_path.resolve()),
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "width": width,
        "gaze_annot": True,
        "caption_annot": True,
        "label_border_annot": True,
        "labels_only": labels_only,
        "labels": [
            {
                "label_id": label.label_id,
                "start_frame": label.start_frame,
                "end_frame_exclusive": label.end_frame_exclusive,
                "color_rgb": list(label.color),
            }
            for label in label_ranges
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create caption+gaze GIFs with GT label borders for matching dataset episodes."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("debug/task_intervention_caption_gifs"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Episode name prefix to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit-per-prefix",
        type=int,
        default=0,
        help="If set, keep only the first N naturally-sorted episodes for each prefix.",
    )
    parser.add_argument("--labels-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    prefixes = args.prefixes or ["task_intervention"]
    episodes: list[Path] = []
    seen: set[str] = set()
    for prefix in prefixes:
        matched = sorted(
            [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)],
            key=lambda path: natural_key(path.name),
        )
        if args.limit_per_prefix > 0:
            matched = matched[: args.limit_per_prefix]
        for episode_dir in matched:
            if episode_dir.name in seen:
                continue
            episodes.append(episode_dir)
            seen.add(episode_dir.name)

    for episode_dir in episodes:
        output_path = output_dir / f"{episode_dir.name}.gif"
        manifest.append(
            render_episode(
                episode_dir=episode_dir,
                output_path=output_path,
                fps=args.fps,
                width=args.width,
                labels_only=args.labels_only,
                overwrite=args.overwrite,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Created {len(manifest)} GIFs")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
