#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def ffprobe_video_info(path: Path) -> tuple[int, int, float]:
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
                str(path),
            ]
        )
    )
    stream = payload["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / float(den)


def load_font(size: int):
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


def load_caption_ranges(payload_dir: Path) -> list[tuple[float, float, str]]:
    ranges: list[tuple[float, float, str]] = []
    for payload_path in sorted(payload_dir.glob("step_*_payload.json")):
        payload = json.loads(payload_path.read_text())
        segment_name = Path(payload["input_segment_path"]).name
        match = re.search(r"_(\d+(?:\.\d+)?)s_(\d+(?:\.\d+)?)s\.mp4$", segment_name)
        if not match:
            continue
        start_sec = float(match.group(1))
        end_sec = float(match.group(2))
        text = payload.get("transcript_text", "").strip()
        if text:
            ranges.append((start_sec, end_sec, text))

    ranges.sort()
    merged: list[tuple[float, float, str]] = []
    for start_sec, end_sec, text in ranges:
        if merged and merged[-1][2] == text and start_sec <= merged[-1][1] + 0.15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_sec), text)
        else:
            merged.append((start_sec, end_sec, text))
    return merged


def load_word_timestamps(transcript_json: Path) -> list[dict]:
    payload = json.loads(transcript_json.read_text())
    return payload.get("words", [])


def active_caption_from_ranges(ranges: list[tuple[float, float, str]], time_sec: float) -> str:
    for start_sec, end_sec, text in ranges:
        if start_sec <= time_sec <= end_sec:
            return text
    return ""


def active_caption_from_words(words: list[dict], time_sec: float) -> str:
    active = [word["word"] for word in words if word["start"] <= time_sec <= word["end"]]
    return " ".join(active)


def draw_caption(draw: ImageDraw.ImageDraw, size: tuple[int, int], text: str) -> None:
    if not text:
        return
    width, height = size
    font = load_font(max(28, height // 18))
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = max(10, width // 80)
    pad_y = max(8, height // 110)
    x0 = max(0, (width - text_w) // 2 - pad_x)
    y0 = max(0, height // 28 - pad_y)
    x1 = min(width, x0 + text_w + pad_x * 2)
    y1 = min(height, y0 + text_h + pad_y * 2)
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=max(8, height // 70),
        fill=(0, 0, 0, 165),
    )
    draw.text((x0 + pad_x, y0 + pad_y - 1), text, font=font, fill=(255, 255, 255, 255))


def draw_gaze(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    gaze_xy: tuple[float, float],
    y_offset_px: float,
) -> None:
    width, height = size
    x = max(0, min(width - 1, int(round(gaze_xy[0]))))
    y = max(0, min(height - 1, int(round(gaze_xy[1] + y_offset_px))))
    radius = max(6, height // 60)
    arm = radius * 2
    outline = radius + 2

    draw.line((x - arm, y, x + arm, y), fill=(0, 0, 0, 255), width=4)
    draw.line((x, y - arm, x, y + arm), fill=(0, 0, 0, 255), width=4)
    draw.line((x - arm, y, x + arm, y), fill=(0, 255, 0, 255), width=2)
    draw.line((x, y - arm, x, y + arm), fill=(0, 255, 0, 255), width=2)
    draw.ellipse((x - outline, y - outline, x + outline, y + outline), outline=(0, 0, 0, 255), width=2)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0, 255), width=2)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 0, 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a process video with merged audio, shifted gaze, and caption overlays."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--gaze-npz", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--payload-dir", type=Path)
    group.add_argument("--transcript-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaze-y-offset", type=float, default=-90.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)

    width, height, fps = ffprobe_video_info(args.video)
    frame_size = width * height * 3
    gaze = np.load(args.gaze_npz)["gaze"]
    caption_getter: callable
    if args.transcript_json is not None:
        words = load_word_timestamps(args.transcript_json)
        print(f"loaded_words {len(words)}")
        caption_getter = lambda time_sec: active_caption_from_words(words, time_sec)
    else:
        caption_ranges = load_caption_ranges(args.payload_dir)
        print("caption_ranges", caption_ranges)
        caption_getter = lambda time_sec: active_caption_from_ranges(caption_ranges, time_sec)

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(args.video),
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
            "-y" if args.overwrite else "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-i",
            str(args.audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )

    frame_idx = 0
    while True:
        assert decoder.stdout is not None
        chunk = decoder.stdout.read(frame_size)
        if len(chunk) < frame_size:
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
        image = Image.fromarray(frame, mode="RGB").convert("RGBA")
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw_caption(draw, (width, height), caption_getter(frame_idx / fps))
        if frame_idx < len(gaze):
            draw_gaze(draw, (width, height), tuple(map(float, gaze[frame_idx])), args.gaze_y_offset)

        composed = Image.alpha_composite(image, overlay).convert("RGB")
        assert encoder.stdin is not None
        encoder.stdin.write(composed.tobytes())
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

    print(f"output {args.output.resolve()}")


if __name__ == "__main__":
    main()
