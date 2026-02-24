#!/usr/bin/env python3
"""
Pre-compute Whisper transcripts for all tasks in preproc_files/.

Usage:
    export OPENAI_API_KEY="sk-..."
    python whisper_cache.py

Generates preproc_files/transcripts/{task_name}_transcript.json for each task.
Skips tasks that already have cached transcripts.
"""
from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from data_loader import discover_tasks

PROJECT_DIR = Path(__file__).parent
PREPROC_DIR = PROJECT_DIR / "preproc_files"
TRANSCRIPT_DIR = PREPROC_DIR / "transcripts"


def convert_to_mono(wav_path: str) -> str:
    """Convert multi-channel WAV to 16kHz mono temp file using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", wav_path,
            "-ac", "1", "-ar", "16000",
            tmp.name,
        ],
        check=True,
    )
    return tmp.name


def transcribe_task(client: OpenAI, task_name: str) -> dict:
    """Transcribe audio and return word-level timestamps."""
    audio_path = str(PREPROC_DIR / f"{task_name}_audio.wav")
    mono_path = convert_to_mono(audio_path)

    try:
        with open(mono_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
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


def main():
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    client = OpenAI()  # uses OPENAI_API_KEY env var

    tasks = discover_tasks(str(PREPROC_DIR))
    print(f"Found {len(tasks)} tasks")

    for i, task in enumerate(tasks):
        out_path = TRANSCRIPT_DIR / f"{task}_transcript.json"

        if out_path.exists():
            print(f"  [{i + 1}/{len(tasks)}] {task} -- CACHED, skipping")
            continue

        print(f"  [{i + 1}/{len(tasks)}] {task} -- transcribing...")
        try:
            result = transcribe_task(client, task)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"    -> {len(result['words'])} words saved")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
