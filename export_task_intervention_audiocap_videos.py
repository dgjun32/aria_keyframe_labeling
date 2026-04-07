#!/usr/bin/env python3
"""Export task_intervention videos with burned-in captions and muxed audio."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from eval_vlm_baseline import prepare_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default='./dataset')
    p.add_argument('--audio-dir', default='./preproc_files')
    p.add_argument('--output-dir', default='./results/task_intervention_audiocap_gaze_videos')
    p.add_argument('--episodes', default='')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--target-resolution', default='none')
    return p.parse_args()


def parse_target_resolution(value: str):
    txt = value.strip().lower()
    if txt in {'none', 'raw', 'original'}:
        return None
    if 'x' in txt:
        w, h = txt.split('x', 1)
        return (int(w), int(h))
    size = int(txt)
    return (size, size)


def resolve_episodes(dataset_dir: str, episodes_arg: str):
    if episodes_arg.strip():
        return [e.strip() for e in episodes_arg.split(',') if e.strip()]
    return sorted(
        name for name in os.listdir(dataset_dir)
        if name.startswith('task_intervention') and os.path.isdir(os.path.join(dataset_dir, name))
    )


def export_one(dataset_dir: str, audio_dir: str, output_dir: str, episode: str, target_resolution):
    episode_dir = Path(dataset_dir) / episode
    video_path = episode_dir / 'video.mp4'
    transcript_path = episode_dir / 'transcript.json'
    gaze_path = episode_dir / 'gaze.json'
    audio_path = Path(audio_dir) / f'{episode}_audio.wav'
    if not video_path.exists():
        return {'episode': episode, 'success': False, 'error': f'missing {video_path}'}
    if not transcript_path.exists():
        return {'episode': episode, 'success': False, 'error': f'missing {transcript_path}'}
    if not gaze_path.exists():
        return {'episode': episode, 'success': False, 'error': f'missing {gaze_path}'}
    if not audio_path.exists():
        return {'episode': episode, 'success': False, 'error': f'missing {audio_path}'}
    transcript = json.loads(transcript_path.read_text())
    gaze_payload = json.loads(gaze_path.read_text())
    gaze_frames = gaze_payload.get('frames', []) if isinstance(gaze_payload, dict) else []
    mp4_bytes = prepare_video(
        str(video_path),
        caption=True,
        gaze_annot=True,
        transcript=transcript,
        gaze_data=gaze_frames,
        target_resolution=target_resolution,
        include_audio=True,
        audio_path=str(audio_path),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{episode}_audiocap_gaze.mp4'
    out_path.write_bytes(mp4_bytes)
    return {'episode': episode, 'success': True, 'output_path': str(out_path), 'size_bytes': len(mp4_bytes)}


def main():
    args = parse_args()
    target_resolution = parse_target_resolution(args.target_resolution)
    episodes = resolve_episodes(args.dataset_dir, args.episodes)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(export_one, args.dataset_dir, args.audio_dir, args.output_dir, episode, target_resolution): episode
            for episode in episodes
        }
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if result['success']:
                print(f"[ok] {result['episode']} -> {result['output_path']}")
            else:
                print(f"[fail] {result['episode']}: {result['error']}")
    results.sort(key=lambda x: x['episode'])
    success = sum(1 for r in results if r['success'])
    print(f'[done] {success}/{len(results)} exported')


if __name__ == '__main__':
    main()
