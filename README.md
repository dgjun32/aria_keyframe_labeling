# Aria VRS Keyframe Detection Pipeline

Project Aria 글래스로 수집한 VRS 데이터에서 키프레임(사용자의 주요 행동 시점)을 자동 탐지하는 파이프라인.

---

## 프로젝트 구조

```
project_root/
├── vrs_files/                 # 원본 VRS 파일 (입력)
│   ├── {name}.vrs
│   └── {name}.vrs.json
│
├── preproc_files/             # 전처리 결과 (중간 산출물)
│   ├── {name}_rgb.mp4              # RGB 영상 (256x256, 15fps)
│   ├── {name}_rgb_with_gaze.mp4    # RGB + 시선 오버레이 영상
│   ├── {name}_gaze.npz             # 시선 좌표 (T, 2)
│   ├── {name}_pitch_yaw.npz        # 시선 각도 (T, 6)
│   ├── {name}_audio.wav            # 오디오 (48kHz)
│   ├── keyframe_annotations.json   # 키프레임 라벨 (labeler에서 생성)
│   └── transcripts/
│       └── {name}_transcript.json  # Whisper 음성 인식 결과
│
├── dataset/                   # 학습용 데이터셋 (최종 출력)
│   └── {episode_name}/
│       ├── video.mp4               # RGB 영상 복사본
│       ├── transcript.json         # 음성 인식 결과
│       ├── gaze.json               # 프레임별 시선 데이터
│       └── labels.npy              # 키프레임 라벨 (T,) int8 바이너리
│
├── model/                     # 키프레임 탐지 모델
│   ├── dataset.py                  # PyTorch Dataset (인터리빙 토큰 시퀀스)
│   ├── model.py                    # Multimodal Transformer
│   ├── train.py                    # LOO-CV 학습 스크립트
│   └── checkpoints/                # 학습된 모델 체크포인트
│
├── preprocess.py              # Step 1: VRS → preproc_files/
├── labeler.py                 # Step 2: GUI 라벨링 + dataset/ 내보내기
├── data_loader.py             # 데이터 로딩 유틸리티
├── whisper_cache.py           # Whisper 일괄 전사 스크립트
├── eye_gaze.py                # Aria 시선 추론 모듈
├── eval_vlm_baseline.py 
└── requirements.txt
```

---

## 파이프라인 Overview

```
┌─────────────┐     Step 1      ┌───────────────┐     Step 2           ┌─────────────┐
│  vrs_files/  │ ──────────────→ │ preproc_files/ │ ──────────────→    │  dataset/   │
│  (.vrs)      │   preprocess.py │  (mp4, npz,   │   labeler.py        │  (episode별  │
│              │                 │   wav, json)   │(GUI keyframe 라벨링) │   eval 데이터)│
└─────────────┘                 └───────────────┘                       └──────┬──────┘
                                                                              │
                                                                          Step 3
                                                                              │
                                                                        ┌──────▼──────┐
                                                                        │ model/      │
                                                                        │ train.py    │
                                                                        │ (학습)       │ 
                                                                        └─────────────┘
```

---

## Step 1: Preprocessing VRS files (`preprocess.py`)

Extract video, audio, gaze (pitch, yaw) from the .vrs files (which are located under `vrs_files/`).

> **주의**: `projectaria_tools` 패키지가 필요합니다 (Aria SDK가 설치된 환경).

### Usage

```bash
python preprocess.py --vrs-dir ./vrs_files --out-dir ./preproc_files
```
Extracted files will be saved under `--output-dir`. 

### Details

1. `vrs_files/` 내 모든 `.vrs` 파일을 탐색
2. 각 VRS에서 다음 5개 파일을 생성:
   - `{name}_rgb.mp4` — RGB 영상 (256x256, 15fps)
   - `{name}_rgb_with_gaze.mp4` — 시선 오버레이가 그려진 RGB 영상
   - `{name}_gaze.npz` — 프레임별 시선 픽셀 좌표 `(T, 2)`
   - `{name}_pitch_yaw.npz` — 시선 각도 데이터 `(T, 6)` : `[pitch, pitch_lower, pitch_upper, yaw, yaw_lower, yaw_upper]`
   - `{name}_audio.wav` — 원본 오디오 (48kHz)
3. **Idempotent**: 5개 출력 파일이 이미 존재하면 해당 VRS는 건너뜀

### Example outputs

```
preproc_files/
├── Home_dongjun_test_1_rgb.mp4
├── Home_dongjun_test_1_rgb_with_gaze.mp4
├── Home_dongjun_test_1_gaze.npz
├── Home_dongjun_test_1_pitch_yaw.npz
└── Home_dongjun_test_1_audio.wav
```

---

## hisper voice transcription (`whisper_cache.py`)

From audio files, extract timestamp of the spoken word

### Usage

```bash
export OPENAI_API_KEY=""
python whisper_cache.py
```

### Example output

`preproc_files/transcripts/{name}_transcript.json`:

```json
{
  "text": "Clean this table, please.",
  "words": [
    {"word": "Clean", "start": 4.86, "end": 5.42},
    {"word": "this",  "start": 5.42, "end": 5.58},
    {"word": "table,","start": 5.58, "end": 5.82},
    {"word": "please.","start": 5.82, "end": 6.24}
  ]
}
```

---

## Step 2: Labeling keyframe & build evaluation dataset (`labeler.py`)

Label the keyframe(s) using GUI tools, and save it under `dataset/` directory.
(If the dataset is saved, it is okay to remove `preproc_files/` and `vrs_files/`).

### Usage

```bash
python labeler.py --prefix {task_name}
```

prefix: signature of the task type (e.g., `2f_store_dongjun_single_object_pickup`, `2f_store_juheon_multiple_object_pickup`).


### 화면 구성

- **상단**: RGB 영상 + 시선 오버레이 + 현재 발화 자막
- **하단**: Angular velocity 그래프 (현재 프레임 위치 표시)
- **우측**: 키프레임 리스트, 에피소드 네비게이션

### 라벨 저장 & 데이터셋 내보내기

`S` 키를 누르면:

1. **라벨 저장**: `preproc_files/keyframe_annotations.json`에 저장

   ```json
   {
     "Home_dongjun_test_1": [
       {"frame": 85}, {"frame": 86}, {"frame": 87},
       {"frame": 88}, {"frame": 89}, {"frame": 90}
     ]
   }
   ```

2. **dataset/ 내보내기**: 각 에피소드별 폴더 자동 생성

   ```
   dataset/Home_dongjun_test_1/
   ├── video.mp4          # preproc_files에서 복사
   ├── transcript.json    # Whisper 전사 결과
   ├── gaze.json          # 프레임별 시선 데이터
   └── labels.npy         # (T,) int8 바이너리 배열
   ```

### `dataset/` 출력 상세

#### `labels.npy`

- Shape: `(T,)`, dtype: `int8`
- 키프레임 = `1`, 그 외 = `0`
- 예: `[0, 0, 0, ..., 1, 1, 1, 1, 1, 1, 0, ..., 0]`

#### `gaze.json`

프레임별 시선 좌표 및 pitch/yaw 각도:

```json
{
  "frames": [
    {
      "frame_idx": 0,
      "gaze_x": 105.27,
      "gaze_y": 171.82,
      "pitch": -0.482,
      "pitch_lower": -0.505,
      "pitch_upper": -0.444,
      "yaw": 0.265,
      "yaw_lower": 0.244,
      "yaw_upper": 0.281
    }
  ]
}
```

#### `transcript.json`

Whisper API의 단어 단위 전사 결과 (whisper_cache.py 출력과 동일 형식).

---

## Step 3: Evaluation

### 3-1. VLM Baseline Evaluation (`eval_vlm_baseline.py`)

Sends each episode's video (with optional overlays) to Gemini VLM, asks it to
predict the keyframe time range, and evaluates against ground-truth labels.

#### Quick Start

```bash
# Simplest run (no overlays, default 256x256 resize, gemini-2.5-pro)
python eval_vlm_baseline.py

# With caption overlay
python eval_vlm_baseline.py --caption

# With gaze overlay
python eval_vlm_baseline.py --gaze-annot

# Both overlays + debug videos saved
python eval_vlm_baseline.py --caption --gaze-annot --debug

# Different model, custom FPS hint, verbose logging
python eval_vlm_baseline.py --model gemini-2.5-flash --video-fps 4 --verbose
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-dir` | `./dataset` | Path to the `dataset/` directory containing episode folders. |
| `--model` | `gemini-2.5-pro` | Gemini model to use. Choices: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-flash`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`. |
| `--output-dir` | `./eval_results` | Directory for JSONL results and summary JSON. |
| `--resume` | `false` | Skip episodes that already have results in the JSONL file. Useful for resuming interrupted runs. |
| `--episodes` | all | Comma-separated episode names to evaluate (e.g. `ep_1,ep_2`). Default: all episodes in dataset dir. |
| `--video-fps` | `2` | FPS hint sent to Gemini for video sampling. Lower = fewer frames sent = cheaper. |
| `--caption` | `false` | Overlay real-time transcript captions on the video (burned-in subtitles at the top). |
| `--gaze-annot` | `false` | Overlay the wearer's gaze point as a green dot on the video. |
| `--target-resolution` | `256` | Resize all videos to this resolution **before** overlaying captions/gaze. Accepts `N` for NxN or `WxH`. Set to `none` to keep original resolution. Videos in the dataset vary (256x256 or 1408x1408), so this ensures consistent VLM input. |
| `--tolerance` | `0.5,1.0,1.5` | Comma-separated time tolerance values (seconds) for midpoint hit-rate metrics. |
| `--frame-tolerance` | `0,1,3,5` | Comma-separated frame tolerance values for frame-level hit-rate metrics. |
| `--debug` | `false` | Save the exact video sent to the VLM as `debug/{config_name}/input_video_{i}.mp4` (H.264 encoded, playable in VSCode). |
| `--api-key` | *(built-in)* | Gemini API key. |
| `--verbose` | `false` | Print per-episode prediction details and reasoning. |

#### Video Preparation Pipeline

When `--caption`, `--gaze-annot`, or `--target-resolution` is specified, each
video goes through the following pipeline:

```
read frame → resize to target resolution → draw caption → draw gaze dot → encode H.264
```

- **Resize happens first**, so overlay text and markers are rendered at the
  output resolution and remain crisp.
- **Gaze coordinates are automatically rescaled** from the original resolution
  to the target resolution (`gaze_x *= M/N`, `gaze_y *= M/N`).
- The output is re-encoded to H.264 via ffmpeg for broad player compatibility.

#### 2x2 Configuration Matrix (caption x gaze)

The `--caption` and `--gaze-annot` flags create four configurations, each with
a tailored VLM prompt that describes which cues are available:

| Config Name | `--caption` | `--gaze-annot` | VLM Prompt Focus |
|-------------|:-----------:|:--------------:|------------------|
| `no_cap_no_gaze` | | | Hand gesture + utterance content only |
| `cap_only` | x | | Caption timing + hand gesture correlation |
| `gaze_only` | | x | Gaze fixation + hand approach convergence |
| `cap_gaze` | x | x | All three cues: caption timing, gaze fixation, hand gesture |

The prompt instructs the VLM to find the keyframe by looking for the
**convergence of multiple cues** (utterance timing, hand gesture, gaze
fixation) rather than any single signal alone.

#### Output Files

The script produces three types of output:

**1. Per-episode JSONL** — `eval_results/{exp_tag}_results.jsonl`

One JSON object per line. Contains full details for each episode:

```jsonc
{
  "episode": "2f_store_dongjun_setup_single_object_pickup_1",
  "gt": { "kf_start_frame": 85, "kf_end_frame": 90, "duration_sec": 14.07, ... },
  "prediction": { "keyframe_start": 5.5, "keyframe_end": 6.2, "reasoning": "..." },
  "metrics": { "iou": 0.45, "mid_frame_hit": true, "nearest_gt_dist": 2, ... },
  "cost": { "input_tokens": 1234, "cost_usd": 0.0042 },
  "inference_time_sec": 8.213
}
```

**2. Aggregate summary** — `eval_results/{exp_tag}_summary.json`

Single JSON with aggregated statistics (mean IoU, hit rates, total cost, etc.).

**3. Structured results JSON** — `results/result_{model}_gaze_{true|false}_cap_{true|false}_fps{N}.json`

Clean per-experiment results file. This is the primary output for downstream
analysis and the input for gaze refinement (see 3-2 below).

```jsonc
{
  "config": {
    "model": "gemini-2.5-pro",
    "gaze_annot": false,
    "caption": true,
    "video_fps": 2,
    "config_name": "cap_only"
  },
  "episodes": [
    {
      "episode": "2f_store_dongjun_setup_single_object_pickup_1",
      "predicted_interval_sec": [5.5, 6.2],
      "predicted_interval_frames": [82, 93],
      "gt_interval_frames": [85, 90],
      "iou": 0.45,
      "mid_frame_hit": true,
      "midpoint_error_sec": 0.17,
      "pred_mid_frame": 87,
      "inference_time_sec": 8.213
    }
    // ... one entry per episode
  ],
  "aggregate": { /* mean IoU, hit rates, cost totals, ... */ }
}
```

Filename examples:
- `result_gemini-2.5-pro_gaze_false_cap_false_fps2.json`
- `result_gemini-2.5-flash_gaze_true_cap_true_fps4.json`

**4. Debug videos** (only with `--debug`) — `debug/{config_name}/input_video_{i}.mp4`

The exact video bytes sent to the VLM, saved as H.264 MP4.
Useful for visually verifying that captions, gaze dots, and resizing
are applied correctly.

```
debug/
├── no_cap_no_gaze/
│   ├── input_video_0.mp4
│   └── input_video_1.mp4
├── cap_only/
├── gaze_only/
└── cap_gaze/
```

#### Metrics

| Metric | Description |
|--------|-------------|
| `mid_frame_hit` | Whether the predicted midpoint frame falls within the GT keyframe segment (exact match). |
| `frame_hit@k` | Whether any frame within ±k of the predicted midpoint is a GT keyframe (k = 0, 1, 3, 5). |
| `nearest_gt_dist` | Distance (in frames) from the predicted midpoint to the nearest GT keyframe. |
| `iou` | Intersection-over-Union between predicted and GT time intervals. |
| `midpoint_error` | Absolute difference (seconds) between predicted and GT midpoints. |
| `hit@τs` | Whether midpoint error is within τ seconds (τ = 0.5, 1.0, 1.5). |
| `gt_coverage` | Fraction of the GT segment covered by the prediction. |
| `pred_precision` | Fraction of the predicted segment that overlaps with GT. |

---

### 3-2. Gaze-Velocity Keyframe Refinement (`eval_gaze_refinement.py`)

A **post-processing** script that refines VLM-predicted keyframe intervals
using gaze angular velocity. No VLM calls are made — it reads the results
JSON from Step 3-1 and applies gaze-based refinement.

**Idea**: Within the VLM-predicted interval, the frame where gaze angular
velocity is lowest corresponds to a fixation (the wearer staring at the
target object). This frame is often a better keyframe than the interval
midpoint.

#### Quick Start

```bash
# Refine a VLM baseline result
python eval_gaze_refinement.py \
    --vlm-results results/result_gemini-2.5-pro_gaze_false_cap_false_fps2.json

# Custom smoothing parameters
python eval_gaze_refinement.py \
    --vlm-results results/result_gemini-2.5-pro_gaze_true_cap_true_fps2.json \
    --savgol-window 15 --savgol-poly 3
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vlm-results` | *(required)* | Path to the results JSON produced by `eval_vlm_baseline.py` (the file under `results/`). |
| `--dataset-dir` | `./dataset` | Path to the `dataset/` directory (needed to load `gaze.json` and `labels.npy`). |
| `--frame-tolerance` | `0,1,3,5` | Frame tolerance values for hit-rate evaluation. |
| `--savgol-window` | `11` | Savitzky-Golay filter window length for smoothing gaze pitch/yaw. Must be odd and >= 3. |
| `--savgol-poly` | `2` | Savitzky-Golay polynomial order. Must be < `savgol-window`. |
| `--verbose` | `false` | Print detailed per-episode gaze velocity info. |

#### How It Works

```
Phase 1: Precompute angular velocity
  For each episode:
    load gaze.json → extract pitch/yaw (radians)
    → Savitzky-Golay smooth → convert to 3D unit vectors
    → angular distance between consecutive frames × FPS → deg/s

Phase 2: Refine keyframe selection
  For each episode:
    VLM-predicted interval [start_frame, end_frame]
    ├── Midpoint selection (baseline):  frame = (start + end) // 2
    └── Gaze min-velocity selection:    frame = argmin(angular_velocity[start:end])
    Evaluate both against GT labels
```

#### Output

Saved to `results/refined_{original_filename}`:

```jsonc
{
  "source": "results/result_gemini-2.5-pro_gaze_false_cap_false_fps2.json",
  "vlm_config": { /* original VLM config */ },
  "gaze_config": { "savgol_window": 11, "savgol_poly": 2 },
  "episodes": [
    {
      "episode": "2f_store_dongjun_setup_single_object_pickup_1",
      "vlm_interval_frames": [82, 93],
      "gt_interval_frames": [85, 90],
      "iou": 0.45,
      "midpoint": {
        "selected_frame": 87,
        "hit": true,
        "nearest_gt_dist": 0,
        "hit@0": true, "hit@1": true, "hit@3": true, "hit@5": true
      },
      "gaze_velocity": {
        "selected_frame": 86,
        "velocity_deg_s": 3.21,
        "hit": true,
        "nearest_gt_dist": 0,
        "hit@0": true, "hit@1": true, "hit@3": true, "hit@5": true
      }
    }
  ],
  "aggregate": {
    "midpoint":      { "exact_accuracy": 0.65, "mean_nearest_gt_dist": 2.3, ... },
    "gaze_velocity": { "exact_accuracy": 0.72, "mean_nearest_gt_dist": 1.8, ... }
  }
}
```

The console output includes a side-by-side comparison table:

```
  Metric                         Midpoint   Gaze (min vel)    Delta
  ------------------------------ ----------  -------------- --------
  Exact match (+-0)                 65.0%          72.0%    + 7.0%
  frame_hit@+-1 frames              78.0%          84.0%    + 6.0%
  frame_hit@+-3 frames              91.0%          94.0%    + 3.0%
  frame_hit@+-5 frames              96.0%          98.0%    + 2.0%
```

## 환경 설정

```bash
pip install -r requirements.txt

# 모델 학습 추가 의존성
pip install transformers sentencepiece scikit-learn
```

| 패키지                  | 용도                                    |
| ---------------------- | --------------------------------------- |
| `projectaria-tools`    | VRS 파일 파싱 (preprocess.py에서만 필요) |
| `torch`, `torchvision` | 시선 추론 + 모델 학습                    |
| `opencv-python`        | 영상 처리                                |
| `scipy`                | Angular velocity 계산 (savgol filter)    |
| `Pillow`, `matplotlib` | 라벨링 GUI                               |
| `openai`               | Whisper API 음성 전사                    |
| `transformers`         | DeBERTa-v3-base 임베딩 추출              |
| `sentencepiece`        | DeBERTa 토크나이저                        |
| `scikit-learn`         | 학습 평가 메트릭 (F1, AUROC 등)          |
