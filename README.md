# Aria VRS Keyframe Detection Pipeline

Project Aria 글래스로 수집한 VRS 데이터에서 키프레임(사용자의 주요 행동 시점)을 자동 탐지하는 파이프라인.

---

## 프로젝트 구조

```
260211_aria_vrs/
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
└── requirements.txt
```

---

## 파이프라인 Overview

```
┌─────────────┐     Step 1      ┌───────────────┐     Step 2      ┌─────────────┐
│  vrs_files/  │ ──────────────→ │ preproc_files/ │ ──────────────→ │  dataset/    │
│  (.vrs)      │   preprocess.py │  (mp4, npz,   │   labeler.py    │  (episode별  │
│              │                 │   wav, json)   │   (GUI 라벨링)  │   학습 데이터)│
└─────────────┘                 └───────────────┘                 └──────┬──────┘
                                                                         │
                                                                    Step 3
                                                                         │
                                                                  ┌──────▼──────┐
                                                                  │ model/      │
                                                                  │ train.py    │
                                                                  │ (학습)      │
                                                                  └─────────────┘
```

---

## Step 1: VRS 전처리 (`preprocess.py`)

VRS 파일에서 RGB 영상, 시선 데이터, 오디오를 추출합니다.

> **주의**: `projectaria_tools` 패키지가 필요합니다 (Aria SDK가 설치된 환경).

### 사용법

```bash
python preprocess.py [--vrs-dir ./vrs_files] [--out-dir ./preproc_files]
```

### 처리 과정

1. `vrs_files/` 내 모든 `.vrs` 파일을 탐색
2. 각 VRS에서 다음 5개 파일을 생성:
   - `{name}_rgb.mp4` — RGB 영상 (256x256, 15fps)
   - `{name}_rgb_with_gaze.mp4` — 시선 오버레이가 그려진 RGB 영상
   - `{name}_gaze.npz` — 프레임별 시선 픽셀 좌표 `(T, 2)`
   - `{name}_pitch_yaw.npz` — 시선 각도 데이터 `(T, 6)` : `[pitch, pitch_lower, pitch_upper, yaw, yaw_lower, yaw_upper]`
   - `{name}_audio.wav` — 원본 오디오 (48kHz)
3. **Idempotent**: 5개 출력 파일이 이미 존재하면 해당 VRS는 건너뜀

### 출력 예시

```
preproc_files/
├── Home_dongjun_test_1_rgb.mp4
├── Home_dongjun_test_1_rgb_with_gaze.mp4
├── Home_dongjun_test_1_gaze.npz
├── Home_dongjun_test_1_pitch_yaw.npz
└── Home_dongjun_test_1_audio.wav
```

---

## (선택) Whisper 음성 전사 (`whisper_cache.py`)

오디오에서 단어 단위 타임스탬프가 포함된 텍스트를 추출합니다.

> `labeler.py` 실행 시 자동으로 전사하므로 별도 실행은 선택사항입니다.

### 사전 요구사항

```bash
export OPENAI_API_KEY="sk-..."
pip install openai
```

### 사용법

```bash
python whisper_cache.py
```

### 출력 형식

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

## Step 2: 라벨링 + 데이터셋 내보내기 (`labeler.py`)

Tkinter GUI에서 영상을 보면서 키프레임을 지정하고, 저장 시 자동으로 `dataset/`에 내보냅니다.

### 실행

```bash
python labeler.py
```

### GUI 조작법

| 키                          | 동작                    |
| -------------------------- | --------------------- |
| `→` / `D` / `L`           | 다음 프레임               |
| `←` / `A` / `H`           | 이전 프레임               |
| `Shift+→` / `Shift+D`     | +10 프레임              |
| `Shift+←` / `Shift+A`     | -10 프레임              |
| `Ctrl+→`                   | +100 프레임             |
| `Ctrl+←`                   | -100 프레임             |
| `Space`                    | 키프레임 토글 (On/Off)    |
| `Delete` / `Backspace`     | 키프레임 삭제             |
| `P`                        | 재생 / 일시정지           |
| `[` / `]`                  | 이전 / 다음 에피소드       |
| `S`                        | 저장 + dataset 내보내기   |
| `Q` / `Escape`             | 종료 (자동 저장)          |

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

## Step 3: 모델 학습 (`model/`)

시선(gaze)과 발화(word)를 인터리빙한 멀티모달 트랜스포머로 키프레임을 탐지합니다.

### 아키텍처

```
입력 토큰 시퀀스 (한 에피소드):
  [GAZE_0] [GAZE_1] [WORD_"Clean"] [GAZE_2] [WORD_"Clean"] [GAZE_3] ... [GAZE_T-1]

각 프레임에 대해:
  - GAZE 토큰: 해당 프레임의 (pitch, yaw) — 항상 1개
  - WORD 토큰: 해당 프레임 시점에 발화 중인 단어의 DeBERTa 임베딩 — 0개 이상

토큰 임베딩 = ContentProjection + SinusoidalPosEnc(frame_idx) + ModalityTypeEmb
                          ↓
              TransformerEncoder (4 layers)
                          ↓
            ClassificationHead → 프레임별 키프레임 확률
```

- **Text Encoder**: DeBERTa-v3-base (frozen, 768dim) — 서브워드 → 단어 mean pooling
- **Transformer**: d_model=128, nhead=4, 4 layers, FFN=256
- **파라미터**: ~680K (DeBERTa 제외)

### 학습

```bash
# 기본값 (200 epochs, LOO-CV)
python -m model.train

# 옵션 조정
python -m model.train \
    --epochs 200 \
    --lr 1e-3 \
    --d-model 128 \
    --num-layers 4 \
    --device cpu \
    --save-dir ./model/checkpoints
```

#### 주요 옵션

| 옵션                | 기본값               | 설명                                    |
| ------------------- | -------------------- | --------------------------------------- |
| `--preproc-dir`     | `./preproc_files`    | 전처리 데이터 경로                       |
| `--epochs`          | `200`                | 에포크 수 (fold당)                       |
| `--lr`              | `1e-3`               | 학습률                                   |
| `--weight-decay`    | `1e-2`               | AdamW weight decay                       |
| `--d-model`         | `128`                | Transformer hidden dim                   |
| `--nhead`           | `4`                  | Attention heads                          |
| `--num-layers`      | `4`                  | Transformer layers                       |
| `--dim-feedforward` | `256`                | FFN dimension                            |
| `--dropout`         | `0.1`                | Dropout                                  |
| `--device`          | `auto`               | `cpu` / `cuda` / `mps` / `auto`         |
| `--save-dir`        | `./model/checkpoints`| 체크포인트 저장 경로                      |

### 학습 전략

- **Leave-One-Out CV**: 5개 에피소드 중 1개를 validation으로 교대 (5-fold)
- **Class Imbalance**: BCE loss에 `pos_weight` 적용 (~9:1 ~ 13:1)
- **Optimizer**: AdamW + CosineAnnealingLR
- **Gradient Clipping**: max_norm=1.0

### 출력

```
model/checkpoints/
├── fold0_Home_dongjun_test_1.pt       # fold별 best 모델
├── fold1_Home_dongjun_test_1_2.pt
├── fold2_Home_dongjun_test_1_3.pt
├── fold3_Home_dongjun_test_1_4.pt
├── fold4_Home_dongjun_test_1_5.pt
└── loo_cv_summary.json                # 전체 CV 결과 요약
```

---

## 새 데이터 추가 워크플로우

```bash
# 1. 새 VRS 파일을 vrs_files/에 넣기
cp /path/to/new_recording.vrs vrs_files/

# 2. 전처리 (기존 파일은 자동 스킵)
python preprocess.py

# 3. (선택) Whisper 전사 선실행
export OPENAI_API_KEY="sk-..."
python whisper_cache.py

# 4. 라벨링 GUI에서 키프레임 지정 후 저장
python labeler.py

# 5. 모델 학습
python -m model.train --epochs 200
```

---

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
