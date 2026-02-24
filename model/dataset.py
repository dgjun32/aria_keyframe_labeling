"""
dataset.py — Multimodal Dataset for Keyframe Detection
=======================================================
Builds interleaved (WORD, GAZE) token sequences per episode.

Token sequence example (5 frames, word "give" spoken during frames 2-3):
  [GAZE_0] [GAZE_1] [WORD_"give"] [GAZE_2] [WORD_"give"] [GAZE_3] [GAZE_4]

A word token is inserted at EVERY frame within the word's spoken duration,
placed before that frame's GAZE token.

Each token carries:
  - content embedding (gaze features or DeBERTa word embedding)
  - frame index (for positional embedding)
  - modality type (0=gaze, 1=word)
  - for GAZE tokens: a binary label (keyframe or not)
"""
from __future__ import annotations

import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


# ── DeBERTa word embedding extractor (run once, cache results) ────────────
class WordEmbeddingExtractor:
    """Extract per-word embeddings from DeBERTa-v3-base (frozen).

    For subword tokens belonging to the same word, mean-pool their
    hidden states to get a single word-level embedding.
    """

    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def extract(self, sentence: str, words: list[dict]) -> dict[str, torch.Tensor]:
        """Extract per-word embeddings by aligning subwords to words.

        Args:
            sentence: full sentence text
            words: list of {"word": str, "start": float, "end": float}

        Returns:
            dict mapping word string to (hidden_dim,) tensor.
            If a word appears multiple times, they get separate entries
            keyed as "word__0", "word__1", etc.
        """
        enc = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"][0].tolist()  # list of (start, end)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0]  # (seq_len, 768)

        # Build character-level word boundaries from the original sentence
        word_char_spans = []
        search_start = 0
        for w_info in words:
            w = w_info["word"]
            idx = sentence.find(w, search_start)
            if idx == -1:
                # fallback: case-insensitive search
                idx = sentence.lower().find(w.lower(), search_start)
            if idx == -1:
                idx = search_start  # last resort
            word_char_spans.append((idx, idx + len(w)))
            search_start = idx + len(w)

        # Map each word to its subword token indices
        result = {}
        word_counts = {}
        for i, (w_info, (w_start, w_end)) in enumerate(
            zip(words, word_char_spans)
        ):
            # Find tokens that overlap with this word's character span
            token_indices = []
            for t_idx, (t_start, t_end) in enumerate(offsets):
                if t_start == 0 and t_end == 0:
                    continue  # special token
                if t_start < w_end and t_end > w_start:
                    token_indices.append(t_idx)

            if token_indices:
                word_emb = hidden[token_indices].mean(dim=0)  # mean pool
            else:
                word_emb = torch.zeros(self.hidden_dim, device=self.device)

            # Handle duplicate words in same sentence
            w_text = w_info["word"]
            count = word_counts.get(w_text, 0)
            key = f"{w_text}__{count}"
            word_counts[w_text] = count + 1
            result[key] = word_emb.cpu()

        return result


# ── Episode data structure ────────────────────────────────────────────────
class EpisodeData:
    """Holds all preprocessed data for one episode."""

    def __init__(
        self,
        name: str,
        num_frames: int,
        gaze_features: np.ndarray,       # (T, 2) pitch, yaw
        word_embeddings: list[torch.Tensor],  # list of (768,) per word
        word_frame_ranges: list[tuple[int, int]],  # (start_frame, end_frame) per word
        keyframe_labels: np.ndarray,      # (T,) binary
    ):
        self.name = name
        self.num_frames = num_frames
        self.gaze_features = gaze_features
        self.word_embeddings = word_embeddings
        self.word_frame_ranges = word_frame_ranges
        self.keyframe_labels = keyframe_labels


# ── Build interleaved token sequence ──────────────────────────────────────
TOKEN_TYPE_GAZE = 0
TOKEN_TYPE_WORD = 1


def build_token_sequence(episode: EpisodeData, gaze_dim: int = 2,
                         word_dim: int = 768):
    """Build interleaved token sequence from an episode.

    A WORD token is inserted at **every** frame within the word's spoken
    duration (start_frame to end_frame inclusive), not just the first frame.
    This means a word like "give" spoken over frames 2-4 produces three
    separate WORD tokens (one per frame), each carrying the same DeBERTa
    embedding but associated with a different frame index.

    Returns:
        gaze_features:  (seq_len, gaze_dim) — gaze content (zero for WORD tokens)
        word_features:  (seq_len, word_dim) — word content (zero for GAZE tokens)
        token_types:    (seq_len,) — 0=gaze, 1=word
        frame_indices:  (seq_len,) — which frame each token belongs to
        is_gaze_token:  (seq_len,) — bool mask for GAZE tokens
        labels:         (num_frames,) — binary keyframe labels
    """
    T = episode.num_frames

    # For each frame, determine which words are active (spoken at that frame)
    frame_to_words: dict[int, list[int]] = {}  # frame -> list of word indices
    for w_idx, (f_start, f_end) in enumerate(episode.word_frame_ranges):
        for f in range(f_start, f_end + 1):
            if 0 <= f < T:
                frame_to_words.setdefault(f, []).append(w_idx)

    # Build sequence: for each frame, insert all active WORD tokens then GAZE
    gaze_list = []
    word_list = []
    type_list = []
    frame_list = []
    is_gaze_list = []

    for f in range(T):
        # Insert WORD tokens for ALL words active at this frame
        if f in frame_to_words:
            for w_idx in frame_to_words[f]:
                gaze_list.append(np.zeros(gaze_dim, dtype=np.float32))
                word_list.append(episode.word_embeddings[w_idx].numpy())
                type_list.append(TOKEN_TYPE_WORD)
                frame_list.append(f)
                is_gaze_list.append(False)

        # Always insert GAZE token
        gaze_list.append(episode.gaze_features[f].astype(np.float32))
        word_list.append(np.zeros(word_dim, dtype=np.float32))
        type_list.append(TOKEN_TYPE_GAZE)
        frame_list.append(f)
        is_gaze_list.append(True)

    return {
        "gaze_features": np.stack(gaze_list),       # (S, 2)
        "word_features": np.stack(word_list),        # (S, 768)
        "token_types": np.array(type_list, dtype=np.int64),  # (S,)
        "frame_indices": np.array(frame_list, dtype=np.int64),  # (S,)
        "is_gaze_token": np.array(is_gaze_list, dtype=np.bool_),  # (S,)
        "labels": episode.keyframe_labels.astype(np.float32),  # (T,)
    }


# ── PyTorch Dataset ───────────────────────────────────────────────────────
class KeyframeDataset(Dataset):
    """Dataset that yields interleaved multimodal token sequences.

    Each item is one episode (since episodes are few and short).
    """

    def __init__(self, preproc_dir: str, episode_names: list[str],
                 word_extractor: WordEmbeddingExtractor | None = None,
                 fps: float = 15.0):
        self.preproc_dir = preproc_dir
        self.fps = fps
        self.episodes: list[dict] = []

        # Load annotations
        ann_path = os.path.join(preproc_dir, "keyframe_annotations.json")
        with open(ann_path) as f:
            annotations = json.load(f)

        # Create word extractor if not provided
        if word_extractor is None:
            word_extractor = WordEmbeddingExtractor()

        for name in episode_names:
            ep = self._load_episode(name, annotations, word_extractor)
            seq = build_token_sequence(ep)
            self.episodes.append(seq)

    def _load_episode(self, name: str, annotations: dict,
                      extractor: WordEmbeddingExtractor) -> EpisodeData:
        """Load and preprocess a single episode."""
        # Gaze (pitch, yaw)
        py_path = os.path.join(self.preproc_dir, f"{name}_pitch_yaw.npz")
        pitch_yaw = np.load(py_path)["pitch_yaw"]  # (T, 6)
        T = len(pitch_yaw)
        gaze_features = np.stack([pitch_yaw[:, 0], pitch_yaw[:, 3]], axis=1)  # (T, 2)

        # Transcript
        tr_path = os.path.join(
            self.preproc_dir, "transcripts", f"{name}_transcript.json"
        )
        if os.path.exists(tr_path):
            with open(tr_path) as f:
                transcript = json.load(f)
            words_info = transcript.get("words", [])
            sentence = transcript.get("text", "")
        else:
            words_info = []
            sentence = ""

        # Extract word embeddings via DeBERTa
        if words_info and sentence:
            word_embs_dict = extractor.extract(sentence, words_info)
            word_embeddings = list(word_embs_dict.values())
        else:
            word_embeddings = []

        # Word-to-frame mapping
        word_frame_ranges = []
        for w in words_info:
            f_start = int(w["start"] * self.fps)
            f_end = int(w["end"] * self.fps)
            f_start = max(0, min(f_start, T - 1))
            f_end = max(0, min(f_end, T - 1))
            word_frame_ranges.append((f_start, f_end))

        # Keyframe labels
        kfs = annotations.get(name, [])
        labels = np.zeros(T, dtype=np.float32)
        for kf in kfs:
            idx = kf["frame"] if isinstance(kf, dict) else kf
            if 0 <= idx < T:
                labels[idx] = 1.0

        return EpisodeData(
            name=name,
            num_frames=T,
            gaze_features=gaze_features,
            word_embeddings=word_embeddings,
            word_frame_ranges=word_frame_ranges,
            keyframe_labels=labels,
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        seq = self.episodes[idx]
        return {
            "gaze_features": torch.from_numpy(seq["gaze_features"]),
            "word_features": torch.from_numpy(seq["word_features"]),
            "token_types": torch.from_numpy(seq["token_types"]),
            "frame_indices": torch.from_numpy(seq["frame_indices"]),
            "is_gaze_token": torch.from_numpy(seq["is_gaze_token"]),
            "labels": torch.from_numpy(seq["labels"]),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length episodes with padding.

    Pads token sequences to the max sequence length in the batch,
    and label arrays to the max number of frames.
    """
    max_seq_len = max(b["gaze_features"].shape[0] for b in batch)
    max_frames = max(b["labels"].shape[0] for b in batch)
    B = len(batch)

    gaze_dim = batch[0]["gaze_features"].shape[1]
    word_dim = batch[0]["word_features"].shape[1]

    gaze_features = torch.zeros(B, max_seq_len, gaze_dim)
    word_features = torch.zeros(B, max_seq_len, word_dim)
    token_types = torch.zeros(B, max_seq_len, dtype=torch.long)
    frame_indices = torch.zeros(B, max_seq_len, dtype=torch.long)
    is_gaze_token = torch.zeros(B, max_seq_len, dtype=torch.bool)
    attention_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)
    labels = torch.zeros(B, max_frames)
    label_mask = torch.zeros(B, max_frames, dtype=torch.bool)

    for i, b in enumerate(batch):
        S = b["gaze_features"].shape[0]
        T = b["labels"].shape[0]
        gaze_features[i, :S] = b["gaze_features"]
        word_features[i, :S] = b["word_features"]
        token_types[i, :S] = b["token_types"]
        frame_indices[i, :S] = b["frame_indices"]
        is_gaze_token[i, :S] = b["is_gaze_token"]
        attention_mask[i, :S] = True
        labels[i, :T] = b["labels"]
        label_mask[i, :T] = True

    return {
        "gaze_features": gaze_features,
        "word_features": word_features,
        "token_types": token_types,
        "frame_indices": frame_indices,
        "is_gaze_token": is_gaze_token,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_mask": label_mask,
    }
