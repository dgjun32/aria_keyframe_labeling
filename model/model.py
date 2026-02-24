"""
model.py — Multimodal Transformer for Keyframe Detection
=========================================================
Architecture:
  1. Modality-specific projections:
     - GazeProjection: Linear(2 → d_model)
     - WordProjection: Linear(768 → d_model)
  2. Embeddings added per token:
     - Content embedding (from projection above)
     - Positional embedding (based on frame index, shared across modalities)
     - Modality type embedding (gaze=0, word=1)
  3. Transformer encoder (N layers)
  4. Classification head: extract GAZE token hidden states → Linear → sigmoid
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding based on frame index."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_indices: (B, S) long tensor of frame indices
        Returns:
            (B, S, d_model)
        """
        return self.pe[frame_indices]


class KeyframeTransformer(nn.Module):
    """Multimodal transformer for binary keyframe detection.

    Input: interleaved sequence of GAZE and WORD tokens.
    Output: per-frame binary keyframe probability.
    """

    def __init__(
        self,
        gaze_dim: int = 2,
        word_dim: int = 768,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_frames: int = 256,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Modality projections ──
        self.gaze_proj = nn.Sequential(
            nn.Linear(gaze_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.word_proj = nn.Sequential(
            nn.Linear(word_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── Embeddings ──
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_frames)
        self.type_emb = nn.Embedding(2, d_model)  # 0=gaze, 1=word

        # ── Pre-transformer LayerNorm ──
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # ── Transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        gaze_features: torch.Tensor,    # (B, S, 2)
        word_features: torch.Tensor,    # (B, S, 768)
        token_types: torch.Tensor,      # (B, S) long — 0=gaze, 1=word
        frame_indices: torch.Tensor,    # (B, S) long
        is_gaze_token: torch.Tensor,    # (B, S) bool
        attention_mask: torch.Tensor,   # (B, S) bool — True=valid
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, S) — logits for every token position.
                    Use is_gaze_token mask to extract per-frame predictions.
        """
        B, S = token_types.shape

        # ── Project each modality ──
        gaze_emb = self.gaze_proj(gaze_features)   # (B, S, d_model)
        word_emb = self.word_proj(word_features)    # (B, S, d_model)

        # Select content embedding based on token type
        is_word = (token_types == 1).unsqueeze(-1)  # (B, S, 1)
        content = torch.where(is_word, word_emb, gaze_emb)  # (B, S, d_model)

        # ── Add positional and type embeddings ──
        pos = self.pos_enc(frame_indices)          # (B, S, d_model)
        typ = self.type_emb(token_types)           # (B, S, d_model)
        x = content + pos + typ

        x = self.input_norm(x)
        x = self.input_dropout(x)

        # ── Transformer ──
        # Convert attention_mask: True=valid → key_padding_mask needs True=PADDED
        key_padding_mask = ~attention_mask  # (B, S)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # ── Classify every position ──
        logits = self.classifier(x).squeeze(-1)  # (B, S)

        return logits


def extract_frame_logits(
    logits: torch.Tensor,        # (B, S)
    is_gaze_token: torch.Tensor, # (B, S) bool
    frame_indices: torch.Tensor, # (B, S) long
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-frame predictions from GAZE token positions.

    Returns:
        frame_logits: (B, max_frames)
        frame_mask:   (B, max_frames) bool — True for valid frames
    """
    B = logits.shape[0]
    frame_logits = torch.full((B, max_frames), float("-inf"),
                              device=logits.device)
    frame_mask = torch.zeros(B, max_frames, dtype=torch.bool,
                             device=logits.device)

    for b in range(B):
        gaze_mask = is_gaze_token[b]  # (S,)
        gaze_positions = gaze_mask.nonzero(as_tuple=True)[0]
        gaze_frame_idx = frame_indices[b][gaze_positions]  # frame indices
        gaze_logit_vals = logits[b][gaze_positions]

        frame_logits[b, gaze_frame_idx] = gaze_logit_vals
        frame_mask[b, gaze_frame_idx] = True

    return frame_logits, frame_mask
