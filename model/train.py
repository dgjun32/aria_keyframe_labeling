"""
train.py — Leave-One-Out Cross-Validation Training for Keyframe Detection
==========================================================================

Usage:
    python -m model.train [--epochs 200] [--lr 1e-3] [--device cpu]

Strategy:
    - 5 episodes → 5-fold LOO-CV (train on 4, validate on 1)
    - DeBERTa embeddings extracted once at dataset init (frozen)
    - BCE loss with pos_weight to handle class imbalance (~8.7% positive)
    - AdamW optimizer with cosine annealing LR scheduler
    - Reports per-fold and aggregate metrics (F1, precision, recall, AUROC)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

from model.dataset import KeyframeDataset, WordEmbeddingExtractor, collate_fn
from model.model import KeyframeTransformer, extract_frame_logits


# ── All episodes ──────────────────────────────────────────────────────────
ALL_EPISODES = [
    "Home_dongjun_test_1",
    "Home_dongjun_test_1_2",
    "Home_dongjun_test_1_3",
    "Home_dongjun_test_1_4",
    "Home_dongjun_test_1_5",
]


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(
    all_labels: np.ndarray, all_probs: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute classification metrics.

    Args:
        all_labels: (N,) binary ground truth
        all_probs:  (N,) predicted probabilities
        threshold:  classification threshold

    Returns:
        dict with f1, precision, recall, auroc, num_pos, num_neg
    """
    preds = (all_probs >= threshold).astype(int)
    labels_int = all_labels.astype(int)

    num_pos = int(labels_int.sum())
    num_neg = int(len(labels_int) - num_pos)

    metrics = {
        "num_pos": num_pos,
        "num_neg": num_neg,
        "threshold": threshold,
    }

    # Handle edge cases (all same class)
    if num_pos == 0 or num_neg == 0:
        metrics["f1"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["auroc"] = 0.0
        return metrics

    metrics["f1"] = f1_score(labels_int, preds, zero_division=0)
    metrics["precision"] = precision_score(labels_int, preds, zero_division=0)
    metrics["recall"] = recall_score(labels_int, preds, zero_division=0)

    try:
        metrics["auroc"] = roc_auc_score(labels_int, all_probs)
    except ValueError:
        metrics["auroc"] = 0.0

    return metrics


# ── Training one fold ─────────────────────────────────────────────────────
def train_one_fold(
    fold_idx: int,
    train_episodes: list[str],
    val_episodes: list[str],
    preproc_dir: str,
    word_extractor: WordEmbeddingExtractor,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
) -> dict:
    """Train and evaluate one LOO-CV fold.

    Returns:
        dict with training history and best validation metrics.
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/5 — Val episode: {val_episodes[0]}")
    print(f"Train episodes: {train_episodes}")
    print(f"{'='*60}")

    # ── Build datasets ──
    train_ds = KeyframeDataset(preproc_dir, train_episodes, word_extractor)
    val_ds = KeyframeDataset(preproc_dir, val_episodes, word_extractor)

    train_loader = DataLoader(
        train_ds, batch_size=len(train_ds), shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # ── Compute pos_weight from training data ──
    total_pos = 0
    total_neg = 0
    for seq in train_ds.episodes:
        labels = seq["labels"]
        total_pos += labels.sum()
        total_neg += (1 - labels).sum()

    if total_pos > 0:
        pos_weight_val = total_neg / total_pos
    else:
        pos_weight_val = 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"  pos_weight = {pos_weight_val:.2f}  "
          f"(pos={int(total_pos)}, neg={int(total_neg)})")

    # ── Build model ──
    model = KeyframeTransformer(
        gaze_dim=2,
        word_dim=768,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_frames=256,
    ).to(device)

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── Training loop ──
    best_val_f1 = -1.0
    best_val_metrics = {}
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auroc": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                gaze_features=batch["gaze_features"],
                word_features=batch["word_features"],
                token_types=batch["token_types"],
                frame_indices=batch["frame_indices"],
                is_gaze_token=batch["is_gaze_token"],
                attention_mask=batch["attention_mask"],
            )  # (B, S)

            # Extract per-frame logits
            max_frames = batch["labels"].shape[1]
            frame_logits, frame_mask = extract_frame_logits(
                logits,
                batch["is_gaze_token"],
                batch["frame_indices"],
                max_frames,
            )

            # Compute loss only on valid frames
            valid = frame_mask & batch["label_mask"]
            if valid.sum() == 0:
                continue

            loss = nn.functional.binary_cross_entropy_with_logits(
                frame_logits[valid],
                batch["labels"][valid],
                pos_weight=pos_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # ── Validate ──
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auroc"].append(val_metrics["auroc"])

        # ── Track best ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # ── Log ──
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_F1={val_metrics['f1']:.3f}  "
                f"val_AUROC={val_metrics['auroc']:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    print(f"\n  Best epoch: {best_epoch}  F1={best_val_f1:.3f}")
    print(f"  Best metrics: {best_val_metrics}")

    return {
        "fold": fold_idx,
        "val_episode": val_episodes[0],
        "best_epoch": best_epoch,
        "best_metrics": best_val_metrics,
        "best_state": best_state,
        "history": history,
    }


# ── Evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict:
    """Evaluate model on a DataLoader.

    Returns:
        dict with loss, f1, precision, recall, auroc
    """
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(
            gaze_features=batch["gaze_features"],
            word_features=batch["word_features"],
            token_types=batch["token_types"],
            frame_indices=batch["frame_indices"],
            is_gaze_token=batch["is_gaze_token"],
            attention_mask=batch["attention_mask"],
        )

        max_frames = batch["labels"].shape[1]
        frame_logits, frame_mask = extract_frame_logits(
            logits,
            batch["is_gaze_token"],
            batch["frame_indices"],
            max_frames,
        )

        valid = frame_mask & batch["label_mask"]
        if valid.sum() == 0:
            continue

        loss = nn.functional.binary_cross_entropy_with_logits(
            frame_logits[valid],
            batch["labels"][valid],
            pos_weight=pos_weight,
        )
        total_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(frame_logits[valid]).cpu().numpy()
        labels = batch["labels"][valid].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    if not all_labels:
        return {"loss": 0, "f1": 0, "precision": 0, "recall": 0, "auroc": 0}

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(num_batches, 1)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train keyframe detection with LOO-CV"
    )
    parser.add_argument("--preproc-dir", type=str,
                        default="./preproc_files",
                        help="Directory with preprocessed episode data")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="AdamW weight decay")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer hidden dimension")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=256,
                        help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--save-dir", type=str, default="./model/checkpoints",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Pre-load DeBERTa (shared across all folds) ──
    print("Loading DeBERTa-v3-base for word embeddings...")
    t0 = time.time()
    word_extractor = WordEmbeddingExtractor(
        model_name="microsoft/deberta-v3-base", device="cpu"
    )
    print(f"  DeBERTa loaded in {time.time() - t0:.1f}s")

    # ── LOO Cross-Validation ──
    fold_results = []

    for fold_idx, val_ep in enumerate(ALL_EPISODES):
        train_eps = [e for e in ALL_EPISODES if e != val_ep]

        result = train_one_fold(
            fold_idx=fold_idx,
            train_episodes=train_eps,
            val_episodes=[val_ep],
            preproc_dir=args.preproc_dir,
            word_extractor=word_extractor,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
        fold_results.append(result)

        # Save best checkpoint for this fold
        if result["best_state"] is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(
                args.save_dir, f"fold{fold_idx}_{val_ep}.pt"
            )
            torch.save(
                {
                    "fold": fold_idx,
                    "val_episode": val_ep,
                    "best_epoch": result["best_epoch"],
                    "best_metrics": result["best_metrics"],
                    "model_state_dict": result["best_state"],
                    "model_config": {
                        "d_model": args.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                    },
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint → {ckpt_path}")

    # ── Aggregate results ──
    print(f"\n{'='*60}")
    print("LOO-CV Results Summary")
    print(f"{'='*60}")

    f1_scores = []
    auroc_scores = []
    precision_scores = []
    recall_scores = []

    for r in fold_results:
        m = r["best_metrics"]
        ep = r["val_episode"]
        print(
            f"  {ep:30s}  "
            f"F1={m.get('f1', 0):.3f}  "
            f"P={m.get('precision', 0):.3f}  "
            f"R={m.get('recall', 0):.3f}  "
            f"AUROC={m.get('auroc', 0):.3f}  "
            f"(best@{r['best_epoch']})"
        )
        f1_scores.append(m.get("f1", 0))
        auroc_scores.append(m.get("auroc", 0))
        precision_scores.append(m.get("precision", 0))
        recall_scores.append(m.get("recall", 0))

    print(f"\n  Mean F1:        {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")
    print(f"  Mean Precision: {np.mean(precision_scores):.3f} +/- {np.std(precision_scores):.3f}")
    print(f"  Mean Recall:    {np.mean(recall_scores):.3f} +/- {np.std(recall_scores):.3f}")
    print(f"  Mean AUROC:     {np.mean(auroc_scores):.3f} +/- {np.std(auroc_scores):.3f}")

    # Save summary
    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, "loo_cv_summary.json")
    summary = {
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
        },
        "folds": [
            {
                "val_episode": r["val_episode"],
                "best_epoch": r["best_epoch"],
                "best_metrics": {
                    k: float(v) for k, v in r["best_metrics"].items()
                },
            }
            for r in fold_results
        ],
        "aggregate": {
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
            "mean_precision": float(np.mean(precision_scores)),
            "std_precision": float(np.std(precision_scores)),
            "mean_recall": float(np.mean(recall_scores)),
            "std_recall": float(np.std(recall_scores)),
            "mean_auroc": float(np.mean(auroc_scores)),
            "std_auroc": float(np.std(auroc_scores)),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
