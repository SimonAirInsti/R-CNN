"""
Training configuration, data loading, loss, training / evaluation functions for the RCNN.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_folds: int = 10
    # Model hyper-parameters
    input_dim: int = 1280
    num_filters: int = 128
    kernel_sizes: list = field(default_factory=lambda: [3, 5, 7])
    rnn_hidden: int = 128
    rnn_layers: int = 2
    fc_hidden: int = 128
    dropout: float = 0.3
    # Early stopping
    patience: int = 15
    # Per-class weights for weighted BCE
    weight_antigenic: float = 0.64
    weight_nonantigenic: float = 2.24


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class ProteinDataset(Dataset):
    """Holds a list of per-residue embedding tensors and their binary labels."""

    def __init__(self, embeddings: list, labels: np.ndarray):
        """
        Args:
            embeddings: list of tensors, each [L_i, D]   (per-residue)
                        or [D] if mean-pooled (will be unsqueezed to [1, D])
            labels:     numpy array of 0/1
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)          # [D] → [1, D]
        return emb, self.labels[idx]


def collate_fn(batch):
    """Zero-pad variable-length embeddings to the longest sequence in the batch."""
    embeddings, labels = zip(*batch)
    lengths = torch.tensor([emb.size(0) for emb in embeddings], dtype=torch.long)
    max_len = lengths.max().item()
    embed_dim = embeddings[0].size(-1)

    padded = torch.zeros(len(embeddings), max_len, embed_dim)
    for i, emb in enumerate(embeddings):
        padded[i, : emb.size(0), :] = emb

    labels = torch.tensor(np.array(labels), dtype=torch.float)
    return padded, labels, lengths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_embeddings(pt_path: str) -> list:
    """
    Load ESM-2 embeddings from a .pt file.

    Supported formats (auto-detected):
      • list of tensors  — one tensor per protein
      • list of dicts    — each with an 'embeddings' (or similar) key
      • dict  with key 'embeddings' holding [N, L, D] or [N, D] tensor
      • dict  mapping ids → tensors
      • single tensor [N, L, D] or [N, D]

    Returns:
        List of float tensors (one per protein).
    """
    data = torch.load(pt_path, weights_only=False)

    # --- list -----------------------------------------------------------
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, torch.Tensor):
                out.append(item.float())
            elif isinstance(item, dict):
                for key in ("embeddings", "embedding", "representation",
                            "mean_representations", "avg"):
                    if key in item:
                        t = item[key]
                        if isinstance(t, dict):          # layer-indexed
                            t = list(t.values())[-1]     # last layer
                        out.append(t.float())
                        break
        if out:
            return out

    # --- dict -----------------------------------------------------------
    if isinstance(data, dict):
        if "embeddings" in data:
            t = data["embeddings"]
            if isinstance(t, torch.Tensor):
                return [t[i].float() for i in range(t.size(0))]
        # mapping id → tensor
        tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
        if tensors:
            return [t.float() for t in tensors]

    # --- single tensor --------------------------------------------------
    if isinstance(data, torch.Tensor):
        if data.dim() >= 2:
            return [data[i].float() for i in range(data.size(0))]

    raise ValueError(
        f"Could not parse embeddings from {pt_path} (type={type(data).__name__}). "
        "Adjust load_embeddings() to match your file format."
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_pos: float,
    w_neg: float,
) -> torch.Tensor:
    """BCE with per-class sample weights (antigenic=w_pos, non-antigenic=w_neg)."""
    weights = torch.where(targets == 1, w_pos, w_neg)
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weights
    )


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = weighted_bce_loss(
            logits, y, config.weight_antigenic, config.weight_nonantigenic
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, config):
    """Return dict with 'loss', 'labels' (np), 'scores' (np)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_scores = [], []
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits = model(x, lengths)
        loss = weighted_bce_loss(
            logits, y, config.weight_antigenic, config.weight_nonantigenic
        )
        if torch.isfinite(loss):
            total_loss += loss.item()
            n += 1
        all_labels.append(y.cpu())
        all_scores.append(torch.sigmoid(logits).cpu())

    return {
        "loss": total_loss / max(1, n),
        "labels": torch.cat(all_labels).numpy(),
        "scores": torch.cat(all_scores).numpy(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, Precision, Recall, Specificity, MCC, AUC-ROC."""
    preds = (scores >= threshold).astype(int)

    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    specificity = tn / max(1, tn + fp)

    try:
        auc_roc = roc_auc_score(labels, scores)
    except ValueError:
        auc_roc = 0.0

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "auc_roc": float(auc_roc),
    }


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------
def train_with_early_stopping(model, train_loader, val_loader, optimizer, device, config):
    """
    Returns:
        best_model_state, history dict, early_stop_epoch
    """
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stop_epoch = config.epochs

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)

        train_eval = evaluate(model, train_loader, device, config)
        val_eval = evaluate(model, val_loader, device, config)

        train_acc = compute_metrics(train_eval["labels"], train_eval["scores"])["accuracy"]
        val_acc = compute_metrics(val_eval["labels"], val_eval["scores"])["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_eval["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                early_stop_epoch = epoch
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch:>3d}/{config.epochs} | "
                f"Train Loss {train_loss:.4f} | Val Loss {val_eval['loss']:.4f} | "
                f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
                f"Patience {patience_counter}/{config.patience}"
            )

    return best_state, history, early_stop_epoch