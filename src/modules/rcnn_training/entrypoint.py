from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset

from src.modules.rcnn_model.entrypoint import RCNN


@dataclass
class TrainingConfig:
    """Training configuration for the RCNN pipeline.

    The canonical source of truth is config/rcnn_config.yaml. This dataclass is
    the typed representation used by the training entrypoint, and it should be
    populated from the YAML file via `from_dict(...)`.
    """

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_folds: int
    input_dim: int
    num_filters: int
    kernel_sizes: list[int]
    rnn_hidden: int
    rnn_layers: int
    fc_hidden: int
    dropout: float
    patience: int
    weight_antigenic: float
    weight_nonantigenic: float

    @classmethod
    def defaults(cls) -> "TrainingConfig":
        """Explicit fallback values only for safety and compatibility.

        In the project architecture, the YAML config is the primary source of
        truth and should always be used when available.
        """
        return cls(
            epochs=100,
            batch_size=32,
            learning_rate=1e-4,
            weight_decay=1e-5,
            num_folds=10,
            input_dim=1280,
            num_filters=128,
            kernel_sizes=[3, 5, 7],
            rnn_hidden=128,
            rnn_layers=2,
            fc_hidden=128,
            dropout=0.3,
            patience=15,
            weight_antigenic=0.64,
            weight_nonantigenic=2.24,
        )

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        """Build a typed configuration from the project YAML.

        This is the canonical constructor for the training pipeline.
        """
        model_cfg = config.get("model", {}).get("base", {})
        training_cfg = config.get("training", {})

        return cls(
            epochs=training_cfg.get("epochs", cls.defaults().epochs),
            batch_size=training_cfg.get("batch_size", cls.defaults().batch_size),
            learning_rate=training_cfg.get("learning_rate", cls.defaults().learning_rate),
            weight_decay=training_cfg.get("weight_decay", cls.defaults().weight_decay),
            num_folds=training_cfg.get("num_folds", cls.defaults().num_folds),
            input_dim=model_cfg.get("input_dim", cls.defaults().input_dim),
            num_filters=model_cfg.get("num_filters", cls.defaults().num_filters),
            kernel_sizes=model_cfg.get("kernel_sizes", cls.defaults().kernel_sizes),
            rnn_hidden=model_cfg.get("rnn_hidden", cls.defaults().rnn_hidden),
            rnn_layers=model_cfg.get("rnn_layers", cls.defaults().rnn_layers),
            fc_hidden=model_cfg.get("fc_hidden", cls.defaults().fc_hidden),
            dropout=model_cfg.get("dropout", cls.defaults().dropout),
            patience=training_cfg.get("patience", cls.defaults().patience),
            weight_antigenic=training_cfg.get("weight_antigenic", cls.defaults().weight_antigenic),
            weight_nonantigenic=training_cfg.get("weight_nonantigenic", cls.defaults().weight_nonantigenic),
        )


class ProteinDataset(Dataset):
    def __init__(self, embeddings: list, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb, self.labels[idx]


def collate_fn(batch):
    embeddings, labels = zip(*batch)
    lengths = torch.tensor([emb.size(0) for emb in embeddings], dtype=torch.long)
    max_len = lengths.max().item()
    embed_dim = embeddings[0].size(-1)

    padded = torch.zeros(len(embeddings), max_len, embed_dim)
    for i, emb in enumerate(embeddings):
        padded[i, : emb.size(0), :] = emb

    labels = torch.tensor(np.array(labels), dtype=torch.float)
    return padded, labels, lengths


def load_embeddings(pt_path: str) -> list:
    data = torch.load(pt_path, weights_only=False)

    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, torch.Tensor):
                out.append(item.float())
            elif isinstance(item, dict):
                for key in (
                    "embeddings",
                    "embedding",
                    "representation",
                    "mean_representations",
                    "avg",
                ):
                    if key in item:
                        t = item[key]
                        if isinstance(t, dict):
                            t = list(t.values())[-1]
                        out.append(t.float())
                        break
        if out:
            return out

    if isinstance(data, dict):
        if "embeddings" in data:
            t = data["embeddings"]
            if isinstance(t, torch.Tensor):
                return [t[i].float() for i in range(t.size(0))]
        tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
        if tensors:
            return [t.float() for t in tensors]

    if isinstance(data, torch.Tensor):
        if data.dim() >= 2:
            return [data[i].float() for i in range(data.size(0))]

    raise ValueError(f"Could not parse embeddings from {pt_path}.")


def weighted_bce_loss(logits: torch.Tensor, targets: torch.Tensor, w_pos: float, w_neg: float) -> torch.Tensor:
    weights = torch.where(targets == 1, w_pos, w_neg)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, lengths)
        loss = weighted_bce_loss(logits, y, config.weight_antigenic, config.weight_nonantigenic)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_scores = [], []
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits = model(x, lengths)
        loss = weighted_bce_loss(logits, y, config.weight_antigenic, config.weight_nonantigenic)
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


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
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


def train_with_early_stopping(model, train_loader, val_loader, optimizer, device, config):
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    early_stop_epoch = config.epochs

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, config)
        train_eval = evaluate(model, train_loader, device, config)
        val_eval = evaluate(model, val_loader, device, config)

        train_metrics = compute_metrics(train_eval["labels"], train_eval["scores"])
        val_metrics = compute_metrics(val_eval["labels"], val_eval["scores"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_eval["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            early_stop_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return best_state, history, early_stop_epoch


def build_model_from_config(config: dict) -> RCNN:
    """Build the RCNN model strictly from the project YAML config."""
    model_cfg = config.get("model", {}).get("base", {})
    return RCNN(
        input_dim=model_cfg.get("input_dim", TrainingConfig.defaults().input_dim),
        num_filters=model_cfg.get("num_filters", TrainingConfig.defaults().num_filters),
        kernel_sizes=model_cfg.get("kernel_sizes", TrainingConfig.defaults().kernel_sizes),
        rnn_hidden=model_cfg.get("rnn_hidden", TrainingConfig.defaults().rnn_hidden),
        rnn_layers=model_cfg.get("rnn_layers", TrainingConfig.defaults().rnn_layers),
        fc_hidden=model_cfg.get("fc_hidden", TrainingConfig.defaults().fc_hidden),
        dropout=model_cfg.get("dropout", TrainingConfig.defaults().dropout),
    )
