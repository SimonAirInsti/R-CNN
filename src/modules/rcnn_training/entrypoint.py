from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.modules.rcnn_model.entrypoint import RCNN, build_model_for_variant


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
    num_workers: int
    pin_memory: bool
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
            num_workers=0,
            pin_memory=False,
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
            num_workers=training_cfg.get("num_workers", cls.defaults().num_workers),
            pin_memory=training_cfg.get("pin_memory", cls.defaults().pin_memory),
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


def resolve_project_path(project_root: str, path_value: str) -> str:
    """Resolve a project-relative path against the repository root."""
    path = str(path_value)
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return path
    return str((project_root / path).resolve())


def load_training_data_from_config(config: dict, project_root: str) -> tuple[list, np.ndarray, list, np.ndarray]:
    """Load the antigenic and non-antigenic data splits from the YAML config."""
    paths = config.get("paths", {})
    project_root_path = project_root if hasattr(project_root, "__fspath__") else __import__("pathlib").Path(project_root)

    train_ag = load_embeddings(resolve_project_path(project_root_path, paths["train_ag"]))
    train_nag = load_embeddings(resolve_project_path(project_root_path, paths["train_nag"]))
    test_ag = load_embeddings(resolve_project_path(project_root_path, paths["test_ag"]))
    test_nag = load_embeddings(resolve_project_path(project_root_path, paths["test_nag"]))

    train_embeddings = train_ag + train_nag
    train_labels = np.array([1] * len(train_ag) + [0] * len(train_nag))
    test_embeddings = test_ag + test_nag
    test_labels = np.array([1] * len(test_ag) + [0] * len(test_nag))
    return train_embeddings, train_labels, test_embeddings, test_labels


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
    return build_model_for_variant(config, "base")


def build_variant_config(config: dict, variant_name: str) -> dict:
    """Return a configuration dictionary for a given architecture variant."""
    if variant_name == "base":
        model_cfg = config.get("model", {}).get("base", {})
    else:
        candidates = config.get("model", {}).get("candidate_architectures", {})
        if variant_name not in candidates:
            raise ValueError(
                f"Unknown variant '{variant_name}'. Available ones: {list(candidates.keys()) + ['base']}"
            )
        model_cfg = candidates[variant_name]

    return {
        "model": {"base": model_cfg},
        "training": config.get("training", {}),
    }


def _plot_architecture_comparison(results: list[dict], output_dir: str) -> str:
    """Create a compact bar chart of test metrics across model variants."""
    metric_names = ["accuracy", "precision", "recall", "specificity", "mcc", "auc_roc"]
    variants = [item["variant"] for item in results]
    values_by_metric = {
        metric: [item["test_metrics"][metric] for item in results]
        for metric in metric_names
    }

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 2.5 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_names):
        ax.bar(variants, values_by_metric[metric], color="steelblue")
        ax.set_title(f"Test {metric}")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.05 if metric != "mcc" else 1.10)
        ax.grid(axis="y", alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)

    fig.tight_layout()
    path = os.path.join(output_dir, "architecture_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_metrics_table(results: list[dict], output_dir: str) -> dict[str, str]:
    """Export a direct comparison table with models as rows and metrics as columns."""
    metric_names = ["accuracy", "precision", "recall", "specificity", "mcc", "auc_roc"]

    csv_path = os.path.join(output_dir, "architecture_comparison_table.csv")
    md_path = os.path.join(output_dir, "architecture_comparison_table.md")

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("variant," + ",".join(metric_names) + "\n")
        for item in results:
            row = [item["variant"]]
            row.extend(f"{item['test_metrics'][metric]:.6f}" for metric in metric_names)
            fh.write(",".join(row) + "\n")

    winner_by_metric = {}
    for metric in metric_names:
        best_value = max(item["test_metrics"][metric] for item in results)
        winner_by_metric[metric] = [
            item["variant"] for item in results if item["test_metrics"][metric] == best_value
        ]

    md_lines = [
        "| model | accuracy | precision | recall | specificity | mcc | auc_roc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        metrics = item["test_metrics"]
        formatted = []
        for metric in metric_names:
            value = metrics[metric]
            if item["variant"] in winner_by_metric.get(metric, []):
                formatted.append(f"**{value:.4f}**")
            else:
                formatted.append(f"{value:.4f}")
        md_lines.append("| " + item["variant"] + " | " + " | ".join(formatted) + " |")

    md_lines.append("")
    md_lines.append("### Best model by metric")
    for metric in metric_names:
        winners = ", ".join(winner_by_metric[metric])
        md_lines.append(f"- **{metric}**: {winners}")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    return {"csv_path": csv_path, "md_path": md_path}


def compare_rcnn_architectures(
    train_embeddings: list,
    train_labels: np.ndarray,
    test_embeddings: list,
    test_labels: np.ndarray,
    config: dict,
    device: torch.device,
    output_dir: str,
    variant_names: list[str] | None = None,
) -> dict:
    """Train and validate a small set of RCNN variants from the YAML config.

    This intentionally keeps the comparison simple and readable: the base model and
    the extra variants declared under `model.candidate_architectures` are trained
    with the same split/evaluation logic and then compared with key metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    candidate_variants = list(config.get("model", {}).get("candidate_architectures", {}).keys())
    ordered_variants = ["base"] + candidate_variants if variant_names is None else list(variant_names)
    ordered_variants = list(dict.fromkeys(ordered_variants))

    summary: list[dict] = []

    for variant_name in ordered_variants:
        variant_cfg = build_variant_config(config, variant_name)
        training_cfg = TrainingConfig.from_dict(variant_cfg)

        idx = np.arange(len(train_labels))
        trn_idx, val_idx = train_test_split(
            idx,
            test_size=0.2,
            stratify=train_labels,
            random_state=42,
        )

        trn_ds = ProteinDataset([train_embeddings[i] for i in trn_idx], train_labels[trn_idx])
        val_ds = ProteinDataset([train_embeddings[i] for i in val_idx], train_labels[val_idx])

        use_cuda = device.type == "cuda"
        pin_memory = bool(training_cfg.pin_memory and use_cuda)
        num_workers = max(0, int(training_cfg.num_workers))

        trn_loader = DataLoader(
            trn_ds,
            batch_size=training_cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_ds = ProteinDataset(test_embeddings, test_labels)
        test_loader = DataLoader(
            test_ds,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model = build_model_for_variant(config, variant_name).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )

        best_state, history, early_epoch = train_with_early_stopping(
            model, trn_loader, val_loader, optimizer, device, training_cfg
        )
        model.load_state_dict(best_state)

        test_eval = evaluate(model, test_loader, device, training_cfg)
        test_metrics = compute_metrics(test_eval["labels"], test_eval["scores"])

        model_path = os.path.join(output_dir, f"{variant_name}_best_model.pt")
        torch.save(best_state, model_path)

        summary.append(
            {
                "variant": variant_name,
                "early_stop_epoch": early_epoch,
                "history": history,
                "val_metrics": compute_metrics(
                    evaluate(model, val_loader, device, training_cfg)["labels"],
                    evaluate(model, val_loader, device, training_cfg)["scores"],
                ),
                "test_metrics": test_metrics,
                "model_path": model_path,
            }
        )

        print(f"Variant {variant_name}: test_acc={test_metrics['accuracy']:.4f}, auc={test_metrics['auc_roc']:.4f}")

    comparison_path = os.path.join(output_dir, "architecture_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as fh:
        json.dump({"variants": summary}, fh, indent=2)

    table_paths = _write_metrics_table(summary, output_dir)
    chart_path = _plot_architecture_comparison(summary, output_dir)
    print(f"Saved architecture comparison summary: {comparison_path}")
    print(f"Saved architecture comparison CSV: {table_paths['csv_path']}")
    print(f"Saved architecture comparison markdown table: {table_paths['md_path']}")
    print(f"Saved architecture comparison chart: {chart_path}")
    return {
        "variants": summary,
        "chart_path": chart_path,
        "table_csv": table_paths["csv_path"],
        "table_md": table_paths["md_path"],
    }

