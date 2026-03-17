"""
10-Fold Cross-Validation, metric aggregation, and plot generation for the RCNN.
"""

import copy
import os
import json
import time
from dataclasses import asdict

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RCNN import RCNN
from RCNN_func import (
    TrainingConfig,
    ProteinDataset,
    collate_fn,
    evaluate,
    compute_metrics,
    train_with_early_stopping,
)


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _mean_sem(values):
    a = np.array(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a)))


def _build_model(config: TrainingConfig, device: torch.device) -> RCNN:
    return RCNN(
        input_dim=config.input_dim,
        num_filters=config.num_filters,
        kernel_sizes=config.kernel_sizes,
        rnn_hidden=config.rnn_hidden,
        rnn_layers=config.rnn_layers,
        fc_hidden=config.fc_hidden,
        dropout=config.dropout,
    ).to(device)


# ──────────────────────────────────────────────────────────────────────────
# Plot 1 & 2 — Accuracy / Loss per epoch  (99 % CI)
# ──────────────────────────────────────────────────────────────────────────
def plot_training_curves(fold_histories, fold_early_stops, output_dir):
    """Two figures: accuracy-per-epoch and loss-per-epoch with 99 % CI."""
    n_folds = len(fold_histories)
    max_epoch = max(len(h["train_loss"]) for h in fold_histories)
    z = 2.576  # 99 % CI

    # NaN-padded arrays  [folds × max_epoch]
    arrays = {}
    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        arr = np.full((n_folds, max_epoch), np.nan)
        for i, h in enumerate(fold_histories):
            n = len(h[key])
            arr[i, :n] = h[key]
        arrays[key] = arr

    epochs = np.arange(1, max_epoch + 1)
    sorted_stops = sorted(fold_early_stops)
    early_lines = sorted_stops[:-1]  # 9 lines (longest fold defines x-max)

    for plot_type, t_key, v_key, ylabel in [
        ("accuracy", "train_acc", "val_acc", "Accuracy"),
        ("loss", "train_loss", "val_loss", "Loss"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))

        for key, color, fill, prefix in [
            (t_key, "blue", "lightskyblue", "Train"),
            (v_key, "orange", "navajowhite", "Validation"),
        ]:
            arr = arrays[key]
            mean = np.nanmean(arr, axis=0)
            cnt = np.sum(~np.isnan(arr), axis=0).astype(float)
            std = np.nanstd(arr, axis=0, ddof=1)
            sem = np.where(cnt > 1, std / np.sqrt(cnt), 0.0)
            ci = z * sem

            ax.plot(epochs, mean, color=color, label=f"Mean {prefix} {ylabel}")
            ax.fill_between(
                epochs,
                mean - ci,
                mean + ci,
                color=fill,
                alpha=0.35,
                label=f"{prefix} 99% CI",
            )

        first = True
        for es in early_lines:
            ax.axvline(
                x=es,
                color="gray",
                linestyle="--",
                alpha=0.55,
                label="Early Stopping" if first else None,
            )
            first = False

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per Epoch (10-Fold CV)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        path = os.path.join(output_dir, f"{plot_type}_per_epoch.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────
# Plot 3 — Data efficiency analysis
# ──────────────────────────────────────────────────────────────────────────
def plot_data_efficiency(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    config,
    device,
    output_dir,
):
    """11-point dual-axis plot: accuracy (blue, left) & training time (yellow, right)."""
    fractions = np.round(np.arange(0.0, 1.1, 0.1), 1)  # 0.0 → 1.0
    accuracies = [0.0]
    times_s = [0.0]

    use_cuda = device.type == "cuda"

    test_ds = ProteinDataset(test_embeddings, test_labels)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=4, pin_memory=use_cuda,
    )

    n_total = len(train_embeddings)
    ag_idx = np.where(train_labels == 1)[0]
    nag_idx = np.where(train_labels == 0)[0]
    rng = np.random.RandomState(42)

    for frac in fractions[1:]:
        n_ag = max(1, int(round(frac * len(ag_idx))))
        n_nag = max(1, int(round(frac * len(nag_idx))))
        sel = np.concatenate(
            [
                rng.choice(ag_idx, size=n_ag, replace=False),
                rng.choice(nag_idx, size=n_nag, replace=False),
            ]
        )
        rng.shuffle(sel)

        sub_emb = [train_embeddings[i] for i in sel]
        sub_lab = train_labels[sel]

        # 80 / 20 internal split for early stopping
        n_val = max(1, int(0.2 * len(sel)))
        perm = rng.permutation(len(sel))
        val_i, trn_i = perm[:n_val], perm[n_val:]

        trn_ds = ProteinDataset([sub_emb[i] for i in trn_i], sub_lab[trn_i])
        val_ds = ProteinDataset([sub_emb[i] for i in val_i], sub_lab[val_i])
        trn_loader = DataLoader(
            trn_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,
            num_workers=4, pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=4, pin_memory=use_cuda,
        )

        model = _build_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        t0 = time.time()
        best_state, _, _ = train_with_early_stopping(
            model, trn_loader, val_loader, optimizer, device, config
        )
        elapsed = time.time() - t0

        model.load_state_dict(best_state)
        test_eval = evaluate(model, test_loader, device, config)
        acc = compute_metrics(test_eval["labels"], test_eval["scores"])["accuracy"]

        accuracies.append(acc)
        times_s.append(elapsed)
        print(f"  Fraction {frac:.1f} — Acc: {acc:.4f}  Time: {elapsed:.1f}s")

    # --- dual-axis plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(fractions, accuracies, "o-", color="blue", label="Mean Accuracy")
    ax2.plot(fractions, times_s, "o-", color="goldenrod", label="Training Time (s)")

    ax1.set_xlabel("Dataset Fraction")
    ax1.set_ylabel("Mean Accuracy", color="blue")
    ax2.set_ylabel("Training Time (s)", color="goldenrod")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="goldenrod")

    # Align both curves so (0,0) and (1, ·) coincide visually
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(0, max(accuracies) * 1.12 if max(accuracies) > 0 else 1)
    if accuracies[-1] > 0 and times_s[-1] > 0:
        scale = times_s[-1] / accuracies[-1]
        ax2.set_ylim(0, ax1.get_ylim()[1] * scale)
    else:
        ax2.set_ylim(0, max(times_s) * 1.12 if max(times_s) > 0 else 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("Data Efficiency Analysis")
    ax1.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "data_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────
# Main CV routine
# ──────────────────────────────────────────────────────────────────────────
def cross_validate_rcnn(
    train_embeddings: list,
    train_labels: np.ndarray,
    test_embeddings: list,
    test_labels: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    output_dir: str,
) -> dict:
    """
    10-fold stratified CV on the training set.
    Each fold's best model is also evaluated on the held-out test set.
    """
    os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)

    fold_val_metrics: list[dict] = []
    fold_test_metrics: list[dict] = []
    fold_histories: list[dict] = []
    fold_early_stops: list[int] = []

    best_fold_score = -1.0
    best_fold_idx = -1
    best_fold_state = None
    best_fold_val_metrics = None
    best_fold_test_metrics = None

    use_cuda = device.type == "cuda"

    test_ds = ProteinDataset(test_embeddings, test_labels)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=4, pin_memory=use_cuda,
    )

    n_ag = int(train_labels.sum())
    n_nag = len(train_labels) - n_ag

    use_cuda = device.type == "cuda"

    print(f"\n{'=' * 80}")
    print("RCNN  —  10-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'=' * 80}")
    print(f"  Device          : {device}")
    print(f"  Train proteins  : {len(train_embeddings)}  (AG={n_ag}, NAG={n_nag})")
    print(f"  Test proteins   : {len(test_embeddings)}")
    print(f"  Folds           : {config.num_folds}")
    print(f"  Epochs (max)    : {config.epochs}")
    print(f"  Patience        : {config.patience}")
    print(f"{'=' * 80}\n")

    for fold_idx, (trn_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(train_labels)), train_labels), start=1
    ):
        print(f"── Fold {fold_idx}/{config.num_folds} "
              f"(train={len(trn_idx)}, val={len(val_idx)}) ──")

        trn_ds = ProteinDataset(
            [train_embeddings[i] for i in trn_idx], train_labels[trn_idx]
        )
        val_ds = ProteinDataset(
            [train_embeddings[i] for i in val_idx], train_labels[val_idx]
        )

        trn_loader = DataLoader(
            trn_ds,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=use_cuda,
        )

        model = _build_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        best_state, history, early_epoch = train_with_early_stopping(
            model, trn_loader, val_loader, optimizer, device, config
        )
        fold_histories.append(history)
        fold_early_stops.append(early_epoch)

        model.load_state_dict(best_state)

        val_eval = evaluate(model, val_loader, device, config)
        val_m = compute_metrics(val_eval["labels"], val_eval["scores"])
        fold_val_metrics.append(val_m)

        test_eval = evaluate(model, test_loader, device, config)
        test_m = compute_metrics(test_eval["labels"], test_eval["scores"])
        fold_test_metrics.append(test_m)

        print(f"    Val  → Acc={val_m['accuracy']:.4f}  AUC={val_m['auc_roc']:.4f}  "
              f"MCC={val_m['mcc']:.4f}")
        print(f"    Test → Acc={test_m['accuracy']:.4f}  AUC={test_m['auc_roc']:.4f}  "
              f"MCC={test_m['mcc']:.4f}\n")

        # Track best fold by (recall + specificity + accuracy) on validation
        fold_score = val_m["recall"] + val_m["specificity"] + val_m["accuracy"]
        if fold_score > best_fold_score:
            best_fold_score = fold_score
            best_fold_idx = fold_idx
            best_fold_state = copy.deepcopy(best_state)
            best_fold_val_metrics = val_m
            best_fold_test_metrics = test_m

    # ── Aggregate metrics ────────────────────────────────────────────────
    metric_names = ["accuracy", "precision", "recall", "specificity", "mcc", "auc_roc"]

    avg_val, sem_val = {}, {}
    avg_test, sem_test = {}, {}
    for m in metric_names:
        vals = [d[m] for d in fold_val_metrics]
        tests = [d[m] for d in fold_test_metrics]
        avg_val[m], sem_val[m] = _mean_sem(vals)
        avg_test[m], sem_test[m] = _mean_sem(tests)

    print(f"\n{'=' * 80}")
    print("AGGREGATE RESULTS  (mean ± SEM)")
    print(f"{'=' * 80}")
    for m in metric_names:
        print(f"  {m:<14s}  Val {avg_val[m]:.4f}±{sem_val[m]:.4f}  |  "
              f"Test {avg_test[m]:.4f}±{sem_test[m]:.4f}")
    print(f"{'=' * 80}\n")

    # ── Save best model ──────────────────────────────────────────────────
    model_path = os.path.join(output_dir, "best_model.pt")
    torch.save(best_fold_state, model_path)
    print(f"  Best fold: {best_fold_idx}  "
          f"(Recall={best_fold_val_metrics['recall']:.4f}  "
          f"Spec={best_fold_val_metrics['specificity']:.4f}  "
          f"Acc={best_fold_val_metrics['accuracy']:.4f})")
    print(f"  Saved best model weights → {model_path}")

    best_metrics_path = os.path.join(output_dir, "best_model_metrics.json")
    with open(best_metrics_path, "w") as f:
        json.dump({
            "best_fold": best_fold_idx,
            "selection_criteria": "recall + specificity + accuracy",
            "selection_score": best_fold_score,
            "val_metrics": best_fold_val_metrics,
            "test_metrics": best_fold_test_metrics,
        }, f, indent=2)
    print(f"  Saved best model metrics → {best_metrics_path}")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "config": asdict(config),
        "average_val_metrics": avg_val,
        "sem_val_metrics": sem_val,
        "average_test_metrics": avg_test,
        "sem_test_metrics": sem_test,
        "fold_val_metrics": fold_val_metrics,
        "fold_test_metrics": fold_test_metrics,
        "fold_early_stops": fold_early_stops,
    }

    json_path = os.path.join(output_dir, "CV_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {json_path}")

    # ── Save readable .txt report ────────────────────────────────────────
    txt_path = os.path.join(output_dir, "CV_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RCNN  —  10-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("VALIDATION SET  (mean ± SEM over folds)\n")
        f.write("-" * 45 + "\n")
        for m in metric_names:
            f.write(f"  {m:<14s}  {avg_val[m]:.4f} ± {sem_val[m]:.4f}\n")

        f.write(f"\nTEST SET  (mean ± SEM over folds)\n")
        f.write("-" * 45 + "\n")
        for m in metric_names:
            f.write(f"  {m:<14s}  {avg_test[m]:.4f} ± {sem_test[m]:.4f}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write(f"BEST MODEL  (fold {best_fold_idx}, "
                f"selected by recall + specificity + accuracy = {best_fold_score:.4f})\n")
        f.write("=" * 70 + "\n\n")

        f.write("  Validation metrics:\n")
        for m in metric_names:
            f.write(f"    {m:<14s}  {best_fold_val_metrics[m]:.4f}\n")

        f.write("\n  Test metrics:\n")
        for m in metric_names:
            f.write(f"    {m:<14s}  {best_fold_test_metrics[m]:.4f}\n")

        f.write(f"\n{'=' * 70}\n")
        f.write("PER-FOLD DETAILS\n")
        f.write("=" * 70 + "\n\n")
        for i in range(config.num_folds):
            f.write(f"  Fold {i+1}  (early stop epoch {fold_early_stops[i]})\n")
            f.write("    Validation: "
                    + "  ".join(f"{m}={fold_val_metrics[i][m]:.4f}" for m in metric_names)
                    + "\n")
            f.write("    Test:       "
                    + "  ".join(f"{m}={fold_test_metrics[i][m]:.4f}" for m in metric_names)
                    + "\n\n")

    print(f"  Saved {txt_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    plot_training_curves(fold_histories, fold_early_stops, output_dir)

    print("\n  Running data-efficiency analysis …")
    plot_data_efficiency(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        config, device, output_dir,
    )

    return results
    print(f"{'=' * 80}")