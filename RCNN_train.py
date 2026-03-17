"""
RCNN_train.py — Train the RCNN via 10-fold CV on ESM-2 per-residue embeddings.

Usage:
    cd <PROJECT_ROOT>
    python src/models/RCNN/RCNN_train.py
"""

import os
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "models", "RCNN"))

from RCNN_func import TrainingConfig, load_embeddings
from RCNN_CV import cross_validate_rcnn

if __name__ == '__main__':
    print(f"Project root directory: {PROJECT_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------
    DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")

    train_ag  = load_embeddings(os.path.join(DATA_DIR, "Table S1. Training set of viral antigens.fasta.pt"))
    train_nag = load_embeddings(os.path.join(DATA_DIR, "Table S2. Training set of viral nonantigens.fasta.pt"))
    test_ag   = load_embeddings(os.path.join(DATA_DIR, "Table S3. Test set of viral antigens.fasta.pt"))
    test_nag  = load_embeddings(os.path.join(DATA_DIR, "Table S4. Test set of viral nonantigens.fasta.pt"))

    train_embeddings = train_ag + train_nag
    train_labels     = np.array([1] * len(train_ag) + [0] * len(train_nag))

    test_embeddings = test_ag + test_nag
    test_labels     = np.array([1] * len(test_ag) + [0] * len(test_nag))

    print(f"Train: {len(train_embeddings)} proteins ({len(train_ag)} AG, {len(train_nag)} NAG)")
    print(f"Test:  {len(test_embeddings)} proteins ({len(test_ag)} AG, {len(test_nag)} NAG)")

    # ---------------------------------------------------------------------------
    # Config & Run
    # ---------------------------------------------------------------------------
    config = TrainingConfig()
    output_dir = os.path.join(PROJECT_ROOT, "results", "RCNN")

    results = cross_validate_rcnn(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        config=config,
        device=device,
        output_dir=output_dir,
    )
