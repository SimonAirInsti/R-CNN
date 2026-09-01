"""Compatibility wrapper for the modularized RCNN training flow."""

from __future__ import annotations

import os

import numpy as np
import torch

from shared.config_loader import get_default_config_path, load_yaml_config
from src.modules.rcnn_training.entrypoint import TrainingConfig, load_embeddings
from src.modules.rcnn_training.entrypoint import RCNN
from RCNN_CV import cross_validate_rcnn


def main() -> None:
    config_path = get_default_config_path()
    config = load_yaml_config(config_path)

    project_root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project root directory: {project_root}")
    print(f"Using device: {device}")

    data_dir = os.path.join(project_root, "src", "data")
    train_ag = load_embeddings(os.path.join(data_dir, "Table S1. Training set of viral antigens.fasta.pt"))
    train_nag = load_embeddings(os.path.join(data_dir, "Table S2. Training set of viral nonantigens.fasta.pt"))
    test_ag = load_embeddings(os.path.join(data_dir, "Table S3. Test set of viral antigens.fasta.pt"))
    test_nag = load_embeddings(os.path.join(data_dir, "Table S4. Test set of viral nonantigens.fasta.pt"))

    train_embeddings = train_ag + train_nag
    train_labels = np.array([1] * len(train_ag) + [0] * len(train_nag))
    test_embeddings = test_ag + test_nag
    test_labels = np.array([1] * len(test_ag) + [0] * len(test_nag))

    print(f"Train: {len(train_embeddings)} proteins ({len(train_ag)} AG, {len(train_nag)} NAG)")
    print(f"Test:  {len(test_embeddings)} proteins ({len(test_ag)} AG, {len(test_nag)} NAG)")

    training_cfg = TrainingConfig.from_dict(config)
    output_dir = os.path.join(project_root, config["project"]["output_dir"])

    cross_validate_rcnn(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        config=training_cfg,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
