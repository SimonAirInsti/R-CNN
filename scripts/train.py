"""Thin execution wrapper for the RCNN training pipeline.

This script is intentionally small and readable. It reads project
configuration, loads the precomputed embeddings and delegates the actual
training logic to the modular training entrypoint.
"""

from __future__ import annotations

import os

import torch

from shared.config_loader import get_default_config_path, load_yaml_config
from src.modules.rcnn_training.entrypoint import (
    TrainingConfig,
    load_training_data_from_config,
)
from legacy.RCNN_CV import cross_validate_rcnn


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_yaml_config(get_default_config_path())
    training_cfg = TrainingConfig.from_dict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Project root directory: {project_root}")
    print(f"Using device: {device}")

    train_embeddings, train_labels, test_embeddings, test_labels = load_training_data_from_config(
        config=config,
        project_root=project_root,
    )

    output_dir = os.path.join(project_root, config["project"]["output_dir"])

    print(f"Train: {len(train_embeddings)} proteins")
    print(f"Test:  {len(test_embeddings)} proteins")

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
