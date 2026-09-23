"""Run cross-validation for one saved RCNN architecture."""

from __future__ import annotations

import argparse
import os

import torch

from shared.config_loader import get_default_config_path, load_yaml_config
from src.modules.rcnn_training.entrypoint import (
    cross_validate_saved_model,
    load_training_data_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stratified cross-validation for a saved RCNN model."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a comparison checkpoint, for example results/RCNN/wide_bilstm_best_model.pt",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Configured variant name; inferred from <variant>_best_model.pt when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CV JSON, plot and best fold checkpoint.",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = load_yaml_config(get_default_config_path())
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    output_dir = args.output_dir or config["project"]["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_embeddings, train_labels, test_embeddings, test_labels = load_training_data_from_config(
        config=config,
        project_root=project_root,
    )

    results = cross_validate_saved_model(
        model_path=model_path,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        config=config,
        device=device,
        output_dir=output_dir,
        variant_name=args.variant,
    )
    print(f"Saved CV results: {results['results_path']}")
    print(f"Saved CV loss plot: {results['plot_path']}")


if __name__ == "__main__":
    main()
