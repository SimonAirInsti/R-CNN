from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a dictionary."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def get_default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "rcnn_config.yaml"
