from __future__ import annotations

import os
from pathlib import Path

from .io import read_yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def base_config() -> dict:
    return read_yaml(Path(os.environ.get("SKILLCORTEX_BASE_CONFIG") or CONFIG_DIR / "base.yaml"))


def training_config() -> dict:
    return read_yaml(CONFIG_DIR / "training.yaml")
