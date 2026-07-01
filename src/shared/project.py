from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .io import read_yaml


PROJECT_CONFIG = ".slmcortex.yaml"


def init_project(root: Path) -> dict:
    root = root.resolve()
    state = root / ".slmcortex"
    slms_dir = state / "slms"
    cache_dir = state / "lora-cache"
    runtimes_dir = state / "runtimes"
    for path in (slms_dir, cache_dir, runtimes_dir):
        path.mkdir(parents=True, exist_ok=True)
    config_path = root / PROJECT_CONFIG
    if not config_path.exists():
        config_path.write_text(_template())
    return {
        "status": "complete",
        "config": str(config_path),
        "slms_dir": str(slms_dir),
        "lora_cache_dir": str(cache_dir),
        "runtimes_dir": str(runtimes_dir),
        "next_steps": [
            "edit .slmcortex.yaml and add Hugging Face LoRAs",
            "slmcortex loras download <name>",
            "slmcortex serve",
        ],
    }


def load_project_config(root: Path | None = None) -> dict:
    root = (root or Path.cwd()).resolve()
    path = root / PROJECT_CONFIG
    if not path.exists():
        return {}
    payload = read_yaml(path)
    payload["_project_root"] = root
    return payload


def project_path(config: dict, key: str, default: str) -> Path:
    root = Path(config.get("_project_root") or Path.cwd()).resolve()
    value = config.get(key) or default
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def project_slms_dir(config: dict) -> Path | None:
    if not config:
        return None
    return project_path(config, "slms_dir", ".slmcortex/slms")


def project_cache_dir(config: dict) -> Path:
    return project_path(config, "lora_cache_dir", ".slmcortex/lora-cache")


def project_dataset(config: dict, key: str, default_name: str) -> Path:
    value = config.get(key)
    if value:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(config.get("_project_root") or Path.cwd()).resolve() / path
    return DATA_DIR / default_name


def configured_loras(config: dict) -> dict[str, dict[str, Any]]:
    loras = config.get("loras") or {}
    if not isinstance(loras, dict):
        raise ValueError("loras must be a mapping in .slmcortex.yaml")
    return {str(name): value for name, value in loras.items() if isinstance(value, dict)}


def _template() -> str:
    return """slms_dir: .slmcortex/slms
lora_cache_dir: .slmcortex/lora-cache

# Optional. Override packaging reference datasets.
# train_dataset: data/train.jsonl
# eval_dataset: data/eval.jsonl

loras: {}
# Example:
# loras:
#   fastapi:
#     source: hf://owner/fastapi-lora
#     name: FastAPI LoRA
#     description: FastAPI and Pydantic coding tasks
"""
