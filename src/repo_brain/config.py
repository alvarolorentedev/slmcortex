from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXCLUDES = (
    ".git",
    ".repo-brain",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    max_file_size: int = 2 * 1024 * 1024


def state_dir(root: Path) -> Path:
    return root / ".repo-brain"


def load_config(root: Path) -> Config:
    directory = state_dir(root)
    directory.mkdir(exist_ok=True)
    ignore = directory / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n!.gitignore\n")
    path = directory / "config.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {"excludes": list(DEFAULT_EXCLUDES), "max_file_size": 2 * 1024 * 1024},
                indent=2,
            )
            + "\n"
        )
        return Config()
    try:
        raw = json.loads(path.read_text())
        excludes = tuple(str(value) for value in raw.get("excludes", DEFAULT_EXCLUDES))
        max_file_size = int(raw.get("max_file_size", 2 * 1024 * 1024))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
    if max_file_size <= 0:
        raise ConfigError("max_file_size must be positive")
    return Config(excludes, max_file_size)

