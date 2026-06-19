from __future__ import annotations

import json
from pathlib import Path

EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}


def detect_language(path: Path) -> str:
    return EXTENSIONS.get(path.suffix.lower(), "unknown")


def detect_frameworks(root: Path) -> set[str]:
    found: set[str] = set()
    package = root / "package.json"
    if package.exists():
        try:
            raw = json.loads(package.read_text())
            names = set(raw.get("dependencies", {})) | set(raw.get("devDependencies", {}))
        except (OSError, json.JSONDecodeError, TypeError):
            names = set()
        mapping = {
            "next": "nextjs",
            "react": "react",
            "express": "express",
            "jest": "jest",
            "vitest": "vitest",
            "@playwright/test": "playwright",
        }
        found.update(value for key, value in mapping.items() if key in names)
    for name in ("pyproject.toml", "requirements.txt", "setup.cfg"):
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(errors="ignore").lower()
        found.update(
            framework
            for marker, framework in {
                "pytest": "pytest",
                "django": "django",
                "flask": "flask",
                "fastapi": "fastapi",
            }.items()
            if marker in text
        )
    return found

