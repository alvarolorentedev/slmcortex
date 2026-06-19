from __future__ import annotations

import subprocess
from pathlib import Path


class RepositoryError(ValueError):
    pass


def resolve_repository(path: Path | str | None = None) -> Path:
    candidate = Path(path).expanduser() if path is not None else Path.cwd()
    if not candidate.exists() or not candidate.is_dir():
        raise RepositoryError(f"repository path does not exist or is not a directory: {candidate}")
    candidate = candidate.resolve()
    if path is None:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            candidate = Path(result.stdout.strip()).resolve()
    return candidate


def git_state(root: Path) -> tuple[str | None, bool]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0:
        return None, False
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return revision.stdout.strip(), bool(dirty.stdout.strip())

