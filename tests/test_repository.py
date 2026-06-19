from pathlib import Path

import pytest

from repo_brain.repository import RepositoryError, resolve_repository


def test_resolves_explicit_repository(tmp_path: Path) -> None:
    assert resolve_repository(tmp_path) == tmp_path.resolve()


def test_rejects_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError):
        resolve_repository(tmp_path / "missing")

