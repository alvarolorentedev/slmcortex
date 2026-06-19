from pathlib import Path

from repo_brain.indexer import index_repository


def test_index_is_incremental_and_removes_deleted_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print('a')\n")
    first = index_repository(tmp_path)
    second = index_repository(tmp_path)
    (tmp_path / "a.py").unlink()
    (tmp_path / "b.ts").write_text("export const b = 1;\n")
    third = index_repository(tmp_path)

    assert first.added == 1
    assert second.unchanged == 1
    assert third.added == 1
    assert third.removed == 1


def test_index_ignores_dependencies_symlinks_and_large_files(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("ignored")
    (tmp_path / "large.py").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    (tmp_path / "ok.py").write_text("ok = True\n")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)

    result = index_repository(tmp_path)

    assert result.added == 1
    assert result.ignored >= 3


def test_malformed_config_is_reported(tmp_path: Path) -> None:
    state = tmp_path / ".repo-brain"
    state.mkdir()
    (state / "config.json").write_text("{")
    result = index_repository(tmp_path)
    assert result.errors

