from pathlib import Path

from repo_brain.indexer import index_repository
from repo_brain.skills.repo_map import repository_map


def test_repository_map_summarizes_index(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "test_app.py").write_text(
        "from pkg.app import main\n\ndef test_main():\n    main()\n"
    )
    assert not index_repository(tmp_path).errors
    result = repository_map(tmp_path)
    assert result["languages"]["python"] == 2
    assert result["symbol_counts"]["function"] >= 2
    assert result["tests"]
