import json
from pathlib import Path

from repo_brain.cli import main


def test_index_json_matches_human_counts(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    assert main(["--repo", str(tmp_path), "--json", "index"]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["schema_version"] == 1
    assert output["command"] == "index"
    assert output["data"]["added"] == 1


def test_malformed_config_returns_storage_error(tmp_path: Path) -> None:
    state = tmp_path / ".repo-brain"
    state.mkdir()
    (state / "config.json").write_text("{")
    assert main(["--repo", str(tmp_path), "index"]) == 3
