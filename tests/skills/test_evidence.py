from pathlib import Path

from repo_brain.indexer import index_repository
from repo_brain.skills.evidence import build_evidence


def test_evidence_is_bounded_and_source_linked(tmp_path: Path) -> None:
    (tmp_path / "api.py").write_text(
        "def authenticate(token: str) -> bool:\n"
        '    """Validate a login token."""\n'
        "    return bool(token)\n"
    )
    (tmp_path / "test_api.py").write_text(
        "from api import authenticate\n\ndef test_authenticate():\n"
        "    assert authenticate('x')\n"
    )
    index_repository(tmp_path)
    bundle = build_evidence(tmp_path, "authenticate login token", max_chars=1200)
    rendered = bundle.render()
    assert len(rendered) <= 1200
    assert "api.py:" in rendered
    assert "authenticate" in rendered


def test_evidence_warns_when_source_changed_after_index(tmp_path: Path) -> None:
    path = tmp_path / "api.py"
    path.write_text("def run():\n    pass\n")
    index_repository(tmp_path)
    path.write_text("def changed():\n    pass\n")
    bundle = build_evidence(tmp_path, "run")
    assert any("changed since indexing" in warning for warning in bundle.warnings)
