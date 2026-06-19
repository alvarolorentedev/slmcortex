from pathlib import Path

from repo_brain.indexer import index_repository
from repo_brain.skills.localize import localize
from repo_brain.text import tokenize


def test_tokenize_splits_identifier_styles() -> None:
    assert {"login", "redirect", "token"} <= set(tokenize("loginRedirect-token"))


def test_localize_ranks_exact_symbols_and_dependencies(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text(
        "from session import expire\n\ndef redirect_after_expiry():\n    return expire()\n"
    )
    (tmp_path / "session.py").write_text("def expire():\n    return '/login'\n")
    (tmp_path / "test_auth.py").write_text(
        "from auth import redirect_after_expiry\n\ndef test_redirect():\n"
        "    assert redirect_after_expiry()\n"
    )
    assert not index_repository(tmp_path).errors
    candidates = localize(tmp_path, "fix redirect_after_expiry login token")
    assert candidates[0].path == "auth.py"
    assert candidates[0].reasons
    assert any(candidate.path == "test_auth.py" for candidate in candidates)


def test_localize_order_is_stable(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def common(): pass\n")
    (tmp_path / "b.py").write_text("def common(): pass\n")
    index_repository(tmp_path)
    first = localize(tmp_path, "common")
    second = localize(tmp_path, "common")
    assert first == second

