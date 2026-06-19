from pathlib import Path

from repo_brain.language_detection import detect_frameworks, detect_language


def test_detects_languages_and_frameworks(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"next":"1","react":"1"},"devDependencies":{"vitest":"1"}}'
    )
    (tmp_path / "pyproject.toml").write_text('dependencies = ["fastapi", "pytest"]')
    assert detect_language(Path("x.tsx")) == "typescript"
    assert {"nextjs", "react", "vitest", "fastapi", "pytest"} <= detect_frameworks(tmp_path)

