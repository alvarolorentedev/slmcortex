from __future__ import annotations

from pathlib import PurePosixPath


def resolve_dependency(
    source_file: str, target: str, files: set[str], language: str
) -> str | None:
    source = PurePosixPath(source_file)
    if language == "python":
        if target.startswith("."):
            levels = len(target) - len(target.lstrip("."))
            base = source.parent
            for _ in range(max(0, levels - 1)):
                base = base.parent
            module = target.lstrip(".").replace(".", "/")
            candidates = [base / f"{module}.py", base / module / "__init__.py"]
        else:
            module = target.replace(".", "/")
            candidates = [PurePosixPath(f"{module}.py"), PurePosixPath(module) / "__init__.py"]
    elif target.startswith("."):
        base = source.parent / target
        candidates = [
            base.with_suffix(".ts"),
            base.with_suffix(".tsx"),
            base.with_suffix(".js"),
            base.with_suffix(".jsx"),
            base / "index.ts",
            base / "index.js",
        ]
    else:
        return None
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in files:
            return normalized
    return None

