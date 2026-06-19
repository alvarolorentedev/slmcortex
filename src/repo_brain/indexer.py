from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path

from repo_brain.config import ConfigError, load_config, state_dir
from repo_brain.models import FileRecord, IndexResult
from repo_brain.repository import RepositoryError, git_state, resolve_repository
from repo_brain.storage.sqlite_store import SQLiteStore

LANGUAGES = {
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
CONFIG_NAMES = {
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "vite.config.js",
    "vite.config.ts",
    "next.config.js",
    "next.config.ts",
}


def _is_binary(path: Path) -> bool:
    try:
        return b"\0" in path.read_bytes()[:4096]
    except OSError:
        return True


def _record(root: Path, path: Path, stat: os.stat_result) -> FileRecord:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = path.name.lower()
    is_test = (
        name.startswith("test_")
        or "_test." in name
        or ".test." in name
        or ".spec." in name
        or "__tests__" in path.parts
    )
    is_generated = name.endswith((".min.js", ".map")) or "generated" in path.parts
    return FileRecord(
        relative,
        LANGUAGES.get(path.suffix.lower(), "unknown"),
        stat.st_size,
        digest,
        stat.st_mtime_ns,
        is_test,
        path.name in CONFIG_NAMES,
        is_generated,
    )


def index_repository(path: Path | str | None = None) -> IndexResult:
    try:
        root = resolve_repository(path)
        config = load_config(root)
    except (RepositoryError, ConfigError, OSError) as exc:
        return IndexResult(errors=(str(exc),))
    store = SQLiteStore(state_dir(root) / "index.sqlite3")
    previous = store.files()
    records: list[FileRecord] = []
    scanned = added = updated = unchanged = ignored = failed = 0
    warnings: list[str] = []
    excluded = set(config.excludes)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        kept_dirs: list[str] = []
        for name in dirnames:
            child = current / name
            if name in excluded or child.is_symlink():
                ignored += 1
            else:
                kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in filenames:
            file_path = current / name
            if file_path.is_symlink():
                ignored += 1
                continue
            try:
                stat = file_path.stat()
                if stat.st_size > config.max_file_size or _is_binary(file_path):
                    ignored += 1
                    continue
                relative = file_path.relative_to(root).as_posix()
                old = previous.get(relative)
                if old and old.size == stat.st_size and old.modified_ns == stat.st_mtime_ns:
                    records.append(old)
                    scanned += 1
                    unchanged += 1
                    continue
                record = _record(root, file_path, stat)
                records.append(record)
                scanned += 1
                if old:
                    updated += 1
                else:
                    added += 1
            except OSError as exc:
                failed += 1
                warnings.append(f"{file_path}: {exc}")
    removed = len(set(previous) - {record.path for record in records})
    store.replace_files(records)
    revision, dirty = git_state(root)
    languages = dict(Counter(record.language for record in records))
    store.set_metadata("root", str(root))
    store.set_metadata("revision", revision or "")
    store.set_metadata("dirty", dirty)
    store.set_metadata("languages", languages)
    return IndexResult(
        scanned,
        added,
        updated,
        unchanged,
        removed,
        ignored,
        failed,
        tuple(warnings),
        (),
        languages,
    )

