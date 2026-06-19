from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from repo_brain.config import state_dir
from repo_brain.graph.symbols import symbol_counts
from repo_brain.storage.sqlite_store import SQLiteStore


def repository_map(root: Path) -> dict[str, object]:
    store = SQLiteStore(state_dir(root) / "index.sqlite3")
    files = list(store.files().values())
    symbols = store.symbols()
    dependencies = store.dependencies()
    tests = store.tests()
    metadata = store.metadata()
    hubs = Counter(edge.resolved_file for edge in dependencies if edge.resolved_file)
    return {
        "revision": metadata.get("revision") or None,
        "dirty": json.loads(metadata.get("dirty", "false")),
        "languages": dict(sorted(Counter(file.language for file in files).items())),
        "frameworks": json.loads(metadata.get("frameworks", "[]")),
        "entrypoints": sorted(
            file.path
            for file in files
            if Path(file.path).name in {"main.py", "__main__.py", "index.js", "index.ts", "app.py"}
        ),
        "config_files": sorted(file.path for file in files if file.is_config),
        "symbol_counts": symbol_counts(symbols),
        "dependency_hubs": [
            {"path": path, "incoming": count} for path, count in hubs.most_common(10) if path
        ],
        "tests": [
            {"path": test.file_path, "name": test.name, "command": test.command}
            for test in tests[:50]
        ],
        "warnings": store.analysis_errors(),
    }
