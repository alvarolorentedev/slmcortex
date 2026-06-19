from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from repo_brain.config import state_dir
from repo_brain.models import LocalizationCandidate
from repo_brain.storage.sqlite_store import SQLiteStore
from repo_brain.text import tokenize


def localize(root: Path, task: str) -> list[LocalizationCandidate]:
    store = SQLiteStore(state_dir(root) / "index.sqlite3")
    files = store.files()
    symbols = store.symbols()
    dependencies = store.dependencies()
    tests = store.tests()
    task_lower = task.lower()
    task_tokens = set(tokenize(task))
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)
    symbol_ids: dict[str, str] = {}

    for path, record in files.items():
        if record.is_generated:
            continue
        path_lower = path.lower()
        if path_lower in task_lower or Path(path).name.lower() in task_lower:
            scores[path] += 8
            reasons[path].append("exact path mention")
        overlap = task_tokens & set(tokenize(path))
        if overlap:
            value = min(9, len(overlap) * 3)
            scores[path] += value
            reasons[path].append(f"path tokens: {', '.join(sorted(overlap))}")
        if record.is_config and set(tokenize(path)) & task_tokens:
            scores[path] += 2
            reasons[path].append("matching configuration")

    for symbol in symbols:
        symbol_tokens = set(tokenize(symbol.qualified_name))
        overlap = task_tokens & symbol_tokens
        if symbol.qualified_name.lower() in task_lower or symbol.name.lower() in task_lower:
            scores[symbol.file_path] += 6
            reasons[symbol.file_path].append(f"exact symbol: {symbol.qualified_name}")
            symbol_ids[symbol.file_path] = symbol.id
        if overlap:
            value = min(9, len(overlap) * 3)
            scores[symbol.file_path] += value
            reasons[symbol.file_path].append(f"symbol tokens: {', '.join(sorted(overlap))}")
            symbol_ids.setdefault(symbol.file_path, symbol.id)

    fts_paths = store.search(task_tokens)
    for path in fts_paths:
        scores[path] += 5
        reasons[path].append("full-text match")

    direct_scores = dict(scores)
    for edge in dependencies:
        source_score = direct_scores.get(edge.source_file, 0)
        target_score = direct_scores.get(edge.resolved_file or "", 0)
        if source_score and edge.resolved_file:
            scores[edge.resolved_file] += source_score * 0.5
            reasons[edge.resolved_file].append(f"dependency of {edge.source_file}")
        if target_score:
            scores[edge.source_file] += target_score * 0.25
            reasons[edge.source_file].append(f"dependent of {edge.resolved_file}")

    for test in tests:
        test_tokens = set(tokenize(test.name)) | set(tokenize(test.file_path))
        overlap = task_tokens & test_tokens
        if overlap:
            scores[test.file_path] += 6
            reasons[test.file_path].append(f"test tokens: {', '.join(sorted(overlap))}")

    metadata = store.metadata()
    if json.loads(metadata.get("dirty", "false")):
        for path in scores:
            scores[path] += 0.25

    candidates = [
        LocalizationCandidate(
            "test" if files[path].is_test else "file",
            path,
            symbol_ids.get(path),
            round(score, 3),
            tuple(dict.fromkeys(reasons[path])),
        )
        for path, score in scores.items()
        if score > 0 and path in files
    ]
    return sorted(candidates, key=lambda item: (-item.score, item.path))[:60]

