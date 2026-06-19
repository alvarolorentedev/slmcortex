from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from repo_brain.config import state_dir
from repo_brain.models import EvidenceBundle, EvidenceItem
from repo_brain.rendering import render_evidence
from repo_brain.skills.localize import localize
from repo_brain.skills.repo_map import repository_map
from repo_brain.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class RenderableEvidence:
    bundle: EvidenceBundle
    max_chars: int

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.bundle.warnings

    def render(self) -> str:
        return render_evidence(self.bundle, self.max_chars)


def build_evidence(root: Path, task: str, max_chars: int = 16_000) -> RenderableEvidence:
    store = SQLiteStore(state_dir(root) / "index.sqlite3")
    files = store.files()
    symbols = {symbol.id: symbol for symbol in store.symbols()}
    tests = store.tests()
    candidates = localize(root, task)
    items: list[EvidenceItem] = []
    warnings: list[str] = []
    used: set[tuple[str, int, int]] = set()
    for candidate in candidates[:12]:
        record = files[candidate.path]
        path = root / candidate.path
        try:
            content = path.read_text(errors="replace")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            warnings.append(f"{candidate.path}: {exc}")
            continue
        if digest != record.digest:
            warnings.append(f"{candidate.path} changed since indexing")
        symbol = symbols.get(candidate.symbol_id or "")
        lines = content.splitlines()
        start = max(1, symbol.line_start if symbol else 1)
        end = min(len(lines), symbol.line_end if symbol else min(len(lines), 80))
        key = (candidate.path, start, end)
        if key in used:
            continue
        used.add(key)
        excerpt = "\n".join(lines[start - 1 : end])
        items.append(
            EvidenceItem(
                candidate.path,
                start,
                end,
                excerpt,
                "; ".join(candidate.reasons),
                candidate.score,
            )
        )
    suggested = tuple(
        test.command
        for test in tests
        if any(test.file_path == candidate.path for candidate in candidates[:20])
    )
    summary = repository_map(root)
    compact_summary = {
        "revision": summary["revision"],
        "languages": summary["languages"],
        "frameworks": summary["frameworks"],
    }
    bundle = EvidenceBundle(
        task,
        compact_summary,
        tuple(candidates),
        tuple(items),
        suggested,
        tuple(warnings),
    )
    return RenderableEvidence(bundle, max_chars)
