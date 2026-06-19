from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from repo_brain.models import DependencyEdge, SymbolRecord


@dataclass(frozen=True)
class AnalysisResult:
    symbols: tuple[SymbolRecord, ...] = ()
    dependencies: tuple[DependencyEdge, ...] = ()
    diagnostics: tuple[str, ...] = ()


class Analyzer(Protocol):
    languages: frozenset[str]

    def analyze(self, path: Path, source: str) -> AnalysisResult: ...

