from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RepositoryInfo:
    root: str
    revision: str | None
    indexed_at: str
    languages: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileRecord:
    path: str
    language: str
    size: int
    digest: str
    modified_ns: int
    is_test: bool
    is_config: bool
    is_generated: bool


@dataclass(frozen=True)
class SymbolRecord:
    id: str
    file_path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    signature: str


@dataclass(frozen=True)
class DependencyEdge:
    source_file: str
    target: str
    kind: str
    resolved_file: str | None = None


@dataclass(frozen=True)
class TestRecord:
    id: str
    file_path: str
    name: str
    framework: str
    line_start: int
    command: str


@dataclass(frozen=True)
class TestLink:
    test_id: str
    target_file: str | None
    target_symbol_id: str | None
    confidence: float
    reason: str


@dataclass(frozen=True)
class LocalizationCandidate:
    kind: str
    path: str
    symbol_id: str | None
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceItem:
    path: str
    line_start: int
    line_end: int
    content: str
    reason: str
    score: float


@dataclass(frozen=True)
class EvidenceBundle:
    task: str
    repository_summary: dict[str, Any]
    candidates: tuple[LocalizationCandidate, ...]
    items: tuple[EvidenceItem, ...]
    suggested_tests: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    command: str
    status: str
    duration_ms: int
    output: str


@dataclass(frozen=True)
class ValidationReport:
    applicable: bool
    checks: tuple[ValidationCheck, ...]
    affected_files: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class FailureExplanation:
    category: str
    primary_message: str
    frames: tuple[dict[str, Any], ...]
    related_symbols: tuple[str, ...]
    suggestions: tuple[str, ...]


@dataclass(frozen=True)
class RiskFinding:
    code: str
    severity: str
    score: int
    message: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class RiskReport:
    score: int
    band: str
    confidence: float
    findings: tuple[RiskFinding, ...]
    affected_symbols: tuple[str, ...]
    impacted_tests: tuple[str, ...]


@dataclass(frozen=True)
class TraceRun:
    id: str
    started_at: str
    repository_revision: str | None
    task: str | None
    status: str


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    sequence: int
    timestamp: str
    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class IndexResult:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    ignored: int = 0
    failed: int = 0
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    languages: dict[str, int] = field(default_factory=dict)

