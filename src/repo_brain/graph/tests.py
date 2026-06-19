from __future__ import annotations

import hashlib

from repo_brain.models import FileRecord, SymbolRecord, TestRecord


def discover_tests(files: list[FileRecord], symbols: list[SymbolRecord]) -> list[TestRecord]:
    test_files = {record.path: record for record in files if record.is_test}
    tests: list[TestRecord] = []
    for symbol in symbols:
        record = test_files.get(symbol.file_path)
        if record is None:
            continue
        is_python_test = record.language == "python" and (
            symbol.name.startswith("test_") or symbol.kind == "class"
        )
        is_js_test = record.language in {"javascript", "typescript"} and symbol.kind in {
            "function",
            "test",
        }
        if not (is_python_test or is_js_test):
            continue
        framework = "pytest" if record.language == "python" else "vitest"
        command = (
            f"pytest {record.path}::{symbol.qualified_name}"
            if framework == "pytest"
            else f"npx vitest run {record.path}"
        )
        identifier = hashlib.sha1(f"{record.path}:{symbol.name}".encode()).hexdigest()
        tests.append(
            TestRecord(identifier, record.path, symbol.name, framework, symbol.line_start, command)
        )
    return tests

