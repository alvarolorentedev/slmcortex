from pathlib import Path

from repo_brain.analyzers.python import PythonAnalyzer


def test_python_analyzer_extracts_symbols_and_imports() -> None:
    source = """
from .service import Client as ApiClient
import os

class Handler:
    @classmethod
    def run(cls, value: str) -> bool:
        return bool(value)

async def fetch() -> None:
    pass
"""
    result = PythonAnalyzer().analyze(Path("pkg/module.py"), source)
    names = {(symbol.qualified_name, symbol.kind) for symbol in result.symbols}
    assert ("Handler", "class") in names
    assert ("Handler.run", "method") in names
    assert ("fetch", "async_function") in names
    assert {edge.target for edge in result.dependencies} == {".service", "os"}


def test_python_analyzer_reports_parse_errors() -> None:
    result = PythonAnalyzer().analyze(Path("bad.py"), "def broken(:")
    assert result.diagnostics

