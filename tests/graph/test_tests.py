from repo_brain.graph.tests import discover_tests
from repo_brain.models import FileRecord, SymbolRecord


def test_discovers_python_and_javascript_tests() -> None:
    files = [
        FileRecord("tests/test_api.py", "python", 1, "a", 1, True, False, False),
        FileRecord("src/api.test.ts", "typescript", 1, "b", 1, True, False, False),
    ]
    symbols = [
        SymbolRecord("1", "tests/test_api.py", "test_login", "test_login", "function", 2, 3, ""),
        SymbolRecord("2", "src/api.test.ts", "loads", "loads", "test", 4, 5, ""),
    ]
    tests = discover_tests(files, symbols)
    assert {test.framework for test in tests} == {"pytest", "vitest"}

