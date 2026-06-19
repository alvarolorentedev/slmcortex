from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import cast

from repo_brain.analyzers.base import AnalysisResult
from repo_brain.models import DependencyEdge, SymbolRecord


def _id(path: Path, qualified_name: str, line: int) -> str:
    value = f"{path.as_posix()}:{qualified_name}:{line}".encode()
    return hashlib.sha1(value).hexdigest()


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [argument.arg for argument in node.args.args]
    return f"{node.name}({', '.join(args)})"


class PythonAnalyzer:
    languages = frozenset({"python"})

    def analyze(self, path: Path, source: str) -> AnalysisResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return AnalysisResult(diagnostics=(f"{exc.msg} at line {exc.lineno}",))
        symbols: list[SymbolRecord] = []
        dependencies: list[DependencyEdge] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []

            def symbol(self, node: ast.stmt, name: str, kind: str, signature: str = "") -> None:
                qualified = ".".join([*self.scope, name])
                line_start = node.lineno
                line_end = cast(int, getattr(node, "end_lineno", line_start) or line_start)
                symbols.append(
                    SymbolRecord(
                        _id(path, qualified, line_start),
                        path.as_posix(),
                        name,
                        qualified,
                        kind,
                        line_start,
                        line_end,
                        signature,
                    )
                )

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.symbol(node, node.name, "class", f"class {node.name}")
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                kind = "method" if self.scope else "function"
                self.symbol(node, node.name, kind, _signature(node))
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                kind = "method" if self.scope else "async_function"
                self.symbol(node, node.name, kind, _signature(node))
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    dependencies.append(DependencyEdge(path.as_posix(), alias.name, "import"))

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                target = "." * node.level + (node.module or "")
                dependencies.append(DependencyEdge(path.as_posix(), target, "import"))

        Visitor().visit(tree)
        return AnalysisResult(tuple(symbols), tuple(dependencies))
