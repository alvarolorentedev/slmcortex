from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tree_sitter_language_pack import get_parser

from repo_brain.analyzers.base import AnalysisResult
from repo_brain.models import DependencyEdge, SymbolRecord

SYMBOL_TYPES = {
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
}


def _walk(node: Any) -> list[Any]:
    result = [node]
    for index in range(node.child_count()):
        child = node.child(index)
        if child is not None:
            result.extend(_walk(child))
    return result


def _text(node: Any, source: bytes) -> str:
    return source[node.start_byte() : node.end_byte()].decode(errors="replace")


class JavaScriptAnalyzer:
    languages = frozenset({"javascript", "typescript"})

    def analyze(self, path: Path, source: str) -> AnalysisResult:
        language = (
            "tsx"
            if path.suffix == ".tsx"
            else "typescript"
            if path.suffix == ".ts"
            else "javascript"
        )
        parser = get_parser(language)
        tree = parser.parse(source)
        if tree is None:
            return AnalysisResult(diagnostics=("tree-sitter returned no tree",))
        encoded = source.encode()
        symbols: list[SymbolRecord] = []
        dependencies: list[DependencyEdge] = []
        diagnostics: list[str] = []
        root = tree.root_node()
        if root.has_error():
            diagnostics.append("tree-sitter parse contains errors")
        for node in _walk(root):
            kind = node.kind()
            if kind in SYMBOL_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = _text(name_node, encoded)
                line = node.start_position().row + 1
                identifier = hashlib.sha1(f"{path}:{name}:{line}".encode()).hexdigest()
                symbols.append(
                    SymbolRecord(
                        identifier,
                        path.as_posix(),
                        name,
                        name,
                        SYMBOL_TYPES[kind],
                        line,
                        node.end_position().row + 1,
                        _text(node, encoded).split("{", 1)[0].strip(),
                    )
                )
            if kind in {"import_statement", "export_statement"}:
                source_node = node.child_by_field_name("source")
                if source_node is not None:
                    dependencies.append(
                        DependencyEdge(
                            path.as_posix(),
                            _text(source_node, encoded).strip("'\""),
                            "import",
                        )
                    )
            if kind == "call_expression":
                function = node.child_by_field_name("function")
                arguments = node.child_by_field_name("arguments")
                if function is not None and arguments is not None:
                    function_name = _text(function, encoded)
                    first = arguments.named_child(0) if arguments.named_child_count() else None
                    if function_name in {"require", "import"} and first is not None:
                        dependencies.append(
                            DependencyEdge(
                                path.as_posix(),
                                _text(first, encoded).strip("'\""),
                                "dynamic_import",
                            )
                        )
        return AnalysisResult(tuple(symbols), tuple(dependencies), tuple(diagnostics))
