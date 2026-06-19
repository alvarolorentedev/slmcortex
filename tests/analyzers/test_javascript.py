from pathlib import Path

from repo_brain.analyzers.javascript import JavaScriptAnalyzer


def test_typescript_analyzer_extracts_symbols_and_dependencies() -> None:
    source = """
import { client } from "./client";
export interface User { id: string }
export type UserId = string;
export class Service { run(): void {} }
export function load(): User { return client(); }
const lazy = import("./lazy");
"""
    result = JavaScriptAnalyzer().analyze(Path("src/service.ts"), source)
    names = {symbol.name for symbol in result.symbols}
    assert {"User", "UserId", "Service", "run", "load"} <= names
    assert {"./client", "./lazy"} <= {edge.target for edge in result.dependencies}

