from repo_brain.graph.dependencies import resolve_dependency


def test_resolves_python_and_typescript_dependencies() -> None:
    files = {"pkg/a.py", "pkg/b.py", "src/a.ts", "src/b.ts", "src/b/index.ts"}
    assert resolve_dependency("pkg/a.py", ".b", files, "python") == "pkg/b.py"
    assert resolve_dependency("src/a.ts", "./b", files, "typescript") in {
        "src/b.ts",
        "src/b/index.ts",
    }

