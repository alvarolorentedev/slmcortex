import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SKILLCORTEX_ROOT = ROOT / "src" / "skillcortex"
ALLOWED_DIRECT_IMPORT_ROOTS = {
    (SKILLCORTEX_ROOT / "backends").resolve(),
}


def _product_python_files() -> list[Path]:
    files = []
    for path in sorted(SKILLCORTEX_ROOT.rglob("*.py")):
        if any(parent.name == "__pycache__" for parent in path.parents):
            continue
        if any(allowed in path.resolve().parents or path.resolve() == allowed for allowed in ALLOWED_DIRECT_IMPORT_ROOTS):
            continue
        files.append(path)
    return files


def test_skillcortex_product_modules_do_not_import_legacy_engine_directly():
    violations = []
    for path in _product_python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "skill_lattice_coder" or alias.name.startswith("skill_lattice_coder."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "skill_lattice_coder" or (
                    node.module is not None and node.module.startswith("skill_lattice_coder.")
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports from {node.module}")
    assert violations == []


def test_console_scripts_expose_only_skillcortex_product_entrypoint():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts["skillcortex"] == "skillcortex.cli:main"
    assert "skill-lattice" not in scripts
