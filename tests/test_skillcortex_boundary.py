from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_src_tree_is_flat_product_layout():
    entries = {
        path.name
        for path in (ROOT / "src").iterdir()
        if path.name != "__pycache__"
    }
    assert entries == {
        "agent",
        "cli",
        "composer",
        "contracts.py",
        "dataset_factory",
        "datasets",
        "packaging",
        "runtime",
        "shared",
        "skillcortex.py",
        "training",
    }


def test_console_scripts_expose_only_skillcortex_product_entrypoint():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = payload["project"]["scripts"]
    assert scripts == {"skillcortex": "skillcortex:main"}
