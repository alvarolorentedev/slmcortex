import hashlib
import ast
import json
from pathlib import Path

import scripts.run_composition_control as composition


ROOT = Path(__file__).resolve().parents[1]


def test_composition_runner_has_no_training_dependency():
    tree = ast.parse(Path(composition.__file__).read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("train" in name for name in imports)


def test_counterfactual_policies_select_expected_source_rows():
    rows = [
        {
            "seed": 11,
            "example_id": "d",
            "task_type": "debugging",
            "mode": mode,
            "selected_skills": skills,
            "execution_passed": True,
            "active_adapter_parameters": len(skills) * 10,
            "benchmark_group": "family",
        }
        for mode, skills in (
            ("base", []),
            ("single-skill", ["debugging_skill"]),
            ("lattice", ["debugging_skill", "test_generation_skill"]),
            ("oracle-lattice", ["python_skill", "debugging_skill"]),
        )
    ]
    selected = composition.select_counterfactual_rows(rows)
    assert selected["single_strongest_skill"][0]["selected_skills"] == [
        "debugging_skill"
    ]
    assert selected["no_harmful_pairs"][0]["selected_skills"] == [
        "debugging_skill"
    ]


def test_dry_run_labels_recombination_and_fresh_inference(monkeypatch, tmp_path):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    monkeypatch.setattr(composition, "_load_existing_rows", lambda *args: [])
    monkeypatch.setattr(composition, "select_counterfactual_rows", lambda rows: {})
    monkeypatch.setattr(composition, "_load_weighted_rows", lambda *args: {})
    monkeypatch.setattr(
        composition,
        "_build_summary",
        lambda *args, **kwargs: {
            "status": "dry-run",
            "requires_training": False,
            "training_invoked": False,
            "benchmark_sha256": before,
            "policies": {
                "current_composition": {
                    "fresh_inference": False,
                    "counterfactual_recombination": True,
                },
                "weighted_task_composition": {
                    "fresh_inference": True,
                    "counterfactual_recombination": False,
                },
            },
        },
    )
    monkeypatch.setattr(composition, "_markdown", lambda summary: "# dry-run\n")
    monkeypatch.setattr(composition, "evaluate", lambda *args, **kwargs: None)

    assert composition.main([
        "--seeds", "11",
        "--dataset", str(benchmark),
        "--five-seed-experiment", str(ROOT / "artifacts/experiments/five-seed"),
        "--protected-experiment", str(
            ROOT / "artifacts/experiments/python-skill-gating-validation"
        ),
        "--output", str(tmp_path),
        "--dry-run",
    ]) == 0

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["policies"]["current_composition"]["fresh_inference"] is False
    assert (
        summary["policies"]["weighted_task_composition"][
            "counterfactual_recombination"
        ]
        is False
    )
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
