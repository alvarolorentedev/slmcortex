import hashlib
import ast
import json
from pathlib import Path

import scripts.run_python_skill_gating_validation as validation


ROOT = Path(__file__).resolve().parents[1]


def test_validation_runner_has_no_training_dependency():
    tree = ast.parse(Path(validation.__file__).read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("train" in name for name in imports)


def test_dry_run_records_fresh_inference_plan_without_training_or_benchmark_change(
    monkeypatch, tmp_path, capsys
):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    calls = []

    def fake_evaluate(dataset, **kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output"])
        output.mkdir(parents=True)
        (output / "results.jsonl").write_text("")
        return output

    monkeypatch.setattr(validation, "evaluate", fake_evaluate)
    monkeypatch.setattr(validation, "_load_rows", lambda *args: [])
    monkeypatch.setattr(
        validation,
        "_build_summary",
        lambda *args, **kwargs: {
            "status": "dry-run",
            "fresh_inference": True,
            "counterfactual_recombination": False,
            "requires_training": False,
            "training_invoked": False,
            "benchmark_sha256": before,
        },
    )
    monkeypatch.setattr(validation, "_markdown", lambda summary: "# dry-run\n")

    assert validation.main([
        "--seeds", "11", "22",
        "--dataset", str(benchmark),
        "--baseline-experiment", str(ROOT / "artifacts/experiments/five-seed"),
        "--output", str(tmp_path),
        "--dry-run",
    ]) == 0

    assert len(calls) == 2
    assert all(call["dry_run"] is True for call in calls)
    assert all(call["modes"] == validation.VALIDATION_MODES for call in calls)
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["fresh_inference"] is True
    assert summary["counterfactual_recombination"] is False
    assert summary["training_invoked"] is False
    output = capsys.readouterr().out
    assert "python scripts/run_python_skill_gating_validation.py" in output
    assert "--seeds 11 22 33 44 55" in output


def test_dry_run_summary_keeps_unavailable_rates_and_deltas_null():
    rows = [
        {
            "seed": 11,
            "example_id": task,
            "task_type": task,
            "mode": mode,
            "execution_passed": None,
            "selected_skills": [],
            "active_adapter_parameters": 0,
            "benchmark_group": task,
        }
        for task in ("python_generation", "debugging", "test_generation")
        for mode in validation.VALIDATION_MODES
    ]
    summary = validation._build_summary(
        rows,
        [],
        {
            "generic": 30,
            "python_skill": 10,
            "debugging_skill": 10,
            "test_generation_skill": 10,
        },
        seeds=[11],
        benchmark_sha256="abc",
        dry_run=True,
    )
    candidate = summary["modes"]["python_only_for_test_generation"]
    assert candidate["overall_execution_pass_rate"] is None
    assert candidate["delta_vs_current_router"] is None
