import hashlib
import json
from pathlib import Path

from scripts.select_failure_cluster import build_cluster_selection, main


ROOT = Path(__file__).resolve().parents[1]


def _row(seed, family, task, mode, passed, skills):
    return {
        "seed": seed,
        "example_id": f"{family}-{task}",
        "benchmark_group": family,
        "task_type": task,
        "mode": mode,
        "execution_passed": passed,
        "selected_skills": skills,
    }


def test_cluster_ranking_prefers_repeated_non_python_localized_failures():
    rows = []
    for seed in (11, 22, 33):
        for task in ("debugging", "test_generation"):
            for mode, passed, skills in (
                ("base", False, []),
                ("lattice", False, ["debugging_skill"]),
                ("oracle-lattice", False, ["python_skill", "debugging_skill"]),
                ("protected_skill_router", False, ["debugging_skill", "python_skill"]),
            ):
                rows.append(_row(seed, "localized", task, mode, passed, skills))
        for mode, passed, skills in (
            ("base", True, []),
            ("lattice", False, ["python_skill"]),
            ("oracle-lattice", False, ["python_skill"]),
            ("protected_skill_router", False, []),
        ):
            rows.append(
                _row(seed, "base_fallback", "python_generation", mode, passed, skills)
            )

    result = build_cluster_selection(rows, seeds=[11, 22, 33], benchmark_sha256="abc")

    assert result["recommended_primary_cluster"] == {
        "semantic_family": "localized",
        "task_type": "debugging",
        "candidate_skill_name": "localized_skill",
    }
    first = result["clusters"][0]
    assert first["fail_count"] == 3
    assert first["failure_count_by_seed"] == {"11": 1, "22": 1, "33": 1}
    assert first["family_non_python_fail_count"] == 6
    assert first["protected_router_failed"] is True
    assert first["distinct_benchmark_examples"] == 1
    assert first["enough_repeated_failures_for_candidate_design"] is True
    assert first["enough_independent_examples_for_promotion"] is False
    assert first["likely_skill_specific"] is True
    python = next(
        cluster
        for cluster in result["clusters"]
        if cluster["task_type"] == "python_generation"
    )
    assert python["eligible_for_failure_born_skill"] is False


def test_main_writes_only_cluster_selection_and_preserves_benchmark(tmp_path):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()

    assert main([
        "--seeds", "11", "22", "33", "44", "55",
        "--dataset", str(benchmark),
        "--validation-experiment", str(
            ROOT / "artifacts/experiments/python-skill-gating-validation"
        ),
        "--output", str(tmp_path),
    ]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cluster_selection.json",
        "cluster_selection.md",
    ]
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "cluster_selection.json").read_text())
    assert len(summary["recommended_primary_cluster"]) == 3
    assert summary["source"]["router"] == "protected_skill_router"
    assert summary["evaluation_leakage_warning"]
