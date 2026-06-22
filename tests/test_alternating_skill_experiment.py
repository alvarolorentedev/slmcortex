import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_alternating_skill_data import build_datasets
from scripts.run_alternating_skill_experiment import (
    promotion_decision,
    main,
)
from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.router import (
    ProtectedRouterPlusAlternatingSkill,
    ProtectedSkillRouter,
)


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_data_counts_metadata_and_no_benchmark_reuse(tmp_path):
    train_path, holdout_path = build_datasets(tmp_path)
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    holdout = [json.loads(line) for line in holdout_path.read_text().splitlines()]
    benchmark = [
        json.loads(line) for line in (ROOT / "data/eval.jsonl").read_text().splitlines()
    ]

    assert len(train) == 40
    assert len(holdout) == 30
    assert {task: sum(row["task_type"] == task for row in train) for task in {"debugging", "test_generation"}} == {
        "debugging": 25,
        "test_generation": 15,
    }
    assert {task: sum(row["task_type"] == task for row in holdout) for task in {"debugging", "test_generation"}} == {
        "debugging": 15,
        "test_generation": 15,
    }
    for split, rows in (("train", train), ("holdout", holdout)):
        assert all(row["metadata"] == {
            "semantic_family": "alternating",
            "candidate_skill": "alternating_skill",
            "source": "synthetic_failure_born",
            "split": split,
            "task_type": row["task_type"],
            "evaluation_leakage_safe": True,
        } for row in rows)
        assert all(row["skills"] == ["alternating_skill"] for row in rows)

    for key in ("id", "prompt", "target"):
        assert not ({row[key] for row in train + holdout} & {row[key] for row in benchmark})
        assert not ({row[key] for row in train} & {row[key] for row in holdout})
    benchmark_fixtures = {
        json.dumps(row.get("execution"), sort_keys=True) for row in benchmark
    }
    assert not (
        {
            json.dumps(row.get("execution"), sort_keys=True)
            for row in train + holdout
        }
        & benchmark_fixtures
    )
    assert all("repair_alternating" not in json.dumps(row) for row in train + holdout)
    assert all("generated_alternating" not in json.dumps(row) for row in train + holdout)
    assert all("subject_alternating" not in json.dumps(row) for row in train + holdout)
    assert load_jsonl(train_path)


def test_candidate_router_is_quarantined_and_only_activates_target_family():
    router = ProtectedRouterPlusAlternatingSkill()
    debug = router.route("debugging", "alternating")
    tests = router.route("test_generation", "alternating")
    other = router.route("debugging", "overlay")
    default = ProtectedSkillRouter().route("debugging")

    assert debug.selected_skills == [
        "debugging_skill",
        "python_skill",
        "alternating_skill",
    ]
    assert tests.selected_skills == [
        "python_skill",
        "test_generation_skill",
        "alternating_skill",
    ]
    assert other.selected_skills == default.selected_skills
    assert "alternating_skill" not in default.selected_skills


@pytest.mark.parametrize(
    ("holdout_delta", "fixed_improved", "overall_delta", "non_target_losses", "expected"),
    [
        (0.25, True, 0.0, 0, "recommend_promotion"),
        (0.0, True, 0.0, 0, "discard_overfit"),
        (0.25, True, 0.0, 2, "keep_quarantined"),
        (0.0, False, 0.0, 0, "discard"),
    ],
)
def test_promotion_decision_is_deterministic(
    holdout_delta, fixed_improved, overall_delta, non_target_losses, expected
):
    assert promotion_decision(
        holdout_delta=holdout_delta,
        fixed_target_improved=fixed_improved,
        fixed_overall_delta=overall_delta,
        non_target_losses=non_target_losses,
    )["status"] == expected


def test_dry_run_preserves_benchmark_and_does_not_train(tmp_path, monkeypatch):
    benchmark = ROOT / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.run_alternating_skill_experiment._train_candidate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained")),
    )

    assert main([
        "--seeds", "11",
        "--dataset", str(benchmark),
        "--baseline-experiment", str(ROOT / "artifacts/experiments/five-seed"),
        "--protected-experiment", str(
            ROOT / "artifacts/experiments/python-skill-gating-validation"
        ),
        "--output", str(tmp_path),
        "--dry-run",
    ]) == 0

    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["quarantine"]["active_by_default"] is False
    assert summary["quarantine"]["auto_promote"] is False
    assert summary["training"]["trained_existing_skills"] == []
    assert summary["status"] == "dry-run"
    candidate = summary["fixed_benchmark"]["modes"][
        "protected_router_plus_alternating_skill"
    ]
    assert candidate["stored_adapter_parameters"] > 933_888
    assert summary["answers"]["promotion_status"] == "pending_evaluation"
