import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_python_skill_gating import (
    POLICIES,
    build_summary,
    requested_skills,
    select_counterfactual_rows,
)


def _row(example, task, mode, skills, passed, family="family", parameters=0):
    return {
        "seed": 11,
        "example_id": example,
        "task_type": task,
        "mode": mode,
        "selected_skills": skills,
        "execution_passed": passed,
        "active_adapter_parameters": parameters,
        "benchmark_group": family,
    }


def test_policies_select_expected_skills():
    current = ["debugging_skill", "python_skill"]
    oracle = ["python_skill", "debugging_skill"]
    assert requested_skills("no_python_for_generation", "python_generation", current, ["python_skill"]) == []
    assert requested_skills("python_only_for_test_generation", "test_generation", ["test_generation_skill"], ["python_skill", "test_generation_skill"]) == ["python_skill", "test_generation_skill"]
    assert requested_skills("debugging_without_python", "debugging", current, oracle) == ["debugging_skill"]
    assert requested_skills("oracle_without_python_generation", "debugging", current, oracle) == oracle
    for policy in POLICIES:
        assert requested_skills(policy, "python_generation", ["python_skill"], ["python_skill"]) == []


def test_counterfactual_selection_fails_on_adapter_tuple_mismatch():
    rows = [
        _row("x", "python_generation", "base", ["python_skill"], True),
        _row("x", "python_generation", "lattice", ["python_skill"], False),
        _row("x", "python_generation", "oracle-lattice", ["python_skill"], False),
    ]
    with pytest.raises(ValueError, match="adapter tuple mismatch"):
        select_counterfactual_rows(rows, {"x": ["python_skill"]}, ["no_python_for_generation"])


def test_summary_has_counterfactual_metadata_deltas_and_parameters():
    rows = []
    for example, task, base_pass, generic_pass, lattice_pass, oracle_pass in (
        ("p", "python_generation", True, False, False, False),
        ("d", "debugging", False, True, True, True),
        ("t", "test_generation", False, False, True, True),
    ):
        oracle_skills = {
            "python_generation": ["python_skill"],
            "debugging": ["python_skill", "debugging_skill"],
            "test_generation": ["python_skill", "test_generation_skill"],
        }[task]
        current_skills = {
            "python_generation": ["python_skill"],
            "debugging": ["debugging_skill", "python_skill"],
            "test_generation": ["test_generation_skill"],
        }[task]
        rows += [
            _row(example, task, "base", [], base_pass),
            _row(example, task, "generic", [], generic_pass, parameters=30),
            _row(example, task, "lattice", current_skills, lattice_pass, parameters=20),
            _row(example, task, "oracle-lattice", oracle_skills, oracle_pass, parameters=20),
            _row(example, task, "single-skill", [{
                "python_generation": "python_skill",
                "debugging": "debugging_skill",
                "test_generation": "test_generation_skill",
            }[task]], lattice_pass, parameters=10),
        ]
    selected = select_counterfactual_rows(
        rows,
        {
            "p": ["python_skill"],
            "d": ["python_skill", "debugging_skill"],
            "t": ["python_skill", "test_generation_skill"],
        },
    )
    summary = build_summary(
        rows,
        selected,
        adapter_parameters={
            "generic": 30,
            "python_skill": 10,
            "debugging_skill": 10,
            "test_generation_skill": 10,
        },
        seeds=[11],
        benchmark_sha256="abc",
    )
    assert summary["is_counterfactual_recombination"] is True
    assert summary["requires_training"] is False
    assert summary["requires_new_inference"] is False
    policy = summary["policies"]["no_python_for_generation"]
    assert policy["source_modes_used"] == ["base", "lattice"]
    assert policy["policy_vs_current_lattice_delta"] == pytest.approx(1 / 3)
    assert policy["policy_vs_oracle_lattice_delta"] == pytest.approx(1 / 3)
    assert policy["policy_vs_base_delta"] == pytest.approx(2 / 3)
    assert policy["policy_vs_generic_delta"] == pytest.approx(2 / 3)
    assert policy["stored_adapter_parameters"] == 30
    assert policy["trainable_adapter_parameters"] == 30
    assert policy["active_stored_parameter_ratio"] == pytest.approx(40 / 3 / 30)
    assert summary["policies"]["generic"]["stored_adapter_parameters"] == 30
    assert summary["answers"]["best_python_recovery_policy"] == "no_python_for_generation"


def test_real_artifacts_are_complete_and_benchmark_is_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    benchmark = root / "data/eval.jsonl"
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    from scripts.run_python_skill_gating import main

    assert main([
        "--seeds", "11",
        "--baseline-experiment", str(root / "artifacts/experiments/five-seed"),
        "--dataset", str(benchmark),
        "--output", str(tmp_path),
    ]) == 0
    assert hashlib.sha256(benchmark.read_bytes()).hexdigest() == before
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert set(summary["policies"]) >= set(POLICIES)
    assert (tmp_path / "summary.md").exists()
