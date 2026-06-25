import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/build_api_contract_fastapi_candidate_data_v2.py"
FASTAPI_BENCHMARK = ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl"
EXISTING_BENCHMARK = ROOT / "data/eval.jsonl"
V1_TRAIN = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/train.jsonl"
V1_HOLDOUT = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/holdout.jsonl"
V1_MANIFEST = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/manifest.json"
ROUTER = ROOT / "src/skill_lattice_coder/router.py"
REGISTRY = ROOT / "configs/skill_registry.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("fastapi_v2_candidate_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_check_build_is_deterministic_balanced_and_independent():
    module = _load_module()

    first = module.build_candidate_data_v2(execute_fixtures=False)
    second = module.build_candidate_data_v2(execute_fixtures=False)

    assert first == second
    assert len(first["train"]) == 48
    assert len(first["holdout"]) == 48
    assert len(first["train"] + first["holdout"]) == 96
    assert set(first["manifest"]["row_counts_by_split"].values()) == {48}
    assert set(first["manifest"]["row_counts_by_task_type_split"]["train"].values()) == {12}
    assert set(first["manifest"]["row_counts_by_task_type_split"]["holdout"].values()) == {12}
    assert set(first["manifest"]["row_counts_by_behavior_group_split"]["train"].values()) == {4}
    assert set(first["manifest"]["row_counts_by_behavior_group_split"]["holdout"].values()) == {4}
    assert first["manifest"]["validation_passed"] is True
    assert first["manifest"]["write_performed"] is False
    assert first["manifest"]["checksums"] == second["manifest"]["checksums"]

    train_domains = {row["domain"] for row in first["train"]}
    holdout_domains = {row["domain"] for row in first["holdout"]}
    assert train_domains.isdisjoint(holdout_domains)

    required = module.REQUIRED_FIELDS
    all_rows = first["train"] + first["holdout"]
    assert all(set(row) == required for row in all_rows)
    assert all(row["metadata"]["curation_decision"] == "accept" for row in all_rows)
    assert all(row["size_guard"]["estimated_target_tokens"] <= 192 for row in all_rows)
    assert all(row["size_guard"]["character_count"] <= 768 for row in all_rows)
    assert all(row["size_guard"]["non_empty_line_count"] <= 32 for row in all_rows)
    assert all(row["metadata"]["anchor_validation_result"] is True for row in all_rows)
    assert all(row["metadata"]["leakage_validation_result"] is True for row in all_rows)
    assert all(not row["shape_guard"]["no_app_test_mixing"] is False for row in all_rows)


def test_v2_check_mode_preserves_governance_files_and_writes_nothing(tmp_path):
    module = _load_module()
    before = {
        path: _hash(path)
        for path in (
            FASTAPI_BENCHMARK,
            EXISTING_BENCHMARK,
            V1_TRAIN,
            V1_HOLDOUT,
            V1_MANIFEST,
            ROUTER,
            REGISTRY,
        )
    }
    output = tmp_path / "candidate"

    assert module.main(["--check", "--output", str(output), "--skip-fixtures"]) == 0
    assert not output.exists()
    assert all(_hash(path) == digest for path, digest in before.items())


def test_v2_fixture_guards_and_mutants_pass_in_full_check():
    module = _load_module()
    result = module.build_candidate_data_v2(execute_fixtures=True)

    assert result["manifest"]["validation_passed"] is True
    test_rows = [row for row in result["train"] + result["holdout"] if row["task_type"] == "fastapi_contract_test_generation"]
    assert len(test_rows) == 24
    assert all(row["fixture_guard"]["passes_correct_app"] is True for row in test_rows)
    assert all(row["fixture_guard"]["fails_both_assigned_mutants"] is True for row in test_rows)
    assert all(len(row["fixture_guard"]["assigned_mutants"]) == 2 for row in test_rows)
    assert all(len(row["fixture_guard"]["mutant_sha256"]) == 2 for row in test_rows)


def test_v2_explicit_write_uses_only_requested_temporary_directory(tmp_path):
    module = _load_module()
    result = module.build_candidate_data_v2(execute_fixtures=False)
    output = tmp_path / "candidate"

    written = module.write_candidate_data_v2(result, output)

    assert written == output
    assert sorted(path.name for path in output.iterdir()) == [
        "holdout.jsonl",
        "manifest.json",
        "train.jsonl",
    ]
    assert len((output / "train.jsonl").read_text().splitlines()) == 48
    assert len((output / "holdout.jsonl").read_text().splitlines()) == 48
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["row_counts_by_split"] == {"train": 48, "holdout": 48}