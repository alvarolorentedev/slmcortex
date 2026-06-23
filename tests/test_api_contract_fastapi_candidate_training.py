import hashlib
import json
from pathlib import Path

from scripts.run_api_contract_fastapi_candidate_training import main


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_preserves_frozen_surfaces_and_writes_quarantined_summary(tmp_path):
    frozen_paths = [
        ROOT / "data/failure_born/api_contract_fastapi_skill/v1/train.jsonl",
        ROOT / "data/failure_born/api_contract_fastapi_skill/v1/holdout.jsonl",
        ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl",
        ROOT / "data/eval.jsonl",
        ROOT / "src/skill_lattice_coder/router.py",
        ROOT / "src/skill_lattice_coder/inference.py",
        ROOT / "src/skill_lattice_coder/schemas.py",
        ROOT / "configs/skill_registry.json",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_paths}

    assert main(["--output", str(tmp_path), "--dry-run", "--force"]) == 0

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_paths}
    assert before == after

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["training"]["occurred"] is False
    assert summary["governance"]["candidate_activated"] is False
    assert summary["governance"]["candidate_promoted"] is False
    assert summary["governance"]["router_changed"] is False
    assert summary["governance"]["registry_changed"] is False
    assert summary["training"]["holdout_used_for_training"] is False
    assert summary["governance"]["capacity"]["projected_total"] == 1_556_480