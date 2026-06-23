#!/usr/bin/env python3
"""Train and evaluate a quarantined FastAPI contract candidate adapter."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

from skill_lattice_coder.config import ROOT, base_config, training_config
from skill_lattice_coder.inference import infer
from skill_lattice_coder.metrics import aggregate_results, extract_code, fuzzy_match, python_syntax_valid
from skill_lattice_coder.model_loader import generate_text, load_model
from skill_lattice_coder.schemas import DatasetExample, ExecutionFixture
from skill_lattice_coder.train_skill import _saved_parameter_count, _training_command
from skill_lattice_coder.utils import run_fixture, write_json


CANDIDATE = "api_contract_fastapi_skill"
VERSION = "v1"
TRAIN_SHA256 = "207651bc7ce884a7b71a9e9e33a852b10c4a4acc19559849c55b055e71038954"
HOLDOUT_SHA256 = "86f4e1dec7ebe3321484a43f3d7483a4d825fc958fcecd6385d9186f3269b2f1"
FASTAPI_BENCHMARK_SHA256 = "05f903fbdb5271e15ebee6edb6d2583f02724678ae946b93817a17b5d9f6d85e"
EVAL_SHA256 = "0ec79d983ba1a9ee2363789288242843e46c78fc0ed997b5a934c2978b89bcc6"
ROUTER_FILES = (
    ROOT / "src/skill_lattice_coder/router.py",
    ROOT / "src/skill_lattice_coder/inference.py",
    ROOT / "src/skill_lattice_coder/schemas.py",
)
REGISTRY_PATH = ROOT / "configs/skill_registry.json"
TRAIN_PATH = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/train.jsonl"
HOLDOUT_PATH = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/holdout.jsonl"
MANIFEST_PATH = ROOT / "data/failure_born/api_contract_fastapi_skill/v1/manifest.json"
FASTAPI_BENCHMARK_PATH = ROOT / "data/benchmarks/fastapi_contract/v1/benchmark.jsonl"
EVAL_PATH = ROOT / "data/eval.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/api_contract_fastapi_skill"
DEFAULT_PROMOTED_ADAPTER_EXPERIMENT = (
    ROOT / "artifacts/experiments/failure-born-skill/alternating_skill"
)
TASK_MAP = {
    "fastapi_contract_generation": "python_generation",
    "fastapi_contract_debugging": "debugging",
    "fastapi_contract_test_generation": "test_generation",
    "fastapi_contract_refactor": "debugging",
}
TRAINING_SEED = 42
EVALUATION_MODES = ("base", "candidate_explicit", "skillcortex_router_v1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _router_snapshot() -> dict[str, str]:
    return {str(path): _sha256(path) for path in (*ROUTER_FILES, REGISTRY_PATH)}


def _load_candidate_rows(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row.get("id")
        if not example_id or example_id in seen:
            raise ValueError(f"{path}:{number}: invalid or duplicate id")
        if row.get("candidate_skill") != CANDIDATE:
            raise ValueError(f"{path}:{number}: unexpected candidate skill")
        if row.get("benchmark_family") != "fastapi_contract":
            raise ValueError(f"{path}:{number}: unexpected benchmark family")
        if row.get("task_type") not in TASK_MAP:
            raise ValueError(f"{path}:{number}: unknown task type")
        seen.add(example_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _load_fastapi_benchmark_rows(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row.get("id")
        if not example_id or example_id in seen:
            raise ValueError(f"{path}:{number}: invalid or duplicate id")
        if row.get("benchmark_family") != "fastapi_contract":
            raise ValueError(f"{path}:{number}: unexpected benchmark family")
        if row.get("task_type") not in TASK_MAP:
            raise ValueError(f"{path}:{number}: unknown task type")
        seen.add(example_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _load_eval_examples(path: Path) -> list[DatasetExample]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        example = DatasetExample.from_dict(json.loads(line))
        if example.id in seen:
            raise ValueError(f"{path}:{number}: duplicate id")
        seen.add(example.id)
        rows.append(example)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _write_candidate_training_dataset(rows: list[dict], directory: Path) -> tuple[Path, int, int]:
    directory.mkdir(parents=True, exist_ok=True)
    valid = rows[::10]
    train = [row for index, row in enumerate(rows) if index % 10]
    for name, dataset in (("train.jsonl", train), ("valid.jsonl", valid)):
        (directory / name).write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": row["target"]},
                        ]
                    }
                )
                + "\n"
                for row in dataset
            )
        )
    return directory, len(train), len(valid)


def _prepare_training(output: Path, train_rows: list[dict], seed: int) -> dict:
    training_dir, optimizer_train_examples, optimizer_valid_examples = _write_candidate_training_dataset(
        train_rows, output / f"seed-{seed}" / "training-data"
    )
    candidate_dir = output / f"seed-{seed}" / "adapters" / CANDIDATE
    command = _training_command(training_dir, candidate_dir, rank=8, seed=seed)
    return {
        "seed": seed,
        "training_dir": training_dir,
        "candidate_dir": candidate_dir,
        "command": command,
        "optimizer_train_examples": optimizer_train_examples,
        "optimizer_valid_examples": optimizer_valid_examples,
    }


def _candidate_metadata(train_rows: list[dict], *, seed: int, elapsed: float, trainable_parameters: int | None, prepared: dict) -> dict:
    return {
        "adapter": CANDIDATE,
        "base_model": base_config()["model"],
        "source_model": base_config()["source_model"],
        "quantization": "4bit",
        "dataset_size": len(train_rows),
        "dataset_hash": hashlib.sha256(
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in train_rows
            ).encode()
        ).hexdigest(),
        "rank": 8,
        "target_modules": training_config()["target_modules"],
        "seed": seed,
        "iterations": training_config()["iterations"],
        "elapsed_seconds": elapsed,
        "trainable_parameters": trainable_parameters,
        "config": {
            **training_config(),
            "seed": seed,
            "rank": 8,
        },
        "quarantined": True,
        "active_by_default": False,
        "candidate_only": True,
        "training_command": prepared["command"],
        "optimizer_train_examples": prepared["optimizer_train_examples"],
        "optimizer_valid_examples": prepared["optimizer_valid_examples"],
        "train_examples_used": len(train_rows),
        "holdout_used_for_training": False,
    }


def _train_candidate(prepared: dict, train_rows: list[dict], *, force: bool, dry_run: bool) -> dict:
    candidate_dir = prepared["candidate_dir"]
    metadata_path = candidate_dir / "metadata.json"
    if candidate_dir.exists() and any(candidate_dir.iterdir()) and not force:
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
        raise FileExistsError(f"{candidate_dir} exists; pass --force to replace it")
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    if dry_run:
        metadata = _candidate_metadata(
            train_rows,
            seed=prepared["seed"],
            elapsed=0.0,
            trainable_parameters=None,
            prepared=prepared,
        )
        return metadata
    start = time.perf_counter()
    subprocess.run(prepared["command"], check=True)
    metadata = _candidate_metadata(
        train_rows,
        seed=prepared["seed"],
        elapsed=time.perf_counter() - start,
        trainable_parameters=_saved_parameter_count(candidate_dir),
        prepared=prepared,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _promoted_adapters_ready(root: Path) -> bool:
    adapter_root = root / "seed-11" / "adapters"
    required = (
        "python_skill",
        "debugging_skill",
        "test_generation_skill",
        "alternating_skill",
    )
    return all((adapter_root / name / "adapters.safetensors").exists() for name in required)


def _candidate_model(model_cache: dict, candidate_dir: Path):
    cache_key = ("candidate_explicit", str(candidate_dir))
    if cache_key not in model_cache:
        model_cache[cache_key] = load_model(candidate_dir)
    return model_cache[cache_key]


def _run_prompt(prompt: str, *, mode: str, candidate_dir: Path | None, adapter_root: Path | None, model_cache: dict, task_type: str | None = None, semantic_family: str | None = None) -> tuple[str, dict]:
    if mode == "base":
        generation = infer("base", prompt, dry_run=False, model_cache=model_cache)
        return extract_code(generation.generation), {
            "selected_skills": generation.selected_skills,
            "active_adapter_count": generation.active_adapter_count,
            "active_adapter_parameters": generation.active_adapter_parameters,
            "latency_seconds": generation.latency_seconds,
            "prompt_tokens": generation.prompt_tokens,
            "generated_tokens": generation.generated_tokens,
            "peak_memory_bytes": generation.peak_memory_bytes,
        }
    if mode == "candidate_explicit":
        model, tokenizer = _candidate_model(model_cache, candidate_dir)
        start = time.perf_counter()
        generation, prompt_tokens, generated_tokens = generate_text(model, tokenizer, prompt)
        return extract_code(generation), {
            "selected_skills": [CANDIDATE],
            "active_adapter_count": 1,
            "active_adapter_parameters": int(
                json.loads((candidate_dir / "metadata.json").read_text())["trainable_parameters"]
            ),
            "latency_seconds": time.perf_counter() - start,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "peak_memory_bytes": None,
        }
    generation = infer(
        "lattice",
        prompt,
        task_type=task_type,
        semantic_family=semantic_family,
        router_policy="skillcortex_router_v1",
        adapter_root=adapter_root,
        dry_run=False,
        model_cache=model_cache,
    )
    return extract_code(generation.generation), {
        "selected_skills": generation.selected_skills,
        "active_adapter_count": generation.active_adapter_count,
        "active_adapter_parameters": generation.active_adapter_parameters,
        "latency_seconds": generation.latency_seconds,
        "prompt_tokens": generation.prompt_tokens,
        "generated_tokens": generation.generated_tokens,
        "peak_memory_bytes": generation.peak_memory_bytes,
    }


def _evaluate_custom_rows(rows: list[dict], *, output: Path, candidate_dir: Path, adapter_root: Path | None, include_router: bool, dry_run: bool) -> dict:
    modes = ["base", "candidate_explicit"] + (["skillcortex_router_v1"] if include_router else [])
    model_cache = {}
    result_rows = []
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("w") as handle:
        for example in rows:
            for mode in modes:
                try:
                    if dry_run:
                        text = example["target"]
                        stats = {
                            "selected_skills": [] if mode == "base" else ([CANDIDATE] if mode == "candidate_explicit" else []),
                            "active_adapter_count": 0 if mode == "base" else 1,
                            "active_adapter_parameters": 0,
                            "latency_seconds": 0.0,
                            "prompt_tokens": None,
                            "generated_tokens": None,
                            "peak_memory_bytes": None,
                        }
                    else:
                        text, stats = _run_prompt(
                            example["prompt"],
                            mode=mode,
                            candidate_dir=candidate_dir,
                            adapter_root=adapter_root,
                            model_cache=model_cache,
                            task_type=TASK_MAP[example["task_type"]],
                            semantic_family="fastapi_contract",
                        )
                    execution_passed = None
                    if not dry_run:
                        execution_passed, _ = run_fixture(
                            ExecutionFixture.from_dict(example["execution"]),
                            text,
                        )
                    row = {
                        "example_id": example["id"],
                        "mode": mode,
                        "task_type": example["task_type"],
                        "benchmark_group": example["behavior_group"],
                        "domain": example["domain"],
                        "generation": text,
                        "exact_match": text.strip() == example["target"].strip(),
                        "fuzzy_score": fuzzy_match(text, example["target"]),
                        "syntax_valid": python_syntax_valid(text),
                        "execution_passed": execution_passed,
                        "error": None,
                        **stats,
                    }
                except Exception as error:  # ponytail: keep the rest of the evaluation running.
                    row = {
                        "example_id": example["id"],
                        "mode": mode,
                        "task_type": example["task_type"],
                        "benchmark_group": example["behavior_group"],
                        "domain": example["domain"],
                        "generation": "",
                        "exact_match": False,
                        "fuzzy_score": 0.0,
                        "syntax_valid": None,
                        "execution_passed": None,
                        "selected_skills": [],
                        "active_adapter_count": 0,
                        "active_adapter_parameters": 0,
                        "latency_seconds": 0.0,
                        "prompt_tokens": None,
                        "generated_tokens": None,
                        "peak_memory_bytes": None,
                        "error": str(error),
                    }
                result_rows.append(row)
                handle.write(json.dumps(row) + "\n")
    summary = {
        "modes": aggregate_results(result_rows),
        "errors": [row for row in result_rows if row["error"]],
    }
    write_json(output / "summary.json", summary)
    return {"rows": result_rows, "summary": summary}


def _evaluate_eval_rows(rows: list[DatasetExample], *, output: Path, candidate_dir: Path, adapter_root: Path | None, include_router: bool, dry_run: bool) -> dict:
    modes = ["base", "candidate_explicit"] + (["skillcortex_router_v1"] if include_router else [])
    model_cache = {}
    result_rows = []
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("w") as handle:
        for example in rows:
            for mode in modes:
                try:
                    if dry_run:
                        text = example.target
                        stats = {
                            "selected_skills": [] if mode == "base" else ([CANDIDATE] if mode == "candidate_explicit" else []),
                            "active_adapter_count": 0 if mode == "base" else 1,
                            "active_adapter_parameters": 0,
                            "latency_seconds": 0.0,
                            "prompt_tokens": None,
                            "generated_tokens": None,
                            "peak_memory_bytes": None,
                        }
                    else:
                        text, stats = _run_prompt(
                            example.prompt,
                            mode=mode,
                            candidate_dir=candidate_dir,
                            adapter_root=adapter_root,
                            model_cache=model_cache,
                            task_type=example.task_type,
                            semantic_family=example.group,
                        )
                    execution_passed = None
                    if example.execution and not dry_run:
                        execution_passed, _ = run_fixture(example.execution, text)
                    row = {
                        "example_id": example.id,
                        "mode": mode,
                        "task_type": example.task_type,
                        "benchmark_group": example.group,
                        "generation": text,
                        "exact_match": text.strip() == example.target.strip(),
                        "fuzzy_score": fuzzy_match(text, example.target),
                        "syntax_valid": None if example.task_type == "test_generation" else python_syntax_valid(text),
                        "execution_passed": execution_passed,
                        "error": None,
                        **stats,
                    }
                except Exception as error:  # ponytail: keep the rest of the evaluation running.
                    row = {
                        "example_id": example.id,
                        "mode": mode,
                        "task_type": example.task_type,
                        "benchmark_group": example.group,
                        "generation": "",
                        "exact_match": False,
                        "fuzzy_score": 0.0,
                        "syntax_valid": None,
                        "execution_passed": None,
                        "selected_skills": [],
                        "active_adapter_count": 0,
                        "active_adapter_parameters": 0,
                        "latency_seconds": 0.0,
                        "prompt_tokens": None,
                        "generated_tokens": None,
                        "peak_memory_bytes": None,
                        "error": str(error),
                    }
                result_rows.append(row)
                handle.write(json.dumps(row) + "\n")
    summary = {
        "modes": aggregate_results(result_rows),
        "errors": [row for row in result_rows if row["error"]],
    }
    write_json(output / "summary.json", summary)
    return {"rows": result_rows, "summary": summary}


def _rate(rows: list[dict], mode: str) -> float | None:
    values = [
        bool(row["execution_passed"])
        for row in rows
        if row["mode"] == mode and row.get("execution_passed") is not None
    ]
    return mean(values) if values else None


def _regressions(rows: list[dict], baseline: str, candidate: str) -> dict:
    indexed = {
        (row["example_id"], row["mode"]): row.get("execution_passed")
        for row in rows
        if row.get("execution_passed") is not None
    }
    keys = {key for key, mode in indexed if mode == baseline} & {key for key, mode in indexed if mode == candidate}
    fail_to_pass = 0
    pass_to_fail = 0
    for key in keys:
        base_value = bool(indexed[(key, baseline)])
        candidate_value = bool(indexed[(key, candidate)])
        fail_to_pass += (not base_value) and candidate_value
        pass_to_fail += base_value and (not candidate_value)
    return {
        "fail_to_pass": fail_to_pass,
        "pass_to_fail": pass_to_fail,
    }


def _subset(rows: list[dict], predicate) -> list[dict]:
    return [row for row in rows if predicate(row)]


def _build_report(*, output: Path, metadata: dict, train_eval: dict, holdout_eval: dict, benchmark_eval: dict, eval_regression: dict, include_router: bool, router_snapshot_before: dict[str, str], router_snapshot_after: dict[str, str], capacity: dict, hashes: dict, dry_run: bool, blocked: list[str]) -> dict:
    fixed_rows = benchmark_eval["rows"]
    eval_rows = eval_regression["rows"]
    modes_available = ["base", "candidate_explicit"] + (["skillcortex_router_v1"] if include_router else [])
    alternating_rows = _subset(
        eval_rows,
        lambda row: row.get("benchmark_group") == "alternating" and row["task_type"] in {"debugging", "test_generation"},
    )
    non_target_rows = _subset(eval_rows, lambda row: row.get("benchmark_group") != "alternating")
    report = {
        "status": "dry-run" if dry_run else "complete",
        "candidate_skill": CANDIDATE,
        "version": VERSION,
        "training": {
            "occurred": not dry_run,
            "command": metadata["training_command"],
            "hyperparameters": {
                "rank": 8,
                "iterations": metadata["iterations"],
                "learning_rate": training_config()["learning_rate"],
                "batch_size": training_config()["batch_size"],
                "lora_layers": training_config()["lora_layers"],
                "target_modules": training_config()["target_modules"],
                "seed": metadata["seed"],
            },
            "duration_seconds": metadata["elapsed_seconds"],
            "train_examples_used": 96,
            "optimizer_train_examples": metadata["optimizer_train_examples"],
            "optimizer_valid_examples": metadata["optimizer_valid_examples"],
            "holdout_used_for_training": False,
            "adapter_parameter_count": metadata["trainable_parameters"],
            "artifact_path": str(output / f"seed-{metadata['seed']}" / "adapters" / CANDIDATE),
            "random_seeds": [metadata["seed"]],
        },
        "input_hashes": hashes,
        "governance": {
            "router_changed": router_snapshot_before != router_snapshot_after,
            "registry_changed": router_snapshot_before[str(REGISTRY_PATH)] != router_snapshot_after[str(REGISTRY_PATH)],
            "candidate_activated": False,
            "candidate_promoted": False,
            "capacity": capacity,
            "blocked_checks": blocked,
        },
        "evaluation": {
            "modes_available": modes_available,
            "candidate_train": train_eval["summary"]["modes"],
            "candidate_holdout": holdout_eval["summary"]["modes"],
            "fixed_fastapi_benchmark": benchmark_eval["summary"]["modes"],
            "non_target_eval": eval_regression["summary"]["modes"],
            "fixed_benchmark_regression": {
                "base_vs_candidate": _regressions(fixed_rows, "base", "candidate_explicit"),
                "router_vs_candidate": (
                    _regressions(fixed_rows, "skillcortex_router_v1", "candidate_explicit")
                    if include_router
                    else None
                ),
            },
            "alternating_skill_regression": {
                "rows": len(alternating_rows) // max(len(modes_available), 1),
                "base_vs_candidate": _regressions(alternating_rows, "base", "candidate_explicit") if alternating_rows else None,
                "router_vs_candidate": (
                    _regressions(alternating_rows, "skillcortex_router_v1", "candidate_explicit")
                    if include_router and alternating_rows
                    else None
                ),
            },
            "non_target_regression": {
                "rows": len(non_target_rows) // max(len(modes_available), 1),
                "base_vs_candidate": _regressions(non_target_rows, "base", "candidate_explicit"),
                "router_vs_candidate": (
                    _regressions(non_target_rows, "skillcortex_router_v1", "candidate_explicit")
                    if include_router
                    else None
                ),
            },
        },
    }
    holdout_candidate = holdout_eval["summary"]["modes"].get("candidate_explicit", {})
    holdout_base = holdout_eval["summary"]["modes"].get("base", {})
    fixed_candidate = benchmark_eval["summary"]["modes"].get("candidate_explicit", {})
    fixed_base = benchmark_eval["summary"]["modes"].get("base", {})
    non_target_losses = report["evaluation"]["non_target_regression"]["base_vs_candidate"]["pass_to_fail"]
    holdout_improved = (
        holdout_candidate.get("execution_pass_rate") is not None
        and holdout_base.get("execution_pass_rate") is not None
        and holdout_candidate["execution_pass_rate"] > holdout_base["execution_pass_rate"]
    )
    fixed_improved = (
        fixed_candidate.get("execution_pass_rate") is not None
        and fixed_base.get("execution_pass_rate") is not None
        and fixed_candidate["execution_pass_rate"] > fixed_base["execution_pass_rate"]
    )
    if blocked:
        recommendation = "proceed_to_quarantined_candidate_validation"
    elif not holdout_improved:
        recommendation = "reject_candidate"
    elif non_target_losses > 0 or not fixed_improved:
        recommendation = "revise_training_or_data"
    else:
        recommendation = "proceed_to_quarantined_candidate_validation"
    report["recommendation"] = recommendation
    write_json(output / "summary.json", report)
    (output / "summary.md").write_text(_markdown(report))
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _mode_line(summary: dict, mode: str) -> str:
    values = summary.get(mode)
    if not values:
        return f"- `{mode}`: unavailable."
    return (
        f"- `{mode}`: execution {_pct(values.get('execution_pass_rate'))}; "
        f"fuzzy {values.get('fuzzy_score', 0):.3f}; "
        f"active {values.get('active_adapter_parameters', 0):.0f}."
    )


def _markdown(report: dict) -> str:
    lines = [
        f"# Quarantined Candidate Training: `{CANDIDATE}`",
        "",
        f"- Status: `{report['status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "- Quarantined: **true**",
        "- Active by default: **false**",
        "- Promoted: **false**",
        "",
        "## Training",
        "",
        f"- Training occurred: **{str(report['training']['occurred']).lower()}**",
        f"- Adapter parameter count: **{report['training']['adapter_parameter_count']}**",
        f"- Capacity after candidate: **{report['governance']['capacity']['projected_total']} / {report['governance']['capacity']['max_total_adapter_parameters']}**",
        f"- Holdout used for training: **{str(report['training']['holdout_used_for_training']).lower()}**",
        "",
        "## Candidate train",
        "",
        _mode_line(report["evaluation"]["candidate_train"], "base"),
        _mode_line(report["evaluation"]["candidate_train"], "candidate_explicit"),
        _mode_line(report["evaluation"]["candidate_train"], "skillcortex_router_v1"),
        "",
        "## Candidate holdout",
        "",
        _mode_line(report["evaluation"]["candidate_holdout"], "base"),
        _mode_line(report["evaluation"]["candidate_holdout"], "candidate_explicit"),
        _mode_line(report["evaluation"]["candidate_holdout"], "skillcortex_router_v1"),
        "",
        "## Frozen FastAPI benchmark",
        "",
        _mode_line(report["evaluation"]["fixed_fastapi_benchmark"], "base"),
        _mode_line(report["evaluation"]["fixed_fastapi_benchmark"], "candidate_explicit"),
        _mode_line(report["evaluation"]["fixed_fastapi_benchmark"], "skillcortex_router_v1"),
        "",
        "## Governance",
        "",
        f"- Router changed: **{str(report['governance']['router_changed']).lower()}**",
        f"- Registry changed: **{str(report['governance']['registry_changed']).lower()}**",
        f"- Candidate activated: **{str(report['governance']['candidate_activated']).lower()}**",
        f"- Candidate promoted: **{str(report['governance']['candidate_promoted']).lower()}**",
    ]
    if report["governance"]["blocked_checks"]:
        lines.extend(
            [
                "",
                "## Blocked checks",
                "",
                *[f"- {item}" for item in report["governance"]["blocked_checks"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _capacity_from_registry(registry: dict) -> dict:
    budget = registry["capacity_budget"]
    expected_rank8_impact = 311296
    projected_total = budget["current_total_adapter_parameters"] + expected_rank8_impact
    if expected_rank8_impact != 311296:
        raise ValueError("unexpected rank-8 impact")
    if budget["current_total_adapter_parameters"] != 1245184:
        raise ValueError("unexpected current adapter parameter count")
    if budget["max_total_adapter_parameters"] != 2500000:
        raise ValueError("unexpected max adapter parameter budget")
    if projected_total != 1556480:
        raise ValueError("unexpected projected adapter parameter total")
    return {
        "expected_rank8_impact": expected_rank8_impact,
        "current_total_adapter_parameters": budget["current_total_adapter_parameters"],
        "max_total_adapter_parameters": budget["max_total_adapter_parameters"],
        "projected_total": projected_total,
        "within_limit": projected_total <= budget["max_total_adapter_parameters"],
    }


def _verify_preconditions(promoted_adapter_experiment: Path) -> tuple[dict, dict, dict[str, str], list[str]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    registry = json.loads(REGISTRY_PATH.read_text())
    hashes = {
        "train_sha256": _sha256(TRAIN_PATH),
        "holdout_sha256": _sha256(HOLDOUT_PATH),
        "fastapi_benchmark_sha256": _sha256(FASTAPI_BENCHMARK_PATH),
        "eval_sha256": _sha256(EVAL_PATH),
    }
    if hashes["train_sha256"] != TRAIN_SHA256:
        raise ValueError("candidate train SHA-256 mismatch")
    if hashes["holdout_sha256"] != HOLDOUT_SHA256:
        raise ValueError("candidate holdout SHA-256 mismatch")
    if hashes["fastapi_benchmark_sha256"] != FASTAPI_BENCHMARK_SHA256:
        raise ValueError("frozen FastAPI benchmark SHA-256 mismatch")
    if hashes["eval_sha256"] != EVAL_SHA256:
        raise ValueError("data/eval.jsonl SHA-256 mismatch")
    if manifest.get("train_sha256") != TRAIN_SHA256 or manifest.get("holdout_sha256") != HOLDOUT_SHA256:
        raise ValueError("candidate manifest hashes do not match approved frozen hashes")
    names = {skill["skill_name"]: skill for skill in registry["skills"]}
    if CANDIDATE in names:
        raise ValueError("candidate is already present in registry")
    router_snapshot = _router_snapshot()
    blocked = []
    if not _promoted_adapters_ready(promoted_adapter_experiment):
        blocked.append(
            f"missing promoted adapters under {promoted_adapter_experiment}; skillcortex_router_v1 comparisons will be omitted"
        )
    return manifest, registry, router_snapshot, blocked


def _record_metadata(metadata: dict, prepared: dict) -> dict:
    metadata = dict(metadata)
    metadata["training_command"] = prepared["command"]
    metadata["optimizer_train_examples"] = prepared["optimizer_train_examples"]
    metadata["optimizer_valid_examples"] = prepared["optimizer_valid_examples"]
    return metadata


def run(*, output: Path, promoted_adapter_experiment: Path, seed: int, dry_run: bool, force: bool) -> dict:
    if base_config()["temperature"] != 0.0:
        raise ValueError("candidate training requires temperature 0.0 for evaluation")
    manifest, registry, router_snapshot_before, blocked = _verify_preconditions(promoted_adapter_experiment)
    capacity = _capacity_from_registry(registry)
    train_rows = _load_candidate_rows(TRAIN_PATH)
    holdout_rows = _load_candidate_rows(HOLDOUT_PATH)
    benchmark_rows = _load_fastapi_benchmark_rows(FASTAPI_BENCHMARK_PATH)
    eval_rows = _load_eval_examples(EVAL_PATH)
    prepared = _prepare_training(output, train_rows, seed)
    metadata = _record_metadata(
        _train_candidate(prepared, train_rows, force=force, dry_run=dry_run),
        prepared,
    )
    include_router = not blocked
    adapter_root = promoted_adapter_experiment / "seed-11" / "adapters" if include_router else None
    candidate_dir = prepared["candidate_dir"]
    train_eval = _evaluate_custom_rows(
        train_rows,
        output=output / f"seed-{seed}" / "train-eval",
        candidate_dir=candidate_dir,
        adapter_root=adapter_root,
        include_router=include_router,
        dry_run=dry_run,
    )
    holdout_eval = _evaluate_custom_rows(
        holdout_rows,
        output=output / f"seed-{seed}" / "holdout-eval",
        candidate_dir=candidate_dir,
        adapter_root=adapter_root,
        include_router=include_router,
        dry_run=dry_run,
    )
    benchmark_eval = _evaluate_custom_rows(
        benchmark_rows,
        output=output / f"seed-{seed}" / "fixed-benchmark-eval",
        candidate_dir=candidate_dir,
        adapter_root=adapter_root,
        include_router=include_router,
        dry_run=dry_run,
    )
    eval_regression = _evaluate_eval_rows(
        eval_rows,
        output=output / f"seed-{seed}" / "eval-regression",
        candidate_dir=candidate_dir,
        adapter_root=adapter_root,
        include_router=include_router,
        dry_run=dry_run,
    )
    if _sha256(TRAIN_PATH) != TRAIN_SHA256 or _sha256(HOLDOUT_PATH) != HOLDOUT_SHA256:
        raise RuntimeError("frozen candidate data changed during training")
    if _sha256(FASTAPI_BENCHMARK_PATH) != FASTAPI_BENCHMARK_SHA256:
        raise RuntimeError("frozen FastAPI benchmark changed during training")
    if _sha256(EVAL_PATH) != EVAL_SHA256:
        raise RuntimeError("data/eval.jsonl changed during training")
    router_snapshot_after = _router_snapshot()
    return _build_report(
        output=output,
        metadata=metadata,
        train_eval=train_eval,
        holdout_eval=holdout_eval,
        benchmark_eval=benchmark_eval,
        eval_regression=eval_regression,
        include_router=include_router,
        router_snapshot_before=router_snapshot_before,
        router_snapshot_after=router_snapshot_after,
        capacity=capacity,
        hashes={
            "train": TRAIN_SHA256,
            "holdout": HOLDOUT_SHA256,
            "fixed_fastapi_benchmark": FASTAPI_BENCHMARK_SHA256,
            "eval": EVAL_SHA256,
            "combined": manifest["combined_sha256"],
        },
        dry_run=dry_run,
        blocked=blocked,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--promoted-adapter-experiment",
        default=str(DEFAULT_PROMOTED_ADAPTER_EXPERIMENT),
    )
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = run(
        output=Path(args.output),
        promoted_adapter_experiment=Path(args.promoted_adapter_experiment),
        seed=args.seed,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(json.dumps({"summary": str(Path(args.output) / "summary.json"), "recommendation": summary["recommendation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())