#!/usr/bin/env python3
"""Prepare and run the quarantined alternating-skill experiment."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import mean

from scripts.build_alternating_skill_data import build_datasets
from skill_lattice_coder.data import load_jsonl, write_mlx_dataset
from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.train_skill import (
    _metadata,
    _saved_parameter_count,
    _training_command,
)
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
MODES = ("protected_skill_router", "protected_router_plus_alternating_skill")


def promotion_decision(
    *,
    holdout_delta,
    fixed_target_improved,
    fixed_overall_delta,
    non_target_losses,
):
    if holdout_delta < 0.20:
        return {
            "status": "discard_overfit" if fixed_target_improved else "discard",
            "auto_promoted": False,
        }
    if non_target_losses > 0 or fixed_overall_delta < -0.005:
        return {"status": "keep_quarantined", "auto_promoted": False}
    if fixed_target_improved:
        return {"status": "recommend_promotion", "auto_promoted": False}
    return {"status": "discard", "auto_promoted": False}


def _snapshot(root, seeds):
    output = {}
    for seed in seeds:
        for path in sorted((Path(root) / f"seed-{seed}" / "adapters").rglob("*")):
            if path.is_file():
                stat = path.stat()
                output[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return output


def _link_existing(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("python_skill", "debugging_skill", "test_generation_skill"):
        target = destination / name
        source_path = source / name
        if not (source_path / "adapters.safetensors").exists():
            raise FileNotFoundError(f"missing existing adapter: {source_path}")
        if not target.exists():
            target.symlink_to(source_path.resolve(), target_is_directory=True)


def _prepare(seed, root, train_path):
    examples = load_jsonl(train_path)
    adapter_root = root / f"seed-{seed}" / "adapters"
    candidate = adapter_root / "alternating_skill"
    data = write_mlx_dataset(
        examples, root / f"seed-{seed}" / "alternating-training-data"
    )
    command = _training_command(data, candidate, rank=8, seed=seed)
    return examples, adapter_root, candidate, command


def _train_candidate(examples, candidate, command, seed, *, force):
    if candidate.exists() and any(candidate.iterdir()) and not force:
        metadata = candidate / "metadata.json"
        if metadata.exists():
            return json.loads(metadata.read_text())
        raise FileExistsError(f"{candidate} exists; pass --force to replace it")
    if candidate.exists():
        shutil.rmtree(candidate)
    start = time.perf_counter()
    subprocess.run(command, check=True)
    metadata = _metadata(
        "alternating_skill",
        examples,
        rank=8,
        elapsed=time.perf_counter() - start,
        seed=seed,
        iterations=100,
    )
    metadata.update(
        {
            "trainable_parameters": _saved_parameter_count(candidate),
            "quarantined": True,
            "semantic_family": "alternating",
        }
    )
    (candidate / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _read_rows(root, seeds, split):
    rows = []
    for seed in seeds:
        path = root / f"seed-{seed}" / split / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing results: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["seed"] = seed
            rows.append(row)
    return rows


def _rate(rows):
    values = [
        bool(row["execution_passed"])
        for row in rows
        if row.get("execution_passed") is not None
    ]
    return mean(values) if values else None


def _delta(left, right):
    return None if left is None or right is None else left - right


def _split(rows, mode):
    return [row for row in rows if row["mode"] == mode]


def _paired(protected, candidate):
    baseline = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in protected
        if row.get("execution_passed") is not None
    }
    comparison = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in candidate
        if row.get("execution_passed") is not None
    }
    keys = baseline.keys() & comparison.keys()
    return {
        "fail_to_pass": sum(not baseline[key] and comparison[key] for key in keys),
        "pass_to_fail": sum(baseline[key] and not comparison[key] for key in keys),
    }


def _active_parameters(rows, parameters):
    return (
        mean(
            sum(parameters[skill] for skill in row.get("selected_skills", []))
            for row in rows
        )
        if rows
        else None
    )


def _dataset_summary(rows, *, fixed, parameters):
    protected = _split(rows, "protected_skill_router")
    candidate = _split(rows, "protected_router_plus_alternating_skill")
    target = lambda row: row.get("benchmark_group") == "alternating" and row[
        "task_type"
    ] in {"debugging", "test_generation"}
    output = {}
    for name, mode_rows in (
        ("protected_skill_router", protected),
        ("protected_router_plus_alternating_skill", candidate),
    ):
        target_rows = [row for row in mode_rows if target(row)]
        non_target = [row for row in mode_rows if not target(row)]
        output[name] = {
            "overall_execution_pass_rate": _rate(mode_rows),
            "alternating_debugging_pass_rate": _rate(
                [
                    row
                    for row in target_rows
                    if row["task_type"] == "debugging"
                ]
            ),
            "alternating_test_generation_pass_rate": _rate(
                [
                    row
                    for row in target_rows
                    if row["task_type"] == "test_generation"
                ]
            ),
            "target_cluster_pass_rate": _rate(target_rows),
            "non_target_pass_rate": _rate(non_target),
            "python_generation_pass_rate": _rate(
                [row for row in mode_rows if row["task_type"] == "python_generation"]
            ),
            "debugging_pass_rate": _rate(
                [row for row in mode_rows if row["task_type"] == "debugging"]
            ),
            "test_generation_pass_rate": _rate(
                [row for row in mode_rows if row["task_type"] == "test_generation"]
            ),
            "active_adapter_parameters": _active_parameters(mode_rows, parameters),
            "stored_adapter_parameters": (
                sum(parameters[name] for name in (
                    "python_skill",
                    "debugging_skill",
                    "test_generation_skill",
                ))
                + (
                    parameters["alternating_skill"]
                    if name == "protected_router_plus_alternating_skill"
                    else 0
                )
            ),
            "trainable_adapter_parameters": (
                sum(parameters[name] for name in (
                    "python_skill",
                    "debugging_skill",
                    "test_generation_skill",
                ))
                + (
                    parameters["alternating_skill"]
                    if name == "protected_router_plus_alternating_skill"
                    else 0
                )
            ),
            "selected_skill_tuple_distribution": [
                {"selected_skills": list(skills), "count": count}
                for skills, count in sorted(
                    Counter(
                        tuple(row.get("selected_skills", [])) for row in mode_rows
                    ).items()
                )
            ],
        }
    transitions = _paired(protected, candidate)
    target_transitions = _paired(
        [row for row in protected if target(row)],
        [row for row in candidate if target(row)],
    )
    non_target_transitions = _paired(
        [row for row in protected if not target(row)],
        [row for row in candidate if not target(row)],
    )
    return {
        "kind": "fixed_benchmark" if fixed else "independent_holdout",
        "modes": output,
        "pass_fail_vs_protected": transitions,
        "target_cluster_wins_losses": target_transitions,
        "non_target_regressions": non_target_transitions["pass_to_fail"],
    }


def _build_summary(
    fixed_rows,
    holdout_rows,
    metadata,
    parameters,
    *,
    seeds,
    checksum,
    dry_run,
):
    fixed = _dataset_summary(fixed_rows, fixed=True, parameters=parameters)
    holdout = _dataset_summary(
        holdout_rows, fixed=False, parameters=parameters
    )
    candidate_fixed = fixed["modes"]["protected_router_plus_alternating_skill"]
    protected_fixed = fixed["modes"]["protected_skill_router"]
    candidate_holdout = holdout["modes"][
        "protected_router_plus_alternating_skill"
    ]
    protected_holdout = holdout["modes"]["protected_skill_router"]
    if dry_run:
        decision = {"status": "pending_evaluation", "auto_promoted": False}
    else:
        decision = promotion_decision(
            holdout_delta=_delta(
                candidate_holdout["target_cluster_pass_rate"],
                protected_holdout["target_cluster_pass_rate"],
            ),
            fixed_target_improved=(
                candidate_fixed["target_cluster_pass_rate"]
                > protected_fixed["target_cluster_pass_rate"]
            ),
            fixed_overall_delta=_delta(
                candidate_fixed["overall_execution_pass_rate"],
                protected_fixed["overall_execution_pass_rate"],
            ),
            non_target_losses=fixed["non_target_regressions"],
        )
    candidate_parameters = [
        value.get("trainable_parameters")
        for value in metadata.values()
        if value.get("trainable_parameters") is not None
    ]
    return {
        "status": "dry-run" if dry_run else "complete",
        "candidate_skill": "alternating_skill",
        "semantic_family": "alternating",
        "requires_training": True,
        "trained_existing_skills": [],
        "benchmark_sha256": checksum,
        "seeds": seeds,
        "quarantine": {
            "active_by_default": False,
            "candidate_router_only": True,
            "auto_promote": False,
        },
        "training": {
            "rank": 8,
            "train_examples": 40,
            "debugging_examples": 25,
            "test_generation_examples": 15,
            "trained_existing_skills": [],
            "candidate_trainable_parameters": (
                mean(candidate_parameters) if candidate_parameters else None
            ),
        },
        "fixed_benchmark": fixed,
        "independent_holdout": holdout,
        "promotion_decision": decision,
        "answers": {
            "failure_pattern": (
                "Select zero-based even indexes with values[::2]; common "
                "failures start at index one or filter by value."
            ),
            "data_created": (
                "40 synthetic training examples and 30 independent holdout "
                "examples, split across debugging and test generation."
            ),
            "improved_independent_holdout": (
                None
                if dry_run
                else candidate_holdout["target_cluster_pass_rate"]
                > protected_holdout["target_cluster_pass_rate"]
            ),
            "improved_fixed_benchmark_target": (
                None
                if dry_run
                else candidate_fixed["target_cluster_pass_rate"]
                > protected_fixed["target_cluster_pass_rate"]
            ),
            "caused_non_target_regression": (
                None if dry_run else fixed["non_target_regressions"] > 0
            ),
            "promotion_status": decision["status"],
            "validates_second_plastic_cortex_mechanism": (
                None
                if dry_run
                else decision["status"] == "recommend_promotion"
            ),
        },
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _markdown(summary):
    lines = [
        "# Failure-Born Skill Experiment 1: `alternating_skill`",
        "",
        f"- Status: `{summary['status']}`",
        "- Quarantined: **true**",
        "- Active by default: **false**",
        "- Auto-promotion: **disabled**",
        "",
        "## Pattern",
        "",
        "`alternating` selects elements at zero-based even indexes: `values[::2]`.",
        "",
        "## Data",
        "",
        "- Train: 25 debugging + 15 test-generation examples.",
        "- Independent holdout: 15 debugging + 15 test-generation examples.",
        "- Fixed benchmark prompts, targets, names, and fixtures are excluded.",
    ]
    for key, title in (
        ("fixed_benchmark", "Fixed benchmark"),
        ("independent_holdout", "Independent holdout"),
    ):
        lines.extend(["", f"## {title}", ""])
        for name, values in summary[key]["modes"].items():
            lines.append(
                f"- `{name}`: overall {_pct(values['overall_execution_pass_rate'])}; "
                f"target {_pct(values['target_cluster_pass_rate'])}; "
                f"non-target {_pct(values['non_target_pass_rate'])}; "
                f"debugging {_pct(values['debugging_pass_rate'])}; "
                f"test generation {_pct(values['test_generation_pass_rate'])}; "
                f"active {values['active_adapter_parameters']:.0f}; "
                f"stored {values['stored_adapter_parameters']:.0f}; "
                f"trainable {values['trainable_adapter_parameters']:.0f}."
            )
        lines.append(
            f"- Target wins/losses: "
            f"{summary[key]['target_cluster_wins_losses']['fail_to_pass']}/"
            f"{summary[key]['target_cluster_wins_losses']['pass_to_fail']}."
        )
        lines.append(
            f"- Non-target regressions: {summary[key]['non_target_regressions']}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['promotion_decision']['status']}`. The runner never promotes automatically.",
            "",
            "## Required answers",
            "",
            f"1. Pattern: {summary['answers']['failure_pattern']}",
            f"2. Data: {summary['answers']['data_created']}",
            f"3. Improved independent holdout: `{summary['answers']['improved_independent_holdout']}`.",
            f"4. Improved fixed benchmark target: `{summary['answers']['improved_fixed_benchmark_target']}`.",
            f"5. Caused non-target regression: `{summary['answers']['caused_non_target_regression']}`.",
            f"6. Decision: `{summary['answers']['promotion_status']}`.",
            f"7. Validates failure-born skill creation: `{summary['answers']['validates_second_plastic_cortex_mechanism']}`.",
        ]
    )
    if summary["status"] == "dry-run":
        lines.extend(["", "Run:", "", "```bash", _command(), "```"])
    return "\n".join(lines) + "\n"


def _command():
    return (
        "PYTHONPATH=. uv run python scripts/run_alternating_skill_experiment.py "
        "--seeds 11 22 33 44 55 --dataset data/eval.jsonl "
        "--baseline-experiment artifacts/experiments/five-seed "
        "--protected-experiment artifacts/experiments/python-skill-gating-validation "
        "--output artifacts/experiments/failure-born-skill/alternating_skill"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--train-data", default="data/failure_born/alternating_skill/train.jsonl")
    parser.add_argument("--holdout-data", default="data/failure_born/alternating_skill/holdout.jsonl")
    parser.add_argument("--baseline-experiment", default="artifacts/experiments/five-seed")
    parser.add_argument("--protected-experiment", default="artifacts/experiments/python-skill-gating-validation")
    parser.add_argument("--output", default="artifacts/experiments/failure-born-skill/alternating_skill")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    train_path, holdout_path = build_datasets(Path(args.train_data).parent)
    benchmark = Path(args.dataset)
    checksum = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    baseline_snapshot = _snapshot(args.baseline_experiment, args.seeds)
    parameters = {}
    for name in ("python_skill", "debugging_skill", "test_generation_skill"):
        path = (
            Path(args.baseline_experiment)
            / f"seed-{args.seeds[0]}"
            / "adapters"
            / name
            / "metadata.json"
        )
        parameters[name] = json.loads(path.read_text())["trainable_parameters"]
    parameters["alternating_skill"] = parameters["python_skill"]
    root = Path(args.output)
    metadata = {}
    for seed in args.seeds:
        examples, adapter_root, candidate, command = _prepare(
            seed, root, train_path
        )
        _link_existing(
            Path(args.baseline_experiment) / f"seed-{seed}" / "adapters",
            adapter_root,
        )
        if args.dry_run:
            metadata[seed] = {
                "command": command,
                "trainable_parameters": None,
                "quarantined": True,
            }
        else:
            metadata[seed] = _train_candidate(
                examples, candidate, command, seed, force=args.force
            )
        evaluate(
            benchmark,
            output=root / f"seed-{seed}" / "fixed",
            adapter_root=adapter_root,
            dry_run=args.dry_run,
            modes=MODES,
        )
        evaluate(
            holdout_path,
            output=root / f"seed-{seed}" / "holdout",
            adapter_root=adapter_root,
            dry_run=args.dry_run,
            modes=MODES,
        )
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("benchmark changed during experiment")
    if _snapshot(args.baseline_experiment, args.seeds) != baseline_snapshot:
        raise RuntimeError("existing adapters changed during experiment")
    summary = _build_summary(
        _read_rows(root, args.seeds, "fixed"),
        _read_rows(root, args.seeds, "holdout"),
        metadata,
        parameters,
        seeds=args.seeds,
        checksum=checksum,
        dry_run=args.dry_run,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(_markdown(summary))
    print(json.dumps({"summary": str(root / "summary.json"), "report": str(root / "summary.md")}, indent=2))
    print(f"\nTraining/evaluation command:\n{_command()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
