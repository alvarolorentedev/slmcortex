#!/usr/bin/env python3
"""Compare composition policies without training adapters."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.schemas import SKILLS
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
FRESH_POLICIES = (
    "weighted_task_composition",
    "reverse_weighted_task_composition",
)
COUNTERFACTUAL_POLICIES = (
    "current_composition",
    "single_strongest_skill",
    "protected_pair_composition",
    "no_harmful_pairs",
)
HARMFUL_PAIR = {"debugging_skill", "test_generation_skill"}


def _read_results(root, seeds):
    rows = []
    for seed in seeds:
        path = Path(root) / f"seed-{seed}" / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing results: {path}")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["seed"] = seed
            rows.append(row)
    return rows


def _load_existing_rows(five_seed, protected, seeds):
    rows = _read_results(five_seed, seeds)
    for row in _read_results(protected, seeds):
        if row["mode"] == "python_only_for_test_generation":
            rows.append({**row, "mode": "protected_skill_router"})
    return rows


def select_counterfactual_rows(rows):
    indexed = {
        (row["seed"], row["example_id"], row["mode"]): row for row in rows
    }
    base_rows = [row for row in rows if row["mode"] == "base"]
    selected = defaultdict(list)
    for base in base_rows:
        key = (base["seed"], base["example_id"])
        current = indexed.get((*key, "lattice"))
        single = indexed.get((*key, "single-skill"))
        protected = indexed.get((*key, "protected_skill_router"))
        if current is None or single is None:
            raise ValueError(f"incomplete source rows for {key}")
        chosen = {
            "current_composition": current,
            "single_strongest_skill": (
                base if base["task_type"] == "python_generation" else single
            ),
            "no_harmful_pairs": (
                single
                if set(current["selected_skills"]) == HARMFUL_PAIR
                else current
            ),
        }
        if protected is not None:
            chosen["protected_pair_composition"] = protected
        for policy, row in chosen.items():
            selected[policy].append(
                {**row, "policy": policy, "source_mode": row["mode"]}
            )
    return dict(selected)


def _load_weighted_rows(root, seeds):
    rows = _read_results(root, seeds)
    return {
        policy: [row for row in rows if row["mode"] == policy]
        for policy in FRESH_POLICIES
    }


def _rate(rows):
    values = [
        bool(row["execution_passed"])
        for row in rows
        if row.get("execution_passed") is not None
    ]
    return mean(values) if values else None


def _delta(left, right):
    return None if left is None or right is None else left - right


def _task_rate(rows, task):
    return _rate([row for row in rows if row["task_type"] == task])


def _parameters(root, seeds):
    expected = None
    for seed in seeds:
        values = {}
        for skill in SKILLS:
            path = Path(root) / f"seed-{seed}" / "adapters" / skill / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"missing adapter metadata: {path}")
            values[skill] = json.loads(path.read_text())["trainable_parameters"]
        if expected is not None and values != expected:
            raise ValueError("inconsistent adapter parameters across seeds")
        expected = values
    return expected


def _snapshot(root, seeds):
    output = {}
    for seed in seeds:
        adapter_root = Path(root) / f"seed-{seed}" / "adapters"
        for path in sorted(adapter_root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                output[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return output


def _transitions(rows, protected):
    baseline = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in protected
        if row.get("execution_passed") is not None
    }
    candidate = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in rows
        if row.get("execution_passed") is not None
    }
    output = {}
    for task in ("python_generation", "debugging", "test_generation"):
        keys = [
            (row["seed"], row["example_id"])
            for row in protected
            if row["task_type"] == task and row.get("execution_passed") is not None
        ]
        output[task] = {
            "fail_to_pass": sum(
                not baseline[key] and candidate.get(key) is True for key in keys
            ),
            "pass_to_fail": sum(
                baseline[key] and candidate.get(key) is False for key in keys
            ),
        }
    return output


def _breakdown(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        value = (
            tuple(row.get("selected_skills", []))
            if key == "selected_skills"
            else row.get(key) or row["example_id"]
        )
        grouped[value].append(row)
    return [
        {
            key: list(value) if isinstance(value, tuple) else value,
            "count": len(items),
            "execution_pass_rate": _rate(items),
        }
        for value, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _tuple_analysis(rows, protected):
    baseline = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in protected
        if row.get("execution_passed") is not None
    }
    grouped = defaultdict(list)
    for row in rows:
        if row.get("execution_passed") is not None:
            grouped[
                (row["task_type"], tuple(row.get("selected_skills", [])))
            ].append(row)
    output = []
    for (task, skills), items in sorted(grouped.items()):
        pairs = [
            (
                baseline[(row["seed"], row["example_id"])],
                bool(row["execution_passed"]),
            )
            for row in items
        ]
        wins = sum(not left and right for left, right in pairs)
        losses = sum(left and not right for left, right in pairs)
        output.append(
            {
                "task_type": task,
                "selected_skills": list(skills),
                "count": len(items),
                "execution_pass_rate": _rate(items),
                "wins_vs_protected": wins,
                "losses_vs_protected": losses,
                "classification": (
                    "helpful" if wins > losses else "harmful" if losses > wins else "neutral"
                ),
            }
        )
    return output


def _policy_summary(name, rows, protected, current, oracle, parameters, fresh):
    skills = {skill for row in rows for skill in row.get("selected_skills", [])}
    stored = sum(parameters[skill] for skill in skills)
    active = mean(row.get("active_adapter_parameters", 0) for row in rows)
    overall = _rate(rows)
    return {
        "fresh_inference": fresh,
        "counterfactual_recombination": not fresh,
        "overall_execution_pass_rate": overall,
        "python_generation_pass_rate": _task_rate(rows, "python_generation"),
        "debugging_pass_rate": _task_rate(rows, "debugging"),
        "test_generation_pass_rate": _task_rate(rows, "test_generation"),
        "delta_vs_protected_skill_router": _delta(overall, _rate(protected)),
        "delta_vs_current_router": _delta(overall, _rate(current)),
        "delta_vs_oracle_lattice": _delta(overall, _rate(oracle)),
        "active_adapter_parameters": active,
        "stored_adapter_parameters": stored,
        "trainable_adapter_parameters": stored,
        "active_stored_parameter_ratio": active / stored if stored else None,
        "selected_skill_tuple_distribution": _breakdown(rows, "selected_skills"),
        "pass_fail_transitions_by_task": _transitions(rows, protected),
        "semantic_family_breakdown": _breakdown(rows, "benchmark_group"),
        "tuple_level_analysis": _tuple_analysis(rows, protected),
    }


def _build_summary(
    counterfactual,
    fresh,
    parameters,
    *,
    seeds,
    benchmark_sha256,
    dry_run,
):
    policies = {**counterfactual, **fresh}
    protected = policies["protected_pair_composition"]
    current = policies["current_composition"]
    oracle = [
        row
        for row in _CURRENT_EXISTING_ROWS
        if row["mode"] == "oracle-lattice"
    ]
    summaries = {
        name: _policy_summary(
            name,
            rows,
            protected,
            current,
            oracle,
            parameters,
            name in FRESH_POLICIES,
        )
        for name, rows in policies.items()
    }
    available = {
        name: value
        for name, value in summaries.items()
        if value["overall_execution_pass_rate"] is not None
    }
    best = max(
        available,
        key=lambda name: available[name]["overall_execution_pass_rate"],
    )
    protected_score = summaries["protected_pair_composition"][
        "overall_execution_pass_rate"
    ]
    composition_wins = (
        not dry_run
        and best != "protected_pair_composition"
        and summaries[best]["overall_execution_pass_rate"] >= protected_score
        and summaries[best]["python_generation_pass_rate"] >= 0.54
        and summaries[best]["debugging_pass_rate"] >= 0.444
        and summaries[best]["test_generation_pass_rate"] >= 0.564
    )
    pair_classes = defaultdict(set)
    for value in summaries.values():
        for item in value["tuple_level_analysis"]:
            if len(item["selected_skills"]) == 2:
                pair_classes[tuple(item["selected_skills"])].add(
                    item["classification"]
                )
    return {
        "status": "dry-run" if dry_run else "complete",
        "contains_fresh_inference": True,
        "contains_counterfactual_recombination": True,
        "requires_training": False,
        "training_invoked": False,
        "seeds": seeds,
        "benchmark_sha256": benchmark_sha256,
        "policies": summaries,
        "answers": {
            "composition_control_improves_protected_router": composition_wins,
            "best_policy": best,
            "equal_composition_is_sufficient": not composition_wins,
            "debugging_needs_python_companion": (
                summaries["protected_pair_composition"]["debugging_pass_rate"]
                > summaries["single_strongest_skill"]["debugging_pass_rate"]
            ),
            "test_generation_needs_python_companion": (
                summaries["protected_pair_composition"][
                    "test_generation_pass_rate"
                ]
                > summaries["single_strongest_skill"][
                    "test_generation_pass_rate"
                ]
            ),
            "consistently_harmful_pairs": sorted(
                pair for pair, classes in pair_classes.items()
                if classes == {"harmful"}
            ),
            "second_validated_mechanism": composition_wins,
            "next_target": (
                "failure-born skills"
                if not dry_run and not composition_wins
                else "composition validation"
                if dry_run
                else "composition"
            ),
        },
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _markdown(summary):
    lines = [
        "# Composition Control",
        "",
        f"- `status`: `{summary['status']}`",
        "- `requires_training`: `false`",
        "- `training_invoked`: `false`",
        "",
        "| Policy | Source | Overall | Python | Debugging | Tests | vs protected | vs current | vs oracle | Active | Stored | Trainable | Active/stored |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["policies"].items():
        source = "fresh inference" if value["fresh_inference"] else "counterfactual recombination"
        lines.append(
            f"| {name} | {source} | {_pct(value['overall_execution_pass_rate'])} | "
            f"{_pct(value['python_generation_pass_rate'])} | "
            f"{_pct(value['debugging_pass_rate'])} | "
            f"{_pct(value['test_generation_pass_rate'])} | "
            f"{_pct(value['delta_vs_protected_skill_router'])} | "
            f"{_pct(value['delta_vs_current_router'])} | "
            f"{_pct(value['delta_vs_oracle_lattice'])} | "
            f"{value['active_adapter_parameters']:.0f} | "
            f"{value['stored_adapter_parameters']:.0f} | "
            f"{value['trainable_adapter_parameters']:.0f} | "
            f"{_pct(value['active_stored_parameter_ratio'])} |"
        )
    if summary["status"] == "complete":
        answers = summary["answers"]
        lines.extend(
            [
                "",
                "## Research questions",
                "",
                f"1. Composition improves the protected router: **{answers['composition_control_improves_protected_router']}**.",
                f"2. Equal composition is sufficient: **{answers['equal_composition_is_sufficient']}**.",
                f"3. Debugging needs `python_skill`: **{answers['debugging_needs_python_companion']}**.",
                f"4. Test generation needs `python_skill`: **{answers['test_generation_needs_python_companion']}**.",
                f"5. Consistently harmful pairs: `{answers['consistently_harmful_pairs']}`.",
                f"6. Composition is the second validated mechanism: **{answers['second_validated_mechanism']}**.",
                f"7. Next target: `{answers['next_target']}`.",
            ]
        )
    else:
        lines.extend(["", "Fresh weighted inference has not been run.", "", "```bash", _command(), "```"])
    return "\n".join(lines) + "\n"


def _command():
    return (
        "PYTHONPATH=. uv run python scripts/run_composition_control.py "
        "--seeds 11 22 33 44 55 --dataset data/eval.jsonl "
        "--five-seed-experiment artifacts/experiments/five-seed "
        "--protected-experiment artifacts/experiments/python-skill-gating-validation "
        "--output artifacts/experiments/composition-control"
    )


_CURRENT_EXISTING_ROWS = []


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--five-seed-experiment", default="artifacts/experiments/five-seed")
    parser.add_argument("--protected-experiment", default="artifacts/experiments/python-skill-gating-validation")
    parser.add_argument("--output", default="artifacts/experiments/composition-control")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    benchmark = Path(args.dataset)
    checksum = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    examples = load_jsonl(benchmark)
    parameters = _parameters(args.five_seed_experiment, args.seeds)
    before = _snapshot(args.five_seed_experiment, args.seeds)
    existing = _load_existing_rows(
        args.five_seed_experiment, args.protected_experiment, args.seeds
    )
    global _CURRENT_EXISTING_ROWS
    _CURRENT_EXISTING_ROWS = existing
    counterfactual = select_counterfactual_rows(existing)

    output = Path(args.output)
    for seed in args.seeds:
        evaluate(
            benchmark,
            output=output / f"seed-{seed}",
            adapter_root=Path(args.five_seed_experiment) / f"seed-{seed}" / "adapters",
            dry_run=args.dry_run,
            modes=FRESH_POLICIES,
        )
    fresh = _load_weighted_rows(output, args.seeds)
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("benchmark changed during composition analysis")
    if _snapshot(args.five_seed_experiment, args.seeds) != before:
        raise RuntimeError("adapter artifacts changed; training or mutation detected")

    summary = _build_summary(
        counterfactual,
        fresh,
        parameters,
        seeds=args.seeds,
        benchmark_sha256=checksum,
        dry_run=args.dry_run,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_markdown(summary))
    print(json.dumps({"summary": str(output / "summary.json"), "report": str(output / "summary.md")}, indent=2))
    print(f"\nFresh weighted-composition command:\n{_command()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
