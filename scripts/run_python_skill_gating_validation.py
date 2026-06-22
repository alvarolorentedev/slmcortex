#!/usr/bin/env python3
"""Run fresh inference for the winning Python-skill gating policy."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.evaluation import evaluate
from skill_lattice_coder.config import base_config
from skill_lattice_coder.router import PythonOnlyForTestGenerationRouter
from skill_lattice_coder.schemas import SKILLS
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
VALIDATION_MODES = (
    "base",
    "legacy_rule_router",
    "oracle-lattice",
    "python_only_for_test_generation",
)
SEVERE_FAMILIES = (
    "divide_or",
    "keys_for_value",
    "mask_address",
    "multiply_values",
    "trim_prefix",
    "without_none",
    "capped_total",
    "decimal_digit_sum",
    "substring_total",
)


def _rate(rows):
    values = [
        bool(row["execution_passed"])
        for row in rows
        if row.get("execution_passed") is not None
    ]
    return mean(values) if values else None


def _task_rate(rows, task):
    return _rate([row for row in rows if row["task_type"] == task])


def _delta(left, right):
    return None if left is None or right is None else left - right


def _load_rows(root, seeds, examples):
    expected = {example.id: example for example in examples}
    rows = []
    for seed in seeds:
        path = Path(root) / f"seed-{seed}" / "results.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing validation results: {path}")
        seed_rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSON at {path}:{number}") from error
            example = expected.get(row.get("example_id"))
            if example is None or row.get("task_type") != example.task_type:
                raise ValueError(f"unexpected result row at {path}:{number}")
            row["seed"] = seed
            seed_rows.append(row)
        counts = Counter((row["example_id"], row["mode"]) for row in seed_rows)
        required = {(example.id, mode) for example in examples for mode in VALIDATION_MODES}
        if set(counts) != required or any(count != 1 for count in counts.values()):
            raise ValueError(f"incomplete or duplicate validation results: {path}")
        rows.extend(seed_rows)
    return rows


def _load_generic_rows(root, seeds, examples):
    ids = {example.id for example in examples}
    rows = []
    for seed in seeds:
        path = Path(root) / f"seed-{seed}" / "results.jsonl"
        if not path.exists():
            return []
        found = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("mode") == "generic" and row.get("example_id") in ids:
                row["seed"] = seed
                found[row["example_id"]] = row
        if set(found) != ids:
            raise ValueError(f"incomplete generic baseline: {path}")
        rows.extend(found.values())
    return rows


def _adapter_parameters(root, seeds):
    expected = None
    for seed in seeds:
        values = {}
        for name in ("generic", *SKILLS):
            path = Path(root) / f"seed-{seed}" / "adapters" / name / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"missing adapter metadata: {path}")
            value = json.loads(path.read_text()).get("trainable_parameters")
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid trainable_parameters: {path}")
            values[name] = value
        if expected is not None and values != expected:
            raise ValueError("inconsistent adapter parameters across seeds")
        expected = values
    return expected


def _adapter_snapshot(root, seeds):
    snapshot = {}
    for seed in seeds:
        adapter_root = Path(root) / f"seed-{seed}" / "adapters"
        if not adapter_root.exists():
            raise FileNotFoundError(f"missing adapters: {adapter_root}")
        for path in sorted(adapter_root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _validate_routes(rows, examples):
    oracle = {example.id: example.skills for example in examples}
    policy_router = PythonOnlyForTestGenerationRouter()
    for row in rows:
        if row["mode"] == "base":
            expected = []
        elif row["mode"] == "oracle-lattice":
            expected = oracle[row["example_id"]]
        elif row["mode"] == "python_only_for_test_generation":
            expected = policy_router.route(row["task_type"]).selected_skills
        else:
            continue
        if row.get("selected_skills") != expected:
            raise ValueError(
                f"adapter tuple mismatch for {row['mode']}/{row['seed']}/"
                f"{row['example_id']}: expected {expected}, "
                f"got {row.get('selected_skills')}"
            )


def _families(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("benchmark_group") or row["example_id"]].append(row)
    return {
        name: {"count": len(items), "execution_pass_rate": _rate(items)}
        for name, items in sorted(grouped.items())
    }


def _transitions(candidate, base):
    base_values = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in base
        if row["task_type"] == "python_generation"
    }
    candidate_values = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in candidate
        if row["task_type"] == "python_generation"
    }
    if candidate_values.keys() != base_values.keys():
        raise ValueError("incomplete Python-generation pairs")
    return {
        "python_pass_to_fail_vs_base": sum(
            base_values[key] and not candidate_values[key] for key in base_values
        ),
        "python_fail_to_pass_vs_base": sum(
            not base_values[key] and candidate_values[key] for key in base_values
        ),
    }


def _mode_summary(name, rows, baselines, parameters):
    stored = (
        0
        if name == "base"
        else parameters["generic"]
        if name == "generic"
        else sum(parameters[skill] for skill in SKILLS)
    )
    active = mean(row.get("active_adapter_parameters", 0) for row in rows)
    selections = Counter(tuple(row.get("selected_skills", [])) for row in rows)
    severe = {}
    for family in SEVERE_FAMILIES:
        severe[family] = {
            "base_pass_rate": _rate(
                [
                    row for row in baselines["base"]
                    if row["task_type"] == "python_generation"
                    and row.get("benchmark_group") == family
                ]
            ),
            "mode_pass_rate": _rate(
                [
                    row for row in rows
                    if row["task_type"] == "python_generation"
                    and row.get("benchmark_group") == family
                ]
            ),
        }
    overall = _rate(rows)
    result = {
        "overall_execution_pass_rate": overall,
        "python_generation_pass_rate": _task_rate(rows, "python_generation"),
        "debugging_pass_rate": _task_rate(rows, "debugging"),
        "test_generation_pass_rate": _task_rate(rows, "test_generation"),
        "delta_vs_current_router": _delta(
            overall, _rate(baselines["current_router"])
        ),
        "delta_vs_oracle_lattice": _delta(
            overall, _rate(baselines["oracle_lattice"])
        ),
        "delta_vs_generic_lora": _delta(
            overall, _rate(baselines["generic"])
        ),
        "active_adapter_parameters": active,
        "stored_adapter_parameters": stored,
        "trainable_adapter_parameters": stored,
        "active_stored_parameter_ratio": active / stored if stored else None,
        "selected_skill_tuple_distribution": [
            {"selected_skills": list(skills), "count": count}
            for skills, count in sorted(selections.items())
        ],
        "severe_python_regression_family_breakdown": severe,
        "semantic_family_breakdown": _families(rows),
    }
    result.update(_transitions(rows, baselines["base"]))
    return result


def _build_summary(
    rows,
    generic_rows,
    parameters,
    *,
    seeds,
    benchmark_sha256,
    dry_run,
):
    baselines = {
        "base": [row for row in rows if row["mode"] == "base"],
        "current_router": [
            row for row in rows if row["mode"] == "legacy_rule_router"
        ],
        "oracle_lattice": [row for row in rows if row["mode"] == "oracle-lattice"],
        "generic": generic_rows,
    }
    modes = {
        **baselines,
        "python_only_for_test_generation": [
            row for row in rows
            if row["mode"] == "python_only_for_test_generation"
        ],
    }
    summaries = {
        name: _mode_summary(name, mode_rows, baselines, parameters)
        for name, mode_rows in modes.items()
        if mode_rows
    }
    candidate = summaries["python_only_for_test_generation"]
    current = summaries["current_router"]
    oracle = summaries["oracle_lattice"]
    confirmed = (
        not dry_run
        and candidate["overall_execution_pass_rate"]
        > current["overall_execution_pass_rate"]
        and candidate["python_generation_pass_rate"] >= 0.50
        and candidate["overall_execution_pass_rate"] >= 0.50
        and candidate["debugging_pass_rate"]
        >= current["debugging_pass_rate"] - 0.02
        and candidate["test_generation_pass_rate"]
        > current["test_generation_pass_rate"]
    )
    return {
        "status": "dry-run" if dry_run else "complete",
        "fresh_inference": True,
        "counterfactual_recombination": False,
        "requires_training": False,
        "training_invoked": False,
        "seeds": seeds,
        "benchmark_sha256": benchmark_sha256,
        "generation_settings": {"temperature": 0.0},
        "parameter_accounting_definitions": {
            "active_adapter_parameters": "Mean adapter parameters active for each generation.",
            "stored_adapter_parameters": "Adapter parameters required on disk for this mode.",
            "trainable_adapter_parameters": "Parameters trained to produce the available adapters.",
        },
        "modes": summaries,
        "answers": {
            "real_inference_confirmed_counterfactual_result": confirmed,
            "python_generation_recovered_to_base_like": (
                not dry_run
                and candidate["python_generation_pass_rate"]
                >= summaries["base"]["python_generation_pass_rate"] - 0.02
            ),
            "debugging_remained_near_current_router": (
                not dry_run
                and candidate["debugging_pass_rate"]
                >= current["debugging_pass_rate"] - 0.02
            ),
            "test_generation_approached_oracle_lattice": (
                not dry_run
                and candidate["test_generation_pass_rate"]
                >= oracle["test_generation_pass_rate"] - 0.02
            ),
            "replace_current_router": confirmed,
            "next_research_target": (
                "composition"
                if confirmed
                else "routing"
                if not dry_run
                else "pending fresh inference"
            ),
        },
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _markdown(summary):
    lines = [
        "# Python-Skill Gating Validation",
        "",
        f"- `status`: `{summary['status']}`",
        "- `fresh_inference`: `true`",
        "- `counterfactual_recombination`: `false`",
        "- `requires_training`: `false`",
        "- `training_invoked`: `false`",
        "",
        "## Results",
        "",
        "| Mode | Overall | Python | Debugging | Tests | vs current | vs oracle | vs generic | Active | Stored | Trainable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary.get("modes", {}).items():
        lines.append(
            f"| {name} | {_pct(value['overall_execution_pass_rate'])} | "
            f"{_pct(value['python_generation_pass_rate'])} | "
            f"{_pct(value['debugging_pass_rate'])} | "
            f"{_pct(value['test_generation_pass_rate'])} | "
            f"{_pct(value['delta_vs_current_router'])} | "
            f"{_pct(value['delta_vs_oracle_lattice'])} | "
            f"{_pct(value['delta_vs_generic_lora'])} | "
            f"{value['active_adapter_parameters']:.0f} | "
            f"{value['stored_adapter_parameters']:.0f} | "
            f"{value['trainable_adapter_parameters']:.0f} |"
        )
    if summary["status"] == "dry-run":
        lines.extend(
            [
                "",
                "Fresh inference has not been run. Execute the command below.",
                "",
                "```bash",
                _command(),
                "```",
            ]
        )
    else:
        answers = summary["answers"]
        lines.extend(
            [
                "",
                "## Research questions",
                "",
                f"1. Real inference confirmed the counterfactual result: **{answers['real_inference_confirmed_counterfactual_result']}**.",
                f"2. Python generation recovered to base-like performance: **{answers['python_generation_recovered_to_base_like']}**.",
                f"3. Debugging remained near current-router performance: **{answers['debugging_remained_near_current_router']}**.",
                f"4. Test generation approached oracle-lattice performance: **{answers['test_generation_approached_oracle_lattice']}**.",
                f"5. Replace the current router: **{answers['replace_current_router']}**.",
                f"6. Next research target: `{answers['next_research_target']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Parameter accounting",
            "",
            "- `active_adapter_parameters`: mean adapter parameters active for each generation.",
            "- `stored_adapter_parameters`: adapter parameters required on disk for this mode.",
            "- `trainable_adapter_parameters`: parameters trained to produce the available adapters.",
        ]
    )
    return "\n".join(lines) + "\n"


def _command():
    return (
        "PYTHONPATH=. uv run python scripts/run_python_skill_gating_validation.py "
        "--seeds 11 22 33 44 55 "
        "--dataset data/eval.jsonl "
        "--baseline-experiment artifacts/experiments/five-seed "
        "--output artifacts/experiments/python-skill-gating-validation"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--baseline-experiment", default="artifacts/experiments/five-seed")
    parser.add_argument("--output", default="artifacts/experiments/python-skill-gating-validation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    benchmark = Path(args.dataset)
    checksum = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    if base_config()["temperature"] != 0.0:
        raise ValueError("validation requires temperature 0.0")
    examples = load_jsonl(benchmark)
    parameters = _adapter_parameters(args.baseline_experiment, args.seeds)
    before_adapters = _adapter_snapshot(args.baseline_experiment, args.seeds)
    root = Path(args.output)
    for seed in args.seeds:
        evaluate(
            benchmark,
            output=root / f"seed-{seed}",
            adapter_root=Path(args.baseline_experiment) / f"seed-{seed}" / "adapters",
            dry_run=args.dry_run,
            modes=VALIDATION_MODES,
        )
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("benchmark changed during validation")
    if _adapter_snapshot(args.baseline_experiment, args.seeds) != before_adapters:
        raise RuntimeError("adapter artifacts changed; training or mutation detected")

    rows = _load_rows(root, args.seeds, examples)
    _validate_routes(rows, examples)
    generic_rows = _load_generic_rows(args.baseline_experiment, args.seeds, examples)
    summary = _build_summary(
        rows,
        generic_rows,
        parameters,
        seeds=args.seeds,
        benchmark_sha256=checksum,
        dry_run=args.dry_run,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(_markdown(summary))
    print(json.dumps({"summary": str(root / "summary.json"), "report": str(root / "summary.md")}, indent=2))
    print(
        "\nValidation plan:\n"
        f"- modes: {', '.join(VALIDATION_MODES)}\n"
        f"- seeds: {', '.join(map(str, DEFAULT_SEEDS))}\n"
        "- temperature: 0.0\n"
        "- training: disabled"
    )
    print(f"\nFresh five-seed validation command:\n{_command()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
