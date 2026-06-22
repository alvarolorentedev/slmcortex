#!/usr/bin/env python3
"""Analyze router policies by recombining existing deterministic generations."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.schemas import MODES, SKILLS
from skill_lattice_coder.utils import write_json

DEFAULT_SEEDS = (11, 22, 33, 44, 55)
POLICIES = (
    "no_python_for_generation",
    "python_only_for_test_generation",
    "conservative_python_skill",
    "debugging_without_python",
    "oracle_without_python_generation",
)
BASELINES = ("base", "generic", "current_router", "oracle_lattice")
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


def requested_skills(policy, task_type, current, oracle):
    if task_type == "python_generation":
        return []
    if policy == "debugging_without_python":
        return (
            ["debugging_skill"]
            if task_type == "debugging"
            else ["python_skill", "test_generation_skill"]
        )
    if policy == "python_only_for_test_generation" and task_type == "test_generation":
        return ["python_skill", "test_generation_skill"]
    if policy == "oracle_without_python_generation":
        return oracle
    if policy in {"no_python_for_generation", "conservative_python_skill", "python_only_for_test_generation"}:
        return current
    raise ValueError(f"unknown policy: {policy}")


def _source_mode(policy, task_type):
    if task_type == "python_generation":
        return "base"
    if policy == "debugging_without_python":
        return "single-skill" if task_type == "debugging" else "oracle-lattice"
    if policy == "python_only_for_test_generation" and task_type == "test_generation":
        return "oracle-lattice"
    if policy == "oracle_without_python_generation":
        return "oracle-lattice"
    return "lattice"


def select_counterfactual_rows(rows, oracle_skills, policies=POLICIES):
    indexed = {}
    for row in rows:
        key = (row["seed"], row["example_id"], row["mode"])
        if key in indexed:
            raise ValueError(f"duplicate result row: {key}")
        indexed[key] = row
    selected = {}
    base_rows = [row for row in rows if row["mode"] == "base"]
    if not base_rows:
        raise ValueError("no base result rows")
    for policy in policies:
        policy_rows = []
        for base in base_rows:
            key = (base["seed"], base["example_id"])
            current = indexed.get((*key, "lattice"))
            oracle = indexed.get((*key, "oracle-lattice"))
            if current is None or oracle is None:
                raise ValueError(f"incomplete source artifacts for {key}")
            expected = requested_skills(
                policy,
                base["task_type"],
                current["selected_skills"],
                oracle_skills[base["example_id"]],
            )
            mode = _source_mode(policy, base["task_type"])
            source = indexed.get((*key, mode))
            if source is None:
                raise ValueError(f"missing source mode {mode} for {key}")
            actual = source.get("selected_skills")
            if actual != expected:
                raise ValueError(
                    f"adapter tuple mismatch for {policy}/{key}: "
                    f"expected {expected}, got {actual}"
                )
            policy_rows.append({**source, "policy": policy, "source_mode": mode})
        selected[policy] = policy_rows
    return selected


def _rate(rows):
    values = [
        bool(row["execution_passed"])
        for row in rows
        if row.get("execution_passed") is not None
    ]
    return mean(values) if values else None


def _task_rate(rows, task):
    return _rate([row for row in rows if row["task_type"] == task])


def _family_breakdown(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("benchmark_group") or row["example_id"]].append(row)
    return {
        family: {"count": len(items), "execution_pass_rate": _rate(items)}
        for family, items in sorted(grouped.items())
    }


def _transitions(rows, base_rows):
    base = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in base_rows
        if row["task_type"] == "python_generation"
    }
    candidate = {
        (row["seed"], row["example_id"]): bool(row["execution_passed"])
        for row in rows
        if row["task_type"] == "python_generation"
    }
    if candidate.keys() != base.keys():
        raise ValueError("incomplete Python-generation pairs")
    return {
        "python_pass_to_fail_vs_base": sum(base[key] and not candidate[key] for key in base),
        "python_fail_to_pass_vs_base": sum(not base[key] and candidate[key] for key in base),
    }


def _policy_summary(name, rows, baselines, adapter_parameters):
    overall = _rate(rows)
    skills = sorted({skill for row in rows for skill in row["selected_skills"]})
    stored = (
        adapter_parameters["generic"]
        if name == "generic"
        else sum(adapter_parameters[skill] for skill in skills)
    )
    active = mean(row.get("active_adapter_parameters", 0) for row in rows)
    selections = Counter(tuple(row["selected_skills"]) for row in rows)
    severe = {}
    base_python = [
        row for row in baselines["base"]
        if row["task_type"] == "python_generation"
    ]
    for family in SEVERE_FAMILIES:
        candidate = [
            row for row in rows
            if row["task_type"] == "python_generation"
            and row.get("benchmark_group") == family
        ]
        base = [row for row in base_python if row.get("benchmark_group") == family]
        severe[family] = {
            "base_pass_rate": _rate(base),
            "policy_pass_rate": _rate(candidate),
        }
    result = {
        "overall_execution_pass_rate": overall,
        "python_generation_pass_rate": _task_rate(rows, "python_generation"),
        "debugging_pass_rate": _task_rate(rows, "debugging"),
        "test_generation_pass_rate": _task_rate(rows, "test_generation"),
        "policy_vs_oracle_lattice_delta": overall - _rate(baselines["oracle_lattice"]),
        "policy_vs_current_lattice_delta": overall - _rate(baselines["current_router"]),
        "policy_vs_base_delta": overall - _rate(baselines["base"]),
        "policy_vs_generic_delta": overall - _rate(baselines["generic"]),
        "selected_skill_tuple_distribution": [
            {"selected_skills": list(skills), "count": count}
            for skills, count in sorted(selections.items())
        ],
        "source_modes_used": sorted({row.get("source_mode", row["mode"]) for row in rows}),
        "active_adapter_parameters": active,
        "stored_adapter_parameters": stored,
        "trainable_adapter_parameters": stored,
        "active_stored_parameter_ratio": active / stored if stored else None,
        "semantic_family_breakdown": _family_breakdown(rows),
        "severe_python_regression_family_breakdown": severe,
    }
    result.update(_transitions(rows, baselines["base"]))
    return result


def build_summary(rows, selected, *, adapter_parameters, seeds, benchmark_sha256):
    baselines = {
        "base": [row for row in rows if row["mode"] == "base"],
        "generic": [row for row in rows if row["mode"] == "generic"],
        "current_router": [row for row in rows if row["mode"] == "lattice"],
        "oracle_lattice": [row for row in rows if row["mode"] == "oracle-lattice"],
    }
    all_rows = {**baselines, **selected}
    policies = {
        name: _policy_summary(name, policy_rows, baselines, adapter_parameters)
        for name, policy_rows in all_rows.items()
    }
    best_overall = max(selected, key=lambda name: policies[name]["overall_execution_pass_rate"])
    best_python = max(selected, key=lambda name: policies[name]["python_generation_pass_rate"])
    current = policies["current_router"]
    test_policy = policies["python_only_for_test_generation"]
    debug_policy = policies["debugging_without_python"]
    return {
        "is_counterfactual_recombination": True,
        "requires_training": False,
        "requires_new_inference": False,
        "source_modes_used": sorted({mode for value in policies.values() for mode in value["source_modes_used"]}),
        "seeds": seeds,
        "benchmark_sha256": benchmark_sha256,
        "parameter_accounting_definitions": {
            "active_adapter_parameters": "Mean adapter parameters active for each generation.",
            "stored_adapter_parameters": "Adapter parameters required on disk for the mode or policy.",
            "trainable_adapter_parameters": "Parameters trained to produce the available adapters.",
        },
        "policies": policies,
        "answers": {
            "best_overall_policy": best_overall,
            "best_python_recovery_policy": best_python,
            "suppressing_python_for_generation_helps": policies["no_python_for_generation"]["python_generation_pass_rate"] > current["python_generation_pass_rate"],
            "python_for_test_generation_recovers_oracle_gap": test_policy["test_generation_pass_rate"] == policies["oracle_lattice"]["test_generation_pass_rate"],
            "debugging_needs_python_skill": debug_policy["debugging_pass_rate"] < current["debugging_pass_rate"],
            "tradeoff": (
                f"{best_overall} changes overall execution by "
                f"{policies[best_overall]['policy_vs_current_lattice_delta']:+.1%} "
                "while recovering Python generation to "
                f"{policies[best_overall]['python_generation_pass_rate']:.1%}."
            ),
            "replacement_recommendation": (
                f"Validate {best_overall} with a real router-driven inference/evaluation run "
                "before replacing the current router."
            ),
            "next_target": "Routing validation; do not move to composition or failure-born skills yet.",
        },
    }


def _load_artifacts(root, seeds, examples):
    expected = {example.id: example for example in examples}
    rows, parameters = [], None
    for seed in seeds:
        seed_root = Path(root) / f"seed-{seed}"
        results = seed_root / "results.jsonl"
        if not results.exists():
            raise FileNotFoundError(f"missing results: {results}")
        seed_rows = []
        for number, line in enumerate(results.read_text().splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSON at {results}:{number}") from error
            for field in ("example_id", "task_type", "mode", "selected_skills", "execution_passed"):
                if field not in row:
                    raise ValueError(f"missing {field} at {results}:{number}")
            example = expected.get(row["example_id"])
            if example is None or row["task_type"] != example.task_type or row["mode"] not in MODES:
                raise ValueError(f"unexpected result row at {results}:{number}")
            row["seed"] = seed
            seed_rows.append(row)
        counts = Counter((row["example_id"], row["mode"]) for row in seed_rows)
        required = {(example.id, mode) for example in examples for mode in MODES}
        if set(counts) != required or any(count != 1 for count in counts.values()):
            raise ValueError(f"incomplete or duplicate results: {results}")
        rows.extend(seed_rows)
        seed_parameters = {}
        for name in ("generic", *SKILLS):
            path = seed_root / "adapters" / name / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(f"missing adapter metadata: {path}")
            metadata = json.loads(path.read_text())
            value = metadata.get("trainable_parameters")
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid trainable_parameters: {path}")
            seed_parameters[name] = value
        if parameters is not None and seed_parameters != parameters:
            raise ValueError("inconsistent adapter parameters across seeds")
        parameters = seed_parameters
    return rows, parameters


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _markdown(summary):
    lines = [
        "# Python-Skill Gating: Counterfactual Recombination",
        "",
        "- `is_counterfactual_recombination`: `true`",
        "- `requires_training`: `false`",
        "- `requires_new_inference`: `false`",
        f"- `source_modes_used`: `{', '.join(summary['source_modes_used'])}`",
        "",
        "This report deterministically recombines existing five-seed outputs. "
        "Any winning policy must next be validated by a real router-driven inference/evaluation run.",
        "",
        "## Results",
        "",
        "| Policy | Overall | Python | Debugging | Tests | vs oracle | vs current | vs base | vs generic | Active | Stored | Trainable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["policies"].items():
        lines.append(
            f"| {name} | {_pct(value['overall_execution_pass_rate'])} | "
            f"{_pct(value['python_generation_pass_rate'])} | {_pct(value['debugging_pass_rate'])} | "
            f"{_pct(value['test_generation_pass_rate'])} | {_pct(value['policy_vs_oracle_lattice_delta'])} | "
            f"{_pct(value['policy_vs_current_lattice_delta'])} | {_pct(value['policy_vs_base_delta'])} | "
            f"{_pct(value['policy_vs_generic_delta'])} | {value['active_adapter_parameters']:.0f} | "
            f"{value['stored_adapter_parameters']:.0f} | {value['trainable_adapter_parameters']:.0f} |"
        )
    answers = summary["answers"]
    lines.extend(
        [
            "",
            "## Research questions",
            "",
            f"1. Best overall pass rate: `{answers['best_overall_policy']}`.",
            f"2. Best Python recovery: `{answers['best_python_recovery_policy']}`.",
            f"3. Suppressing `python_skill` for pure generation helps: **{answers['suppressing_python_for_generation_helps']}**.",
            f"4. Adding `python_skill` for test generation recovers the oracle test gap: **{answers['python_for_test_generation_recovers_oracle_gap']}**.",
            f"5. Debugging needs `python_skill`: **{answers['debugging_needs_python_skill']}**.",
            f"6. Tradeoff: {answers['tradeoff']}",
            f"7. Router replacement: {answers['replacement_recommendation']}",
            f"8. Next target: {answers['next_target']}",
            "",
            "## Parameter accounting",
            "",
            "- `active_adapter_parameters`: mean adapter parameters active for each generation.",
            "- `stored_adapter_parameters`: adapter parameters required on disk for the mode or policy.",
            "- `trainable_adapter_parameters`: parameters trained to produce the available adapters.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--baseline-experiment", default="artifacts/experiments/five-seed")
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--output", default="artifacts/experiments/python-skill-gating")
    args = parser.parse_args(argv)
    benchmark = Path(args.dataset)
    before = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    examples = load_jsonl(benchmark)
    rows, parameters = _load_artifacts(args.baseline_experiment, args.seeds, examples)
    oracle_skills = {example.id: example.skills for example in examples}
    selected = select_counterfactual_rows(rows, oracle_skills)
    summary = build_summary(
        rows,
        selected,
        adapter_parameters=parameters,
        seeds=args.seeds,
        benchmark_sha256=before,
    )
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != before:
        raise RuntimeError("benchmark changed during analysis")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_markdown(summary))
    print(json.dumps({"summary": str(output / "summary.json"), "report": str(output / "summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
