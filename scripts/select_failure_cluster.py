#!/usr/bin/env python3
"""Rank protected-router failure clusters without training or inference."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from skill_lattice_coder.data import load_jsonl
from skill_lattice_coder.utils import write_json

PROTECTED_MODE = "python_only_for_test_generation"
SOURCE_MODES = {
    "base": "base",
    "current_router": "lattice",
    "oracle_lattice": "oracle-lattice",
    "protected_skill_router": PROTECTED_MODE,
}


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
            if row.get("mode") in SOURCE_MODES.values():
                row["seed"] = seed
                seed_rows.append(row)
        required = {
            (example.id, mode)
            for example in examples
            for mode in SOURCE_MODES.values()
        }
        counts = Counter((row["example_id"], row["mode"]) for row in seed_rows)
        if set(counts) != required or any(value != 1 for value in counts.values()):
            raise ValueError(f"incomplete or duplicate validation rows: {path}")
        rows.extend(seed_rows)
    return rows


def build_cluster_selection(rows, *, seeds, benchmark_sha256):
    indexed = {
        (row["seed"], row["example_id"], row["mode"]): row for row in rows
    }
    protected_rows = [
        row for row in rows if row["mode"] in {PROTECTED_MODE, "protected_skill_router"}
    ]
    if not protected_rows:
        raise ValueError("no protected-router rows")

    grouped = defaultdict(list)
    for row in protected_rows:
        grouped[(row["benchmark_group"], row["task_type"])].append(row)

    family_non_python_failures = Counter()
    for (family, task), items in grouped.items():
        if task != "python_generation":
            family_non_python_failures[family] += sum(
                not bool(row.get("execution_passed")) for row in items
            )

    clusters = []
    for (family, task), items in grouped.items():
        items = sorted(items, key=lambda row: row["seed"])
        failures = [row for row in items if not bool(row.get("execution_passed"))]
        mode_passes = {name: 0 for name in SOURCE_MODES}
        selected = Counter(tuple(row.get("selected_skills", [])) for row in items)
        for row in items:
            for name, mode in SOURCE_MODES.items():
                source = (
                    row
                    if name == "protected_skill_router"
                    else indexed.get((row["seed"], row["example_id"], mode))
                )
                if source is None:
                    raise ValueError(
                        f"missing {mode} row for seed {row['seed']} "
                        f"example {row['example_id']}"
                    )
                mode_passes[name] += bool(source.get("execution_passed"))
        failure_by_seed = {
            str(seed): sum(row["seed"] == seed for row in failures) for seed in seeds
        }
        existing_router_rescues = (
            mode_passes["current_router"] + mode_passes["oracle_lattice"]
        )
        eligible = task != "python_generation" and bool(failures)
        distinct_examples = len({row["example_id"] for row in items})
        likely_skill_specific = (
            eligible
            and len(failures) == len(items)
            and mode_passes["base"] == 0
            and existing_router_rescues == 0
        )
        clusters.append(
            {
                "semantic_family": family,
                "task_type": task,
                "total_examples": len(items),
                "pass_count": len(items) - len(failures),
                "fail_count": len(failures),
                "pass_rate": (len(items) - len(failures)) / len(items),
                "failure_count_by_seed": failure_by_seed,
                "selected_skill_tuple": [
                    {
                        "selected_skills": list(skills),
                        "count": count,
                    }
                    for skills, count in sorted(selected.items())
                ],
                "base_pass_count": mode_passes["base"],
                "base_passed": mode_passes["base"] > 0,
                "current_router_pass_count": mode_passes["current_router"],
                "current_router_passed": mode_passes["current_router"] > 0,
                "oracle_lattice_pass_count": mode_passes["oracle_lattice"],
                "oracle_lattice_passed": mode_passes["oracle_lattice"] > 0,
                "protected_router_failed": bool(failures),
                "distinct_benchmark_examples": distinct_examples,
                "family_non_python_fail_count": family_non_python_failures[family],
                "failure_seed_count": sum(value > 0 for value in failure_by_seed.values()),
                "existing_router_rescue_count": existing_router_rescues,
                "localized_semantic_pattern": True,
                "enough_repeated_failures_for_candidate_design": (
                    len(failures) >= 3
                    and sum(value > 0 for value in failure_by_seed.values()) >= 3
                ),
                "enough_independent_examples_for_promotion": distinct_examples >= 3,
                "likely_skill_specific": likely_skill_specific,
                "eligible_for_failure_born_skill": eligible,
                "exclusion_reason": (
                    "Python generation uses validated base fallback."
                    if task == "python_generation"
                    else None
                ),
            }
        )

    clusters.sort(
        key=lambda item: (
            not item["eligible_for_failure_born_skill"],
            -item["fail_count"],
            -item["family_non_python_fail_count"],
            -item["failure_seed_count"],
            item["base_pass_count"],
            item["existing_router_rescue_count"],
            item["semantic_family"],
            item["task_type"],
        )
    )
    primary = next(
        (cluster for cluster in clusters if cluster["eligible_for_failure_born_skill"]),
        None,
    )
    if primary is None:
        raise ValueError("no eligible failure cluster")
    recommendation = {
        "semantic_family": primary["semantic_family"],
        "task_type": primary["task_type"],
        "candidate_skill_name": f"{primary['semantic_family']}_skill",
    }
    return {
        "step": 1,
        "analysis_only": True,
        "requires_training": False,
        "requires_new_inference": False,
        "benchmark_sha256": benchmark_sha256,
        "seeds": seeds,
        "source": {
            "router": "protected_skill_router",
            "concrete_mode": PROTECTED_MODE,
        },
        "evaluation_leakage_warning": (
            "Seed repetitions are repeated generations of the same benchmark "
            "item, not independent examples. The selected benchmark item must "
            "remain evaluation-only; any candidate training data must be newly "
            "created training-only variants."
        ),
        "ranking_criteria": [
            "exclude Python-generation base-fallback failures",
            "higher protected-router failure count",
            "higher same-family non-Python failure count",
            "failure recurrence across more seeds",
            "fewer base-model rescues",
            "fewer current/oracle router rescues",
            "stable semantic-family and task-type tie-break",
        ],
        "recommended_primary_cluster": recommendation,
        "recommendation_reason": (
            f"`{primary['semantic_family']}` / `{primary['task_type']}` failed "
            f"{primary['fail_count']}/{primary['total_examples']} protected-router "
            f"runs across {primary['failure_seed_count']} seeds; its family has "
            f"{primary['family_non_python_fail_count']} non-Python failures and "
            f"{primary['base_pass_count']} base rescues."
        ),
        "clusters": clusters,
    }


def _pct(value):
    return f"{value:.1%}"


def _markdown(summary):
    primary = summary["recommended_primary_cluster"]
    lines = [
        "# Failure-Born Skill: Cluster Selection",
        "",
        "- Step: **1 — analysis only**",
        "- Training performed: **no**",
        "- New inference performed: **no**",
        f"- Source router: `{summary['source']['router']}`",
        "",
        "## Recommendation",
        "",
        f"Primary cluster: **`{primary['semantic_family']}` / "
        f"`{primary['task_type']}`**.",
        "",
        f"Provisional candidate name for the next decision: "
        f"`{primary['candidate_skill_name']}`.",
        "",
        summary["recommendation_reason"],
        "",
        "This is a selection recommendation only. No candidate skill, dataset, "
        "router, or training artifact has been created.",
        "",
        f"**Evaluation leakage warning:** {summary['evaluation_leakage_warning']}",
        "",
        "## Ranked clusters",
        "",
        "| Rank | Family | Task | Runs | Unique | Pass | Fail | Rate | Failure seeds | Skills | Base | Current | Oracle | Skill-specific | Eligible |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for rank, cluster in enumerate(summary["clusters"], 1):
        skills = "; ".join(
            "+".join(item["selected_skills"]) or "base"
            for item in cluster["selected_skill_tuple"]
        )
        failure_seeds = ", ".join(
            seed
            for seed, count in cluster["failure_count_by_seed"].items()
            if count
        ) or "none"
        lines.append(
            f"| {rank} | {cluster['semantic_family']} | {cluster['task_type']} | "
            f"{cluster['total_examples']} | {cluster['distinct_benchmark_examples']} | "
            f"{cluster['pass_count']} | "
            f"{cluster['fail_count']} | {_pct(cluster['pass_rate'])} | "
            f"{failure_seeds} | {skills} | {cluster['base_pass_count']} | "
            f"{cluster['current_router_pass_count']} | "
            f"{cluster['oracle_lattice_pass_count']} | "
            f"{cluster['likely_skill_specific']} | "
            f"{cluster['eligible_for_failure_born_skill']} |"
        )
    lines.extend(
        [
            "",
            "## Ranking method",
            "",
            *[f"{index}. {criterion}." for index, criterion in enumerate(summary["ranking_criteria"], 1)],
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 22, 33, 44, 55))
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument(
        "--validation-experiment",
        default="artifacts/experiments/python-skill-gating-validation",
    )
    parser.add_argument(
        "--output", default="artifacts/experiments/failure-born-skill"
    )
    args = parser.parse_args(argv)

    benchmark = Path(args.dataset)
    checksum = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    examples = load_jsonl(benchmark)
    rows = _load_rows(args.validation_experiment, args.seeds, examples)
    summary = build_cluster_selection(
        rows, seeds=args.seeds, benchmark_sha256=checksum
    )
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != checksum:
        raise RuntimeError("benchmark changed during cluster analysis")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cluster_selection.json", summary)
    (output / "cluster_selection.md").write_text(_markdown(summary))
    print(
        json.dumps(
            {
                "json": str(output / "cluster_selection.json"),
                "markdown": str(output / "cluster_selection.md"),
                "recommended_primary_cluster": summary[
                    "recommended_primary_cluster"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
