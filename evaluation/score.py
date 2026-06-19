from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def bootstrap_mean_ci(
    values: list[float], *, samples: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    randomizer = random.Random(seed)
    means = sorted(
        statistics.mean(randomizer.choice(values) for _ in values) for _ in range(samples)
    )
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    top1 = []
    top5 = []
    success = []
    iterations = []
    runtime = []
    for row in rows:
        gold = set(row.get("gold_files", []))
        predicted = list(row.get("predicted_files", []))
        top1.append(float(bool(predicted and predicted[0] in gold)))
        top5.append(float(bool(gold & set(predicted[:5]))))
        success.append(float(bool(row.get("patch_success"))))
        iterations.append(float(row.get("iterations", 0)))
        runtime.append(float(row.get("runtime_seconds", 0)))
    return {
        "tasks": float(len(rows)),
        "top1_localization": statistics.mean(top1) if top1 else 0.0,
        "top5_localization": statistics.mean(top5) if top5 else 0.0,
        "patch_success": statistics.mean(success) if success else 0.0,
        "median_iterations": statistics.median(iterations) if iterations else 0.0,
        "median_runtime_seconds": statistics.median(runtime) if runtime else 0.0,
    }


def score_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row["arm"])
        task_id = str(row["task_id"])
        by_arm[arm].append(row)
        by_task[task_id][arm] = row
    arms = {arm: _arm_summary(arm_rows) for arm, arm_rows in sorted(by_arm.items())}
    success_deltas: list[float] = []
    iteration_deltas: list[float] = []
    runtime_deltas: list[float] = []
    for pair in by_task.values():
        if "raw" not in pair or "evidence" not in pair:
            continue
        raw = pair["raw"]
        evidence = pair["evidence"]
        success_deltas.append(
            float(bool(evidence.get("patch_success"))) - float(bool(raw.get("patch_success")))
        )
        iteration_deltas.append(
            float(evidence.get("iterations", 0)) - float(raw.get("iterations", 0))
        )
        runtime_deltas.append(
            float(evidence.get("runtime_seconds", 0)) - float(raw.get("runtime_seconds", 0))
        )
    return {
        "arms": arms,
        "paired": {
            "tasks": len(success_deltas),
            "patch_success_delta": statistics.mean(success_deltas) if success_deltas else 0.0,
            "patch_success_delta_ci95": bootstrap_mean_ci(success_deltas),
            "median_iteration_delta": (
                statistics.median(iteration_deltas) if iteration_deltas else 0.0
            ),
            "median_runtime_delta_seconds": (
                statistics.median(runtime_deltas) if runtime_deltas else 0.0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    print(json.dumps(score_results(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

