from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from repo_brain.indexer import index_repository
from repo_brain.skills.evidence import build_evidence


def _checkout(repo: Path, revision: str, destination: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(repo), str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision],
        check=True,
    )


def _run_arm(
    task: dict[str, Any],
    arm: str,
    command_template: str,
    workspace: Path,
    timeout: int,
) -> dict[str, Any]:
    prompt = str(task["task"])
    preprocessing = 0.0
    if arm == "evidence":
        started = time.monotonic()
        result = index_repository(workspace)
        if result.errors:
            raise RuntimeError("; ".join(result.errors))
        prompt += "\n\n" + build_evidence(workspace, prompt).render()
        preprocessing = time.monotonic() - started
    prompt_path = workspace / ".repo-brain-evaluation-prompt.md"
    patch_path = workspace / ".repo-brain-evaluation.patch"
    metrics_path = workspace / ".repo-brain-evaluation-metrics.json"
    prompt_path.write_text(prompt)
    command = command_template.format(
        repo=workspace,
        prompt=prompt_path,
        patch=patch_path,
        metrics=metrics_path,
    )
    started = time.monotonic()
    process = subprocess.run(
        shlex.split(command),
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    runtime = time.monotonic() - started
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    return {
        "task_id": task["id"],
        "arm": arm,
        "gold_files": task.get("gold_files", []),
        "gold_tests": task.get("gold_tests", []),
        "predicted_files": metrics.get("predicted_files", []),
        "selected_tests": metrics.get("selected_tests", []),
        "patch_success": bool(metrics.get("patch_success", process.returncode == 0)),
        "iterations": int(metrics.get("iterations", 1)),
        "input_tokens": metrics.get("input_tokens"),
        "output_tokens": metrics.get("output_tokens"),
        "runtime_seconds": runtime,
        "preprocessing_seconds": preprocessing,
        "returncode": process.returncode,
        "stderr": process.stderr[-4_000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--agent-command", required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.jsonl"))
    parser.add_argument("--timeout", type=int, default=1_800)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for arm in ("raw", "evidence"):
            with tempfile.TemporaryDirectory(prefix=f"repo-brain-{arm}-") as directory:
                workspace = Path(directory) / "repo"
                _checkout(Path(task["repo"]).resolve(), str(task["revision"]), workspace)
                rows.append(
                    _run_arm(task, arm, args.agent_command, workspace, args.timeout)
                )
                args.output.write_text(
                    "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

