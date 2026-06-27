import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .agent_sandbox import (
    ARTIFACT_DIR_PREFIXES,
    CODE_FILE_SUFFIXES,
    TEXT_FILE_SUFFIXES,
    WRITE_MODES,
    ToolSandbox,
)
from .runtime import SkillRuntime


def run_agent(
    *,
    runtime_path: Path,
    repo: Path,
    task: str | list[str] | None = None,
    writes: str = "confirm",
    test_command: str | None = None,
    trace_out: Path | None = None,
    dry_run: bool = False,
    task_provider: Callable[[], str | None] | None = None,
) -> dict[str, Any]:
    if writes not in WRITE_MODES:
        raise ValueError(f"unknown writes mode: {writes}")
    repo_root = repo.resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")

    runtime = SkillRuntime.load(runtime_path)
    runtime.validate()
    sandbox = ToolSandbox(repo_root, writes)
    tasks: list[str] = []
    trace = {
        "schema_version": "1",
        "run_id": f"agent-{int(time.time() * 1000)}",
        "task": None,
        "tasks": tasks,
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "writes_mode": writes,
        "execution_mode": "dry-run-route-plan-only" if dry_run else f"write-mode-{writes}",
        "test_command": test_command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }
    task_results = []
    for index, task_input in enumerate(_task_sequence(task, task_provider=task_provider), start=1):
        if trace["task"] is None:
            trace["task"] = task_input
        tasks.append(task_input)
        task_result = _run_single_task(
            runtime=runtime,
            sandbox=sandbox,
            repo_root=repo_root,
            task=task_input,
            writes=writes,
            test_command=test_command,
            dry_run=dry_run,
            run_id=trace["run_id"],
            task_index=index,
            prior_task_results=task_results,
        )
        task_results.append(task_result)
        trace["steps"].extend(task_result["steps"])

    if not task_results:
        raise ValueError("at least one task is required")

    final_status = _merge_task_statuses(task_results, dry_run=dry_run)
    final_summary = _multi_task_summary(task_results, writes=writes, dry_run=dry_run)
    final_materialization = task_results[-1]["final_materialization"]
    validations = [result["validation"] for result in task_results]
    trace["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    trace["status"] = final_status
    trace["final_summary"] = final_summary
    trace["review_artifact_path"] = final_materialization.get("review_artifact_path")
    trace["generated_patch"] = final_materialization["diff"]
    trace["generated_actions"] = final_materialization.get("actions")
    trace["validation"] = validations[-1]
    trace["validation_results"] = validations
    trace["task_results"] = [_task_trace_payload(result) for result in task_results]
    if trace_out is not None:
        trace_out = trace_out.resolve()
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        trace_out.write_text(json.dumps(trace, indent=2) + "\n")

    result: dict[str, Any] = {
        "status": final_status,
        "task": tasks[0],
        "tasks": tasks,
        "writes_mode": writes,
        "execution_mode": trace["execution_mode"],
        "repo": str(repo_root),
        "runtime": str(runtime_path.resolve()),
        "trace_path": str(trace_out.resolve()) if trace_out is not None else None,
        "step_count": len(trace["steps"]),
        "final_summary": final_summary,
        "steps": trace["steps"],
        "validation": validations[-1],
        "validation_results": validations,
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_actions": final_materialization.get("actions"),
        "last_proposed_diff": final_materialization["diff"],
        "task_results": [_task_result_payload(result) for result in task_results],
    }
    if len(tasks) == 1:
        result["generated_actions"] = final_materialization.get("actions")
        result["last_proposed_diff"] = final_materialization["diff"]
    return result


def _task_sequence(
    task: str | list[str] | None,
    *,
    task_provider: Callable[[], str | None] | None = None,
) -> list[str]:
    queued = _normalize_task_inputs(task, allow_empty=True)
    yielded = False
    for item in queued:
        yielded = True
        yield item
    if task_provider is not None:
        while True:
            next_task = task_provider()
            if next_task is None:
                break
            normalized = next_task.strip()
            if not normalized:
                continue
            yielded = True
            yield normalized
    if not yielded:
        raise ValueError("at least one task is required")


def _run_single_task(
    *,
    runtime: SkillRuntime,
    sandbox: ToolSandbox,
    repo_root: Path,
    task: str,
    writes: str,
    test_command: str | None,
    dry_run: bool,
    run_id: str,
    task_index: int,
    prior_task_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task_steps: list[dict[str, Any]] = []
    execution_history = _execution_history(prior_task_results or [])

    repo_files = sandbox.list_files()
    patch_target = _choose_patch_target(repo_files, task)
    read_targets = _default_read_targets(repo_files, patch_target=patch_target, prior_task_results=prior_task_results or [])
    focus_targets = {patch_target, *_recent_changed_files(prior_task_results or [])}
    repo_context = {
        path: sandbox.read_file(path, max_chars=12000 if path in focus_targets else 4000)
        for path in read_targets
    }
    _append_step(
        {"steps": task_steps},
        {
            "step_type": "inspect_repo",
            "selected_skills": [],
            "route_type": None,
            "route_reason": None,
            "tool_name": "list_files+read_file",
            "files_read": read_targets,
            "files_changed": [],
            "status": "complete",
            "result_summary": f"Inspected {len(repo_files)} files and read {len(read_targets)} files.",
            "task": task,
            "task_index": task_index,
        },
    )

    plan_result = runtime.infer(
        messages=_step_messages(
            task,
            (
                "Create a short execution plan for a local coding task. "
                "Return concise numbered steps only."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
            execution_history=execution_history,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    plan_step = _inference_step(
        "plan",
        plan_result,
        files_read=read_targets,
        mode_label="route/plan only" if dry_run else None,
    )
    plan_step["task"] = task
    plan_step["task_index"] = task_index
    _append_step({"steps": task_steps}, plan_step)

    patch_result = runtime.infer(
        messages=_step_messages(
            task,
            (
                "Produce JSON-only coding actions for the next step. "
                "Return either a JSON array of action objects or an object with an 'actions' array. "
                "Each action must use kind=file_replace, proposed_diff, or no_change. "
                "Prefer proposed_diff for edits to existing files and preserve unchanged code. "
                "Use file_replace for new files or intentional full-file rewrites only, and never emit a partial file body for an existing file. "
                "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that. "
                f"Prefer editing {patch_target} only when no better path is clear, and create a new source file when the repo has no suitable code target."
            ),
            repo_files=repo_files,
            repo_context=repo_context,
            execution_history=execution_history,
        ),
        task_type="python_generation",
        dry_run=dry_run,
    )
    review_path = _default_review_path(repo_root, f"{run_id}-task-{task_index}") if writes == "confirm" and not dry_run else None
    patch_materialization = _materialize_inference_action(
        sandbox,
        patch_result,
        patch_target,
        dry_run,
        review_path=review_path,
    )
    patch_step = _inference_step(
        "propose_patch",
        patch_result,
        files_read=read_targets,
        files_changed=patch_materialization["files_changed"],
        tool_name="propose_diff" if patch_materialization["write_status"] != "applied" else "apply_patch",
        tool_result_summary=patch_materialization["summary"],
        proposed_diff=patch_materialization["diff"],
        write_status=patch_materialization["write_status"],
        review_artifact_path=patch_materialization.get("review_artifact_path"),
        mode_label="route/plan only" if dry_run else None,
    )
    patch_step["task"] = task
    patch_step["task_index"] = task_index
    _append_step({"steps": task_steps}, patch_step)

    validation = _validation_result_for_mode(test_command, repo_root, dry_run)
    _append_step(
        {"steps": task_steps},
        {
            "step_type": "run_validation",
            "selected_skills": [],
            "route_type": None,
            "route_reason": None,
            "tool_name": "run_validation",
            "tool_args": validation["command"],
            "files_read": [],
            "files_changed": [],
            "validation_exit_code": validation["exit_code"],
            "status": validation["status"],
            "result_summary": (validation["stderr"] or validation["stdout"] or validation["status"])[:400],
            "task": task,
            "task_index": task_index,
        },
    )

    debug_materialization = None
    if validation["status"] == "failed":
        debug_result = runtime.infer(
            messages=_step_messages(
                task,
                (
                    "Debug the failed validation and propose the next code change as JSON-only actions. "
                    "Return either a JSON array of action objects or an object with an 'actions' array. "
                    "Each action must use kind=file_replace, proposed_diff, or no_change. "
                    "Prefer proposed_diff for edits to existing files and preserve unchanged code. "
                    "Use file_replace for new files or intentional full-file rewrites only, and never emit a partial file body for an existing file. "
                    "Do not rewrite runtime bundles, datasets, skill packages, or other generated artifacts unless the task explicitly asks for that."
                ),
                repo_files=repo_files,
                repo_context=repo_context,
                execution_history=execution_history,
                validation_output=_validation_summary(validation),
            ),
            task_type="debugging",
            dry_run=dry_run,
        )
        debug_review_path = _default_review_path(repo_root, f"{run_id}-task-{task_index}-debug") if writes == "confirm" and not dry_run else None
        debug_materialization = _materialize_inference_action(
            sandbox,
            debug_result,
            patch_target,
            dry_run,
            review_path=debug_review_path,
        )
        debug_step = _inference_step(
            "debug_failure",
            debug_result,
            files_read=read_targets,
            files_changed=debug_materialization["files_changed"],
            tool_name="propose_diff" if debug_materialization["write_status"] != "applied" else "apply_patch",
            tool_result_summary=debug_materialization["summary"],
            proposed_diff=debug_materialization["diff"],
            write_status=debug_materialization["write_status"],
            review_artifact_path=debug_materialization.get("review_artifact_path"),
        )
        debug_step["task"] = task
        debug_step["task_index"] = task_index
        _append_step({"steps": task_steps}, debug_step)

    return {
        "task": task,
        "task_index": task_index,
        "steps": task_steps,
        "validation": validation,
        "final_materialization": debug_materialization or patch_materialization,
    }


def _normalize_task_inputs(task: str | list[str] | None, *, allow_empty: bool = False) -> list[str]:
    if task is None:
        tasks = []
    elif isinstance(task, str):
        tasks = [task]
    else:
        tasks = task
    normalized = [item.strip() for item in tasks if isinstance(item, str) and item.strip()]
    if not normalized and not allow_empty:
        raise ValueError("at least one task is required")
    return normalized


def _merge_task_statuses(task_results: list[dict[str, Any]], *, dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    statuses = [
        _final_status(result["steps"], result["validation"], dry_run)
        for result in task_results
    ]
    if "validation_failed" in statuses:
        return "validation_failed"
    if "review_required" in statuses:
        return "review_required"
    if "applied" in statuses:
        return "applied"
    return statuses[-1]


def _multi_task_summary(task_results: list[dict[str, Any]], *, writes: str, dry_run: bool) -> str:
    if len(task_results) == 1:
        return _final_summary(task_results[0]["steps"], task_results[0]["validation"], writes, dry_run=dry_run)
    selected: list[tuple[Any, ...]] = []
    for result in task_results:
        for step in result["steps"]:
            skills = tuple(step.get("selected_skills") or [])
            if skills not in selected:
                selected.append(skills)
    if dry_run:
        return (
            f"Executed {len(task_results)} tasks and {sum(len(result['steps']) for result in task_results)} steps in route/plan only dry-run mode. "
            f"Observed {len(selected)} distinct skill selections. "
            "Generation, writes, and validation were skipped."
        )
    validations = [result["validation"]["status"] for result in task_results]
    return (
        f"Executed {len(task_results)} tasks and {sum(len(result['steps']) for result in task_results)} steps with writes mode '{writes}'. "
        f"Observed {len(selected)} distinct skill selections. "
        f"Validation statuses: {', '.join(validations)}."
    )


def _task_trace_payload(task_result: dict[str, Any]) -> dict[str, Any]:
    final_materialization = task_result["final_materialization"]
    return {
        "task": task_result["task"],
        "task_index": task_result["task_index"],
        "status": _final_status(task_result["steps"], task_result["validation"], False),
        "validation": task_result["validation"],
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_patch": final_materialization["diff"],
        "generated_actions": final_materialization.get("actions"),
        "steps": task_result["steps"],
    }


def _task_result_payload(task_result: dict[str, Any]) -> dict[str, Any]:
    final_materialization = task_result["final_materialization"]
    return {
        "task": task_result["task"],
        "task_index": task_result["task_index"],
        "status": _final_status(task_result["steps"], task_result["validation"], False),
        "validation": task_result["validation"],
        "review_artifact_path": final_materialization.get("review_artifact_path"),
        "generated_actions": final_materialization.get("actions"),
        "last_proposed_diff": final_materialization["diff"],
        "steps": task_result["steps"],
    }


def _choose_patch_target(files: list[str], task: str) -> str:
    for path in _preferred_source_files(files):
        if path.endswith(".py") and not path.startswith("tests/"):
            return path
    preferred = _preferred_source_files(files)
    if preferred:
        return preferred[0]
    return _default_new_source_path(task)


def _default_read_targets(
    files: list[str],
    *,
    patch_target: str | None = None,
    prior_task_results: list[dict[str, Any]] | None = None,
) -> list[str]:
    preferred = _preferred_source_files(files)
    targets: list[str] = []
    for candidate in [patch_target, *_recent_changed_files(prior_task_results or [])]:
        if candidate and candidate in preferred and candidate not in targets:
            targets.append(candidate)
    for candidate in preferred:
        if candidate not in targets:
            targets.append(candidate)
        if len(targets) >= 5:
            break
    return targets


def _recent_changed_files(task_results: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    changed: list[str] = []
    for result in reversed(task_results):
        for path in result["final_materialization"].get("files_changed") or []:
            if path not in changed:
                changed.append(path)
            if len(changed) >= limit:
                return changed
    return changed


def _preferred_source_files(files: list[str]) -> list[str]:
    source_files = [
        path for path in files
        if path.endswith(TEXT_FILE_SUFFIXES)
        and not _is_artifact_path(path)
    ]
    if source_files:
        code_files = [path for path in source_files if path.endswith(CODE_FILE_SUFFIXES)]
        return code_files + [path for path in source_files if path not in code_files]
    return []


def _is_artifact_path(path: str) -> bool:
    return path.startswith(ARTIFACT_DIR_PREFIXES)


def _default_new_source_path(task: str) -> str:
    lowered = task.lower()
    if any(token in lowered for token in ("fastapi", "endpoint", "router", "api")):
        return "app.py"
    return "main.py"


def _extract_actions(generation: str, default_path: str) -> list[dict[str, Any]]:
    text = _strip_code_fence((generation or "").strip())
    if not text:
        return [{"kind": "no_change", "summary": "Empty model output."}]
    try:
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("actions"), list):
            return _normalize_actions(value["actions"], default_path)
        if isinstance(value, list):
            return _normalize_actions(value, default_path)
        if isinstance(value, dict):
            return _normalize_actions([value], default_path)
    except json.JSONDecodeError:
        pass
    if _looks_like_unified_diff(text):
        return [{
            "kind": "proposed_diff",
            "diff": text,
            "summary": f"Unstructured diff output for {default_path}.",
        }]
    content = _extract_code_content(text)
    return [{
        "kind": "file_replace",
        "path": default_path,
        "content": content,
        "summary": f"Unstructured model output converted to file update for {default_path}.",
    }]


def _normalize_actions(actions: list[Any], default_path: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = dict(item)
        if action.get("kind") == "file_replace" and "path" not in action:
            action["path"] = default_path
        normalized.append(action)
    return normalized or [{"kind": "no_change", "summary": "Empty action list."}]


def _final_summary(steps: list[dict[str, Any]], validation: dict[str, Any], writes: str, *, dry_run: bool) -> str:
    selected = [tuple(step.get("selected_skills") or []) for step in steps if step.get("selected_skills") is not None]
    unique = [list(item) for index, item in enumerate(selected) if item not in selected[:index]]
    if dry_run:
        return (
            f"Executed {len(steps)} steps in route/plan only dry-run mode. "
            f"Observed {len(unique)} distinct skill selections. "
            "Generation, writes, and validation were skipped."
        )
    return (
        f"Executed {len(steps)} steps with writes mode '{writes}'. "
        f"Observed {len(unique)} distinct skill selections. "
        f"Validation status: {validation['status']}."
    )


def _inference_step(
    step_type: str,
    result: dict[str, Any],
    *,
    files_read: list[str],
    files_changed: list[str] | None = None,
    tool_name: str | None = None,
    tool_result_summary: str | None = None,
    proposed_diff: str | None = None,
    write_status: str | None = None,
    review_artifact_path: str | None = None,
    mode_label: str | None = None,
) -> dict[str, Any]:
    return {
        "step_type": step_type,
        "selected_skills": result.get("selected_skills") or [],
        "route_type": result.get("route_type"),
        "route_reason": result.get("reason"),
        "tool_name": tool_name,
        "files_read": files_read,
        "files_changed": files_changed or [],
        "status": result.get("status") or "complete",
        "generation": result.get("generation"),
        "result_summary": tool_result_summary or result.get("generation") or result.get("reason"),
        "proposed_diff": proposed_diff,
        "write_status": write_status,
        "review_artifact_path": review_artifact_path,
        "mode_label": mode_label,
    }


def _materialize_inference_action(
    sandbox: ToolSandbox,
    result: dict[str, Any],
    default_path: str,
    dry_run: bool,
    *,
    review_path: Path | None = None,
) -> dict[str, Any]:
    if dry_run:
        return {
            "kind": "dry-run",
            "write_status": "dry-run",
            "files_changed": [],
            "diff": "",
            "review_artifact_path": None,
            "summary": "Dry-run route/plan only: skipped artifact materialization.",
            "actions": [],
        }
    actions = _extract_actions(result.get("generation") or "", default_path)
    return sandbox.materialize_actions(actions, review_path=review_path)


def _append_step(trace: dict[str, Any], step: dict[str, Any]) -> None:
    step["step_index"] = len(trace["steps"]) + 1
    trace["steps"].append(step)


def _default_review_path(repo: Path, run_id: str) -> Path:
    return repo / ".skillcortex" / "reviews" / f"{run_id}.patch"


def _looks_like_unified_diff(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    return lines[0].startswith("--- ") and lines[1].startswith("+++ ") and any(
        line.startswith("@@") for line in lines[2:]
    )


def _looks_like_truncating_prefix_rewrite(before: str, after: str) -> bool:
    stripped_before = before.rstrip()
    stripped_after = after.rstrip()
    if not stripped_after or stripped_after == stripped_before:
        return False
    return stripped_before.startswith(stripped_after)


def _extract_code_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip() + "\n"
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _validation_result_for_mode(command: str | None, repo: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "skipped",
            "command": command,
            "exit_code": None,
            "stdout": "",
            "stderr": "dry-run route/plan only: validation skipped",
        }
    if not command:
        return {
            "status": "skipped",
            "command": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
        }
    return _run_validation_command(command, repo)


def _final_status(steps: list[dict[str, Any]], validation: dict[str, Any], dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    if validation["status"] == "failed":
        return "validation_failed"
    write_statuses = {step.get("write_status") for step in steps if step.get("write_status")}
    if "applied" in write_statuses:
        return "applied"
    if "review_required" in write_statuses:
        return "review_required"
    return "complete"


def _run_validation_command(command: str, repo: Path) -> dict[str, Any]:
    completed = subprocess.run(
        shlex.split(command),
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _step_messages(
    task: str,
    instruction: str,
    *,
    repo_files: list[str],
    repo_context: dict[str, str],
    execution_history: str | None = None,
    validation_output: str | None = None,
) -> list[dict[str, str]]:
    content = [
        f"Task: {task}",
        instruction,
        "Repository files:",
        "\n".join(repo_files[:20]),
        "Repository excerpts:",
        "\n\n".join(f"[{path}]\n{text}" for path, text in repo_context.items()),
    ]
    if execution_history:
        content.extend(["Previous execution context:", execution_history])
    if validation_output:
        content.extend(["Validation output:", validation_output])
    return [{"role": "user", "content": "\n\n".join(content)}]


def _execution_history(task_results: list[dict[str, Any]], *, max_items: int = 5) -> str | None:
    if not task_results:
        return None
    lines: list[str] = []
    for result in task_results[-max_items:]:
        final_materialization = result["final_materialization"]
        changed_files = final_materialization.get("files_changed") or []
        changed_files_label = ", ".join(changed_files) if changed_files else "none"
        summary = final_materialization.get("summary") or result["steps"][-1].get("result_summary") or ""
        lines.extend(
            [
                f"Task {result['task_index']}: {result['task']}",
                f"Status: {_final_status(result['steps'], result['validation'], False)}",
                f"Changed files: {changed_files_label}",
                f"Validation: {result['validation'].get('status')}",
                f"Summary: {summary}",
            ]
        )
        diff_text = (final_materialization.get("diff") or "").strip()
        if diff_text:
            lines.append("Last diff excerpt:")
            lines.append(_truncate_text(diff_text, 800))
    return "\n".join(lines)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _validation_summary(validation: dict[str, Any]) -> str:
    return (validation.get("stderr") or validation.get("stdout") or validation.get("status") or "")[:4000]