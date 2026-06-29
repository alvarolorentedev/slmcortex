from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import run_agent
from .catalog import MAX_REPO_FILES, MAX_TOTAL_BYTES, compose_from_folder, infer_task_hints, scan_repo_context
from .runtime import SlmRuntime, serve_runtime
from .shared.io import load_json_if_exists
from .shared.product import ensure_app_workspace, environment_diagnostics, runtime_name_for_folder


STATE_FILE = "composer-app-state.json"
ONBOARDING_MESSAGE = (
    "Start by selecting a local folder. Slm Cortex scans the codebase, recommends the right "
    "packages, composes a runtime, and then lets you run locally or export the bundle. "
    "Training, packaging, and registry workflows remain available as advanced capabilities, "
    "but they are not required for the default path."
)


def run_composer_app(
    *,
    folder: Path,
    workspace_root: Path | None = None,
    slms_dir: Path | None = None,
    task: str | None = None,
    runtime_name: str | None = None,
    outcome: str = "local_run",
    run_target: str = "compatibility_server",
    prompt: str | None = None,
    export_descriptor: Path | None = None,
    export_logs: bool = False,
    allow_base: bool = False,
    overwrite: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    writes: str = "confirm",
    test_command: str | None = None,
    trace_out: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    workspace = ensure_app_workspace(workspace_root)
    repo = folder.resolve()
    diagnostics = environment_diagnostics(workspace_root=workspace.root, product_mode="composer")
    capabilities = _capability_summary(diagnostics)
    state_path = workspace.state_dir / STATE_FILE
    state = _load_state(state_path)
    first_run = not bool(state.get("onboarding_completed"))
    runtime_slug = runtime_name_for_folder(repo, runtime_name)
    scan_summary = scan_repo_context(repo)
    scan_warnings = _scan_warnings(scan_summary)
    task_hints = infer_task_hints(scan_summary)
    resolved_task = (task or task_hints[0]["suggested_task"]).strip()
    reopened = str(repo) in (state.get("projects") or {})
    resolved_overwrite = overwrite or reopened
    descriptor_target = export_descriptor
    if descriptor_target is None and outcome == "export_bundle":
        descriptor_target = workspace.exports_dir / f"{runtime_slug}.json"

    compose_result = compose_from_folder(
        folder=repo,
        task=resolved_task,
        workspace_root=workspace.root,
        slms_dir=slms_dir,
        runtime_name=runtime_slug,
        export_descriptor=descriptor_target,
        allow_base=allow_base,
        overwrite=resolved_overwrite,
        product_mode="composer",
    )

    state = _record_project_state(
        state,
        repo=repo,
        runtime_name=runtime_slug,
        task=resolved_task,
        scan_summary=scan_summary,
        compose_result=compose_result,
        first_run=first_run,
    )
    _write_state(state_path, state)

    support_bundle = None
    compose_log_path = workspace.logs_dir / f"compose-{runtime_slug}.json"
    if compose_result["status"] == "complete":
        outcome_result = _resolve_outcome(
            compose_result=compose_result,
            outcome=outcome,
            run_target=run_target,
            prompt=prompt,
            host=host,
            port=port,
            writes=writes,
            test_command=test_command,
            trace_out=trace_out,
            dry_run=dry_run,
            dry_run_only=capabilities["dry_run_only"],
        )
        warnings = [*scan_warnings, *compose_result.get("warnings", [])]
        if outcome == "local_run" and capabilities["dry_run_only"]:
            warnings.append(
                "local run is currently limited to dry-run only because no supported runtime backend was detected"
            )
        errors = list(compose_result.get("errors", []))
        status = "complete"
        exit_code = 0
        product_error = None
    else:
        outcome_result = {
            "requested": outcome,
            "status": "blocked",
            "run_target": run_target if outcome == "local_run" else None,
        }
        warnings = [*scan_warnings, *compose_result.get("warnings", [])]
        errors = list(compose_result.get("errors", []))
        product_error = _translate_product_error(
            errors[0] if errors else "composer app workflow failed",
            capabilities=capabilities,
            outcome=outcome,
        )
        status = "failed"
        exit_code = 2
        if export_logs or status == "failed":
            support_bundle = _write_support_bundle(
                workspace=workspace,
                runtime_name=runtime_slug,
                compose_result=compose_result,
                state=state,
                diagnostics=diagnostics,
                scan_summary=scan_summary,
                product_error=product_error,
            )

    if export_logs and support_bundle is None:
        support_bundle = _write_support_bundle(
            workspace=workspace,
            runtime_name=runtime_slug,
            compose_result=compose_result,
            state=state,
            diagnostics=diagnostics,
            scan_summary=scan_summary,
            product_error=product_error,
        )

    return {
        "schema_version": "1",
        "status": status,
        "exit_code": exit_code,
        "operation": "composer_app",
        "product_mode": "composer",
        "onboarding": {
            "first_run": first_run,
            "completed": bool(state.get("onboarding_completed")),
            "message": ONBOARDING_MESSAGE,
            "capabilities": capabilities,
            "state_path": str(state_path.resolve()),
        },
        "project": {
            "folder": str(repo),
            "reopened": reopened,
            "task": resolved_task,
            "task_hints": task_hints,
            "scan_summary": scan_summary,
            "scan_warnings": scan_warnings,
        },
        "workspace": workspace.as_dict(),
        "composition": {
            "runtime_name": runtime_slug,
            "runtime": compose_result.get("runtime"),
            "routing_decision": compose_result.get("routing_decision"),
            "selected_slms": compose_result.get("selected_slms", []),
            "export_bundle": compose_result.get("export_bundle"),
        },
        "outcome": outcome_result,
        "product_error": product_error,
        "support": {
            "logs_dir": str(workspace.logs_dir.resolve()),
            "compose_log_path": str(compose_log_path.resolve()),
            "support_bundle": support_bundle,
        },
        "diagnostics": diagnostics,
        "warnings": warnings,
        "errors": errors,
    }


def _load_state(path: Path) -> dict[str, Any]:
    state = load_json_if_exists(path)
    projects = state.get("projects")
    if not isinstance(projects, dict):
        state["projects"] = {}
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _record_project_state(
    state: dict[str, Any],
    *,
    repo: Path,
    runtime_name: str,
    task: str,
    scan_summary: dict[str, Any],
    compose_result: dict[str, Any],
    first_run: bool,
) -> dict[str, Any]:
    projects = state.setdefault("projects", {})
    projects[str(repo)] = {
        "runtime_name": runtime_name,
        "task": task,
        "scan_summary": scan_summary,
        "last_status": compose_result.get("status"),
        "runtime_path": (compose_result.get("runtime") or {}).get("path"),
        "updated_at": _utc_now(),
    }
    state["onboarding_completed"] = True
    state["last_opened_project"] = str(repo)
    state["updated_at"] = _utc_now()
    if first_run:
        state["onboarding_completed_at"] = _utc_now()
    return state


def _capability_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    backends = diagnostics.get("available_runtime_backends") or []
    runtime_ready = bool(backends)
    run_targets = ["compatibility_server", "agent_flow"] if runtime_ready else []
    return {
        "local_run": {
            "available": bool(run_targets),
            "status": "available" if runtime_ready else "dry-run-only",
            "supported_targets": run_targets,
        },
        "local_inference": {
            "available": runtime_ready,
            "status": "available" if runtime_ready else "dry-run-only",
            "supported_targets": ["inference"] if runtime_ready else [],
        },
        "dry_run_only": not runtime_ready,
        "available_runtime_backends": backends,
    }


def _scan_warnings(scan_summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if scan_summary.get("files_scanned", 0) >= MAX_REPO_FILES:
        warnings.append(
            "folder scan hit the file limit; narrow the folder or remove generated content for better routing"
        )
    if scan_summary.get("bytes_scanned", 0) >= MAX_TOTAL_BYTES:
        warnings.append(
            "folder scan hit the byte limit; routing used a bounded summary instead of the full repository"
        )
    if not scan_summary.get("language_signals"):
        warnings.append(
            "no supported language signals were detected; add source files or choose a more specific project folder"
        )
    return warnings


def _resolve_outcome(
    *,
    compose_result: dict[str, Any],
    outcome: str,
    run_target: str,
    prompt: str | None,
    host: str,
    port: int,
    writes: str,
    test_command: str | None,
    trace_out: Path | None,
    dry_run: bool,
    dry_run_only: bool,
) -> dict[str, Any]:
    runtime_path = Path(compose_result["runtime"]["path"])
    if outcome == "export_bundle":
        return {
            "requested": outcome,
            "status": "written",
            "bundle_path": str(runtime_path.resolve()),
            "summary_path": str((runtime_path / "README.md").resolve()),
            "descriptor": compose_result.get("export_bundle"),
            "dry_run_only": dry_run_only,
        }
    if run_target == "agent_flow":
        resolved_trace_out = trace_out or (runtime_path.parent.parent / "logs" / f"agent-{runtime_path.name}.json")
        agent_result = run_agent(
            runtime_path=runtime_path,
            repo=Path(compose_result["folder"]),
            task=[compose_result["task"]],
            writes=writes,
            test_command=test_command,
            trace_out=resolved_trace_out,
            dry_run=True if dry_run_only else dry_run,
        )
        return {
            "requested": outcome,
            "status": agent_result["status"],
            "run_target": run_target,
            "runtime_path": str(runtime_path.resolve()),
            "agent": agent_result,
            "dry_run_only": dry_run_only,
        }
    if run_target == "compatibility_server":
        server_result = serve_runtime(
            runtime_path=runtime_path,
            slms_dir=None,
            host=host,
            port=port,
            dry_run=True if dry_run_only else dry_run,
        )
        return {
            "requested": outcome,
            "status": server_result["status"],
            "run_target": run_target,
            "runtime_path": str(runtime_path.resolve()),
            "server": server_result,
            "dry_run_only": dry_run_only,
        }
    inference_result = SlmRuntime.load(runtime_path).infer(
        prompt=prompt or compose_result["task"],
        dry_run=True if dry_run_only else dry_run,
    )
    return {
        "requested": outcome,
        "status": inference_result["status"],
        "run_target": run_target,
        "runtime_path": str(runtime_path.resolve()),
        "inference": inference_result,
        "dry_run_only": dry_run_only,
    }


def _translate_product_error(message: str, *, capabilities: dict[str, Any], outcome: str) -> dict[str, Any]:
    lowered = message.lower()
    if "slms directory not found" in lowered or "missing slm.yaml" in lowered:
        return {
            "code": "missing_package_metadata",
            "summary": "Composer could not find installable packages for this project.",
            "likely_cause": "The app workspace package catalog is empty or one of the packages is missing required metadata.",
            "recommended_next_action": "Install or copy validated slm packages into the app workspace packages folder, then retry the compose flow.",
        }
    if "no slm selected" in lowered or "incompatible with selected slm" in lowered:
        return {
            "code": "incompatible_selection",
            "summary": "Composer could not find a compatible slm selection for this folder.",
            "likely_cause": "The detected repository signals or package compatibility rules did not permit a safe slm selection.",
            "recommended_next_action": "Choose a more specific task prompt, install a package that matches this codebase, or allow a base fallback if that is acceptable.",
        }
    if "validation failed" in lowered:
        return {
            "code": "validation_failed",
            "summary": "The runtime bundle did not pass validation.",
            "likely_cause": "One of the selected packages or emitted runtime files is incomplete or inconsistent.",
            "recommended_next_action": "Rebuild the runtime, then validate the source packages before composing again.",
        }
    if "backend" in lowered and (
        "requires" in lowered or "does not support" in lowered or "must be one of" in lowered
    ):
        return {
            "code": "unsupported_backend_choice",
            "summary": "The selected runtime backend is not supported for this composition.",
            "likely_cause": message,
            "recommended_next_action": "Choose a compatible backend or runtime model, or switch to export mode until a supported local backend is available.",
        }
    if outcome == "local_run" and capabilities.get("dry_run_only"):
        return {
            "code": "backend_unavailable",
            "summary": "Local run is not available on this machine yet.",
            "likely_cause": "No supported runtime backend dependency was detected for the current platform.",
            "recommended_next_action": "Install an available runtime backend or use export mode until local inference support is installed.",
        }
    return {
        "code": "invalid_request",
        "summary": "Composer App could not complete this workflow.",
        "likely_cause": message,
        "recommended_next_action": "Review the exported support bundle and the compose log, then retry with a smaller folder or corrected package setup.",
    }


def _write_support_bundle(
    *,
    workspace,
    runtime_name: str,
    compose_result: dict[str, Any],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
    scan_summary: dict[str, Any],
    product_error: dict[str, Any] | None,
) -> str:
    support_dir = workspace.exports_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    target = support_dir / f"{runtime_name}-support.json"
    payload = {
        "schema_version": "1",
        "generated_at": _utc_now(),
        "product_error": product_error,
        "compose_result": compose_result,
        "diagnostics": diagnostics,
        "scan_summary": scan_summary,
        "state": state,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return str(target.resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")