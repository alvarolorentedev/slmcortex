import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from .backends.legacy import adapter_composition_backend
from .contracts import TASK_TYPES
from .packaging import (
    _read_json,
    _read_yaml,
    validate_composition_metadata,
    validate_skill_package,
)


validate_adapter_configs = adapter_composition_backend.validate_adapter_configs
validate_adapter_metadata = adapter_composition_backend.validate_adapter_metadata


def compose_skill_packages(
    *,
    skills: list[Path],
    strategy: str,
    output: Path,
    registry: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    if strategy != "routed":
        raise ValueError("only the routed composition strategy is currently supported")
    if not skills:
        raise ValueError("at least one skill package is required")

    loaded = [_load_skill_package(path) for path in skills]
    _validate_unique_skill_ids(loaded)
    enrichment = _load_registry_enrichment(registry, loaded)
    compatibility = _build_compatibility_report(loaded, enrichment)
    if compatibility["errors"]:
        raise ValueError(compatibility["errors"][0])

    routes = _build_routes(loaded)
    bundle = _build_bundle(loaded, routes, enrichment)
    budget = _build_budget_report(loaded, routes)

    output = output.resolve()
    output_exists = output.exists()
    if dry_run:
        return {
            "status": "dry-run",
            "strategy": strategy,
            "output": str(output),
            "skills": [item["skill_id"] for item in loaded],
            "files": sorted(_bundle_files(bundle, compatibility, budget)),
        }
    if output_exists and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} exists; pass --force to replace it")
    if output_exists:
        shutil.rmtree(output)

    with tempfile.TemporaryDirectory(prefix="skillcortex-compose-") as directory:
        staging = Path(directory) / output.name
        staging.mkdir(parents=True, exist_ok=True)
        files = _bundle_files(bundle, compatibility, budget)
        for relative, content in sorted(files.items()):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
        checksums = {
            relative: _sha256(staging / relative)
            for relative in sorted(files)
        }
        (staging / "checksums.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "files": checksums,
                    "source_packages": [
                        {
                            "skill_id": item["skill_id"],
                            "path": str(item["path"]),
                            "fingerprint": item["fingerprint"],
                        }
                        for item in loaded
                    ],
                    "registry_enrichment": enrichment,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging), str(output))
    return {
        "status": "complete",
        "strategy": strategy,
        "output": str(output),
        "skills": [item["skill_id"] for item in loaded],
    }


def _load_skill_package(path: Path) -> dict:
    root = path.resolve()
    validate_skill_package(root)
    skill_manifest = _read_yaml(root / "skill.yaml")
    metadata = _read_json(root / "metadata.json")
    adapter_config = _read_json(root / "adapter" / "adapter_config.json")
    composition = _load_composition(skill_manifest, metadata)
    return {
        "path": root,
        "skill_id": skill_manifest["skill_id"],
        "name": skill_manifest["name"],
        "version": skill_manifest["version"],
        "manifest": skill_manifest,
        "metadata": metadata,
        "adapter_config": adapter_config,
        "composition": composition,
        "fingerprint": _package_fingerprint(skill_manifest, metadata),
    }


def _load_composition(skill_manifest: dict, metadata: dict) -> dict:
    manifest_composition = skill_manifest.get("composition")
    metadata_composition = metadata.get("composition")
    if manifest_composition is None and metadata_composition is None:
        raise ValueError(
            "package missing composition metadata: composition.capabilities.allowed_task_types"
        )
    if manifest_composition is None or metadata_composition is None:
        raise ValueError("package composition metadata must exist in both skill.yaml and metadata.json")
    if manifest_composition != metadata_composition:
        raise ValueError("manifest mismatch for composition metadata")
    validate_composition_metadata(manifest_composition)
    return manifest_composition


def _validate_unique_skill_ids(loaded: list[dict]) -> None:
    seen = set()
    for item in loaded:
        if item["skill_id"] in seen:
            raise ValueError(f"duplicate skill package: {item['skill_id']}")
        seen.add(item["skill_id"])


def _build_compatibility_report(loaded: list[dict], enrichment: dict) -> dict:
    errors = []
    warnings = []
    manifests = [item["manifest"] for item in loaded]
    first_base = manifests[0].get("base") or {}
    for key in ("source_model", "runtime_model", "quantization"):
        expected = first_base.get(key)
        if any((manifest.get("base") or {}).get(key) != expected for manifest in manifests[1:]):
            errors.append(f"incompatible package base {key}")
    try:
        validate_adapter_metadata(
            [
                {
                    "base_model": item["metadata"]["base"]["runtime_model"],
                    "target_modules": item["metadata"]["adapter"]["target_modules"],
                    "quantization": item["metadata"]["base"]["quantization"],
                }
                for item in loaded
            ]
        )
    except ValueError as error:
        errors.append(str(error))
    try:
        validate_adapter_configs([item["adapter_config"] for item in loaded])
    except ValueError as error:
        errors.append(str(error))
    selected = {item["skill_id"] for item in loaded}
    for item in loaded:
        incompatible = set(
            ((item["composition"].get("compatibility") or {}).get("incompatible_skills") or [])
        )
        overlap = sorted(selected & incompatible)
        if overlap:
            errors.append(
                f"skill {item['skill_id']} is incompatible with selected skill {overlap[0]}"
            )
    for entry in enrichment.get("matched_skills", []):
        package_tasks = set(entry["package_allowed_task_types"])
        registry_tasks = set(entry["registry_allowed_task_types"])
        if registry_tasks and package_tasks != registry_tasks:
            warnings.append(
                f"registry enrichment differs from package metadata for {entry['skill_id']} allowed_task_types"
            )
    return {
        "schema_version": "1",
        "status": "valid" if not errors else "invalid",
        "skills": [item["skill_id"] for item in loaded],
        "errors": errors,
        "warnings": warnings,
        "optional_enrichment_used": enrichment["enabled"] and bool(enrichment["matched_skills"]),
        "registry_enrichment": enrichment,
    }


def _build_routes(loaded: list[dict]) -> list[dict]:
    task_scoped = [
        item for item in loaded if item["composition"]["activation"]["scope"] == "task"
    ]
    semantic_scoped = [item for item in loaded if item not in task_scoped]
    available = {item["skill_id"] for item in loaded}
    routes = []
    for task_type in TASK_TYPES:
        selected = _route_selection(task_scoped, task_type, available)
        routes.append(
            {
                "route_id": f"{task_type}.default",
                "task_type": task_type,
                "semantic_family": None,
                "route_type": "adapter" if selected else "base_fallback",
                "selected_skills": selected,
            }
        )
    semantic_families = sorted(
        {
            family
            for item in semantic_scoped
            for family in item["composition"]["activation"].get("semantic_families", [])
        }
    )
    for semantic_family in semantic_families:
        for task_type in TASK_TYPES:
            selected = _route_selection(task_scoped, task_type, available)
            selected.extend(
                _route_selection(
                    [
                        item
                        for item in semantic_scoped
                        if semantic_family
                        in item["composition"]["activation"].get("semantic_families", [])
                    ],
                    task_type,
                    available,
                )
            )
            default = next(
                route
                for route in routes
                if route["task_type"] == task_type and route["semantic_family"] is None
            )
            if selected == default["selected_skills"]:
                continue
            routes.append(
                {
                    "route_id": f"{task_type}.{semantic_family}",
                    "task_type": task_type,
                    "semantic_family": semantic_family,
                    "route_type": "adapter" if selected else "base_fallback",
                    "selected_skills": selected,
                }
            )
    return routes


def _route_selection(items: list[dict], task_type: str, available: set[str]) -> list[str]:
    selected = []
    candidates = []
    for item in items:
        allowed = item["composition"]["capabilities"].get("allowed_task_types") or []
        if task_type not in allowed:
            continue
        rule = ((item["composition"].get("routing") or {}).get("tasks") or {}).get(
            task_type, {}
        )
        requires_all = set(rule.get("requires_all_of") or [])
        requires_any = set(rule.get("requires_any_of") or [])
        if not requires_all.issubset(available):
            continue
        if requires_any and not (requires_any & available):
            continue
        candidates.append((int(rule.get("order", 100)), item["skill_id"]))
    for _, skill_id in sorted(candidates, key=lambda value: (value[0], value[1])):
        selected.append(skill_id)
    return selected


def _build_bundle(loaded: list[dict], routes: list[dict], enrichment: dict) -> dict:
    base = loaded[0]["metadata"]["base"]
    return {
        "schema_version": "1",
        "composition_type": "skill_runtime_bundle",
        "strategy": "routed",
        "status": "complete",
        "runtime": {
            "source_model": base["source_model"],
            "runtime_model": base["runtime_model"],
            "quantization": base["quantization"],
        },
        "skills": [
            {
                "skill_id": item["skill_id"],
                "name": item["name"],
                "version": item["version"],
                "package_path": str(item["path"]),
                "fingerprint": item["fingerprint"],
                "composition": item["composition"],
                "adapter": item["metadata"]["adapter"],
            }
            for item in sorted(loaded, key=lambda value: value["skill_id"])
        ],
        "routes": routes,
        "provenance": {
            "registry_enrichment": enrichment,
        },
    }


def _build_budget_report(loaded: list[dict], routes: list[dict]) -> dict:
    stored_parameters = sum(
        int(item["metadata"]["adapter"]["trainable_parameters"])
        for item in loaded
    )
    bytes_by_skill = {
        item["skill_id"]: (item["path"] / "adapter" / "adapters.safetensors").stat().st_size
        for item in loaded
    }
    params_by_skill = {
        item["skill_id"]: int(item["metadata"]["adapter"]["trainable_parameters"])
        for item in loaded
    }
    max_active_parameters = max(
        sum(params_by_skill[skill_id] for skill_id in route["selected_skills"])
        for route in routes
    )
    max_active_bytes = max(
        sum(bytes_by_skill[skill_id] for skill_id in route["selected_skills"])
        for route in routes
    )
    return {
        "schema_version": "1",
        "stored_adapter_count": len(loaded),
        "stored_adapter_parameters": stored_parameters,
        "stored_adapter_bytes": sum(bytes_by_skill.values()),
        "max_active_adapter_count": max(len(route["selected_skills"]) for route in routes),
        "max_active_adapter_parameters": max_active_parameters,
        "max_active_adapter_bytes": max_active_bytes,
        "memory_estimate_method": "active packaged adapter file bytes",
    }


def _bundle_files(bundle: dict, compatibility: dict, budget: dict) -> dict[str, str]:
    active_skills = {
        "schema_version": "1",
        "skills": [
            {
                "skill_id": item["skill_id"],
                "name": item["name"],
                "version": item["version"],
                "package_path": item["package_path"],
                "allowed_task_types": item["composition"]["capabilities"]["allowed_task_types"],
                "activation": item["composition"]["activation"],
                "trainable_parameters": item["adapter"]["trainable_parameters"],
                "route_membership": [
                    route["route_id"]
                    for route in bundle["routes"]
                    if item["skill_id"] in route["selected_skills"]
                ],
            }
            for item in bundle["skills"]
        ],
    }
    router_config = {
        "schema_version": "1",
        "strategy": bundle["strategy"],
        "routes": bundle["routes"],
    }
    readme = _build_readme(bundle, compatibility, budget)
    return {
        "composition.yaml": yaml.safe_dump(bundle, sort_keys=False),
        "router_config.json": json.dumps(router_config, indent=2, sort_keys=True) + "\n",
        "active_skills.json": json.dumps(active_skills, indent=2, sort_keys=True) + "\n",
        "compatibility_report.json": json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
        "budget_report.json": json.dumps(budget, indent=2, sort_keys=True) + "\n",
        "README.md": readme,
    }


def _build_readme(bundle: dict, compatibility: dict, budget: dict) -> str:
    lines = [
        "# SkillCortex Runtime Bundle",
        "",
        f"- Strategy: `{bundle['strategy']}`",
        f"- Skills: {', '.join(item['skill_id'] for item in bundle['skills'])}",
        f"- Runtime model: `{bundle['runtime']['runtime_model']}`",
        f"- Compatibility status: `{compatibility['status']}`",
        f"- Max active adapter parameters: **{budget['max_active_adapter_parameters']}**",
        f"- Optional registry enrichment used: `{compatibility['optional_enrichment_used']}`",
        "",
    ]
    return "\n".join(lines)


def _load_registry_enrichment(registry: Path | None, loaded: list[dict]) -> dict:
    if registry is None:
        return {
            "enabled": False,
            "path": None,
            "matched_skills": [],
            "source_of_truth": "package",
            "override_applied": False,
        }
    payload = _read_json(registry.resolve())
    skills = payload.get("skills") or []
    by_name = {
        item.get("skill_name"): item
        for item in skills
        if isinstance(item, dict) and item.get("skill_name")
    }
    matched = []
    for item in loaded:
        registry_skill = by_name.get(item["skill_id"])
        if registry_skill is None:
            continue
        matched.append(
            {
                "skill_id": item["skill_id"],
                "registry_status": registry_skill.get("status"),
                "registry_origin": registry_skill.get("origin"),
                "registry_router": registry_skill.get("router"),
                "registry_activation_scope": registry_skill.get("activation_scope"),
                "registry_allowed_task_types": registry_skill.get("allowed_task_types") or [],
                "package_allowed_task_types": item["composition"]["capabilities"].get(
                    "allowed_task_types"
                )
                or [],
            }
        )
    return {
        "enabled": True,
        "path": str(registry.resolve()),
        "matched_skills": matched,
        "source_of_truth": "package",
        "override_applied": False,
    }


def _package_fingerprint(skill_manifest: dict, metadata: dict) -> str:
    payload = json.dumps(
        {
            "skill_id": skill_manifest["skill_id"],
            "version": skill_manifest["version"],
            "base": metadata["base"],
            "adapter": metadata["adapter"],
            "checksums": metadata["checksums"],
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()