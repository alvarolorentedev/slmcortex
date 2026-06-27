import json
from pathlib import Path

import pytest

from skillcortex.cli import main
from skillcortex.runtime.dynamic import DynamicRuntime, DynamicRouteDecision


def _skill(tmp_path, skill_id, *, description, capabilities=()):
    root = tmp_path / "skills" / skill_id
    (root / "adapter").mkdir(parents=True)
    (root / "adapter" / "adapters.safetensors").write_text("weights")
    (root / "adapter" / "adapter_config.json").write_text("{}")
    (root / "skill.yaml").write_text(
        "\n".join(
            [
                "schema_version: '1'",
                "package_type: skill",
                f"skill_id: {skill_id}",
                f"name: {skill_id}",
                "version: 0.1.0",
                f"description: {description}",
                "status: complete",
                "base:",
                "  runtime_model: mlx-test-base",
                "adapter:",
                "  path: adapter/adapters.safetensors",
                "composition:",
                "  capabilities:",
                "    allowed_task_types:",
                "    - python_generation",
                "  activation:",
                "    default_route_type: adapter",
                "    scope: task",
                "    semantic_families: []",
                "  compatibility:",
                "    compatible_skills: []",
                "    incompatible_skills: []",
                "  routing:",
                "    tasks: {}",
                "capabilities:",
                *[f"- {item}" for item in capabilities],
            ]
        )
        + "\n"
    )
    return root


def test_dynamic_infer_dry_run_selects_matching_lora(tmp_path, capsys):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    _skill(tmp_path, "sql_skill", description="SQL query tuning", capabilities=["sql"])

    assert (
        main(
            [
                "infer",
                "--skills-dir",
                str(tmp_path / "skills"),
                "--prompt",
                "Fix a FastAPI validation bug",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "dry-run"
    assert output["selected_skills"] == ["fastapi_skill"]


def test_dynamic_infer_dry_run_falls_back_to_base(tmp_path, capsys):
    _skill(tmp_path, "sql_skill", description="SQL query tuning", capabilities=["sql"])

    assert (
        main(
            [
                "infer",
                "--skills-dir",
                str(tmp_path / "skills"),
                "--prompt",
                "Write a README",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["selected_skills"] == []
    assert output["reason"] == "base fallback"


def test_dynamic_router_rejects_unknown_skill(tmp_path):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")

    with pytest.raises(ValueError, match="unknown dynamic skill"):
        runtime.route(
            [{"role": "user", "content": "Fix FastAPI"}],
            router=lambda _messages, _skills: DynamicRouteDecision(
                base_model="mlx-test-base",
                selected_skills=["missing_skill"],
                task_type="python_generation",
                semantic_family=None,
                train_new_lora=False,
                reason="bad router",
            ),
        )


def test_dynamic_runtime_cache_key_includes_base_model_and_loras(tmp_path, monkeypatch):
    _skill(tmp_path, "fastapi_skill", description="FastAPI endpoint validation", capabilities=["fastapi"])
    runtime = DynamicRuntime.load(tmp_path / "skills")
    calls = []

    def fake_load_model(adapter=None, model_name=None):
        calls.append((model_name, str(adapter) if adapter else None))
        return f"model:{model_name}", "tokenizer"

    monkeypatch.setattr("skillcortex.runtime.dynamic.load_model", fake_load_model)

    first = runtime._get_model("base-a", ("fastapi_skill",))
    second = runtime._get_model("base-b", ("fastapi_skill",))

    assert first[0] == "model:base-a"
    assert second[0] == "model:base-b"
    assert len(calls) == 2
