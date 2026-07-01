import json
from pathlib import Path

from slmcortex.cli import main
from slmcortex.packaging import package_slm


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())


def _fake_hf_download(monkeypatch):
    def fake_snapshot_download(repo_id, *, revision, local_dir, allow_patterns):
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text("{}")
        (root / "adapters.safetensors").write_text("weights")
        return str(root)

    monkeypatch.setattr("slmcortex.packaging.importers.snapshot_download", fake_snapshot_download)


def _project_config(root: Path) -> None:
    (root / ".slmcortex.yaml").write_text(
        "\n".join(
            [
                "slms_dir: .slmcortex/slms",
                "lora_cache_dir: .slmcortex/lora-cache",
                "loras:",
                "  fastapi:",
                "    source: hf://owner/fastapi-lora",
                "    name: FastAPI LoRA",
                "    description: FastAPI and Pydantic coding tasks",
                "  pytest:",
                "    source: hf://owner/pytest-lora",
                "    name: Pytest LoRA",
                "",
            ]
        )
    )


def _package(root: Path) -> None:
    eval_summary = root / "eval-summary.json"
    eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
    package_slm(
        slm_id="fastapi",
        name="FastAPI",
        adapter_dir=ROOT / "artifacts/adapters/python_slm",
        output=root / ".slmcortex" / "slms" / "fastapi",
        train_dataset=ROOT / "data/train.jsonl",
        eval_dataset=ROOT / "data/eval.jsonl",
        eval_summary=eval_summary,
        version="0.1.1",
        description="FastAPI endpoint validation",
        composition={
            "capabilities": {"allowed_task_types": ["python_generation"]},
            "activation": {
                "default_route_type": "adapter",
                "scope": "task",
                "semantic_families": ["fastapi"],
            },
            "compatibility": {"compatible_slms": [], "incompatible_slms": []},
            "routing": {"tasks": {}},
        },
        force=True,
    )
    (root / ".slmcortex" / "slms" / "fastapi" / "routing_card.json").write_text(
        json.dumps({"positive_examples": ["Fix a FastAPI validation bug"]}) + "\n"
    )


def test_init_writes_project_config_and_folders(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["init"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert (tmp_path / ".slmcortex.yaml").exists()
    assert (tmp_path / ".slmcortex" / "slms").is_dir()
    assert (tmp_path / ".slmcortex" / "lora-cache").is_dir()
    assert (tmp_path / ".slmcortex" / "runtimes").is_dir()


def test_loras_download_imports_only_named_config_entries(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _project_config(tmp_path)
    _fake_hf_download(monkeypatch)

    assert main(["loras", "download", "fastapi"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["downloaded"] == ["fastapi"]
    assert (tmp_path / ".slmcortex" / "slms" / "fastapi" / "slm.yaml").exists()
    assert not (tmp_path / ".slmcortex" / "slms" / "pytest").exists()


def test_loras_download_accepts_one_off_hf_url(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _fake_hf_download(monkeypatch)

    assert main(["loras", "download", "hf://owner/repo", "--as", "fastapi"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["downloaded"] == ["fastapi"]
    assert (tmp_path / ".slmcortex" / "slms" / "fastapi" / "slm.yaml").exists()


def test_loras_download_requires_names_or_all(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _project_config(tmp_path)

    assert main(["loras", "download"]) == 2
    assert "choose LoRA names or use --all" in capsys.readouterr().err


def test_serve_and_agent_use_project_config_defaults(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _project_config(tmp_path)
    _package(tmp_path)

    assert main(["serve", "--dry-run"]) == 0
    serve_output = json.loads(capsys.readouterr().out)
    assert serve_output["status"] == "dry-run"
    assert serve_output["slms"] == ["fastapi"]

    assert main(["agent", "run", "--task", "Fix a FastAPI validation bug", "--dry-run"]) == 0
    agent_output = json.loads(capsys.readouterr().out)
    assert agent_output["mode"] == "dynamic_agent"
    assert agent_output["agent_execution_status"] == "dry_run_completed"
