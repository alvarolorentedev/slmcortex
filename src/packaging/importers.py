from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

from ..shared.config import base_config

def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def import_lora(
    *,
    source: str,
    skill_id: str,
    name: str,
    output: Path,
    train_dataset: Path,
    eval_dataset: Path,
    version: str = "0.1.0",
    description: str | None = None,
    force: bool = False,
) -> dict:
    from . import package_skill

    if not source.startswith("hf://"):
        raise ValueError("only hf://owner/repo sources are supported")
    repo = source.removeprefix("hf://").strip("/")
    if repo.count("/") != 1:
        raise ValueError("hf source must be hf://owner/repo")
    base_url = f"https://huggingface.co/{repo}/resolve/main"
    with tempfile.TemporaryDirectory(prefix=f"skillcortex-import-{skill_id}-") as directory:
        root = Path(directory)
        adapter = root / "adapter"
        adapter.mkdir()
        download_file(f"{base_url}/adapter_config.json", adapter / "adapter_config.json")
        try:
            download_file(f"{base_url}/adapters.safetensors", adapter / "adapters.safetensors")
        except Exception:
            download_file(f"{base_url}/adapter_model.safetensors", adapter / "adapters.safetensors")
        config = base_config()
        (adapter / "metadata.json").write_text(
            json.dumps(
                {
                    "adapter": skill_id,
                    "source_model": config.get("source_model"),
                    "base_model": config.get("default_runtime_model") or config.get("model"),
                    "quantization": "unknown",
                    "rank": 0,
                    "trainable_parameters": 0,
                    "source": source,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        eval_summary = root / "eval-summary.json"
        eval_summary.write_text(json.dumps({"modes": {}, "tasks": {}}) + "\n")
        result = package_skill(
            skill_id=skill_id,
            name=name,
            adapter_dir=adapter,
            output=output,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_summary=eval_summary,
            version=version,
            description=description or f"Imported LoRA package from {source}.",
            force=force,
            source_artifacts={"source": source},
        )
    result["source"] = source
    return result
