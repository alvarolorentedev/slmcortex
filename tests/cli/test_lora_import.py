import json
from pathlib import Path

from skillcortex.cli import main


def test_import_lora_from_huggingface_source(tmp_path, monkeypatch, capsys):
    def fake_download(url, destination):
        destination.write_text(
            "{}" if destination.name.endswith(".json") else "weights"
        )

    monkeypatch.setattr("skillcortex.packaging.importers.download_file", fake_download)

    assert (
        main(
            [
                "import-lora",
                "--source",
                "hf://owner/repo",
                "--skill-id",
                "fastapi_skill",
                "--name",
                "FastAPI Skill",
                "--output",
                str(tmp_path / "fastapi_skill"),
                "--train-dataset",
                "data/train.jsonl",
                "--eval-dataset",
                "data/eval.jsonl",
                "--force",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert (tmp_path / "fastapi_skill" / "skill.yaml").exists()
    assert (tmp_path / "fastapi_skill" / "adapter" / "adapters.safetensors").exists()
