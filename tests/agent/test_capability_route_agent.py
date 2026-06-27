import json

import yaml

from skillcortex.cli import main


def write_fastapi_skill(skills_dir):
    package = skills_dir / "fastapi_contract"
    package.mkdir(parents=True)
    (package / "skill.yaml").write_text(
        yaml.safe_dump(
            {
                "skill_id": "fastapi_contract",
                "name": "FastAPI Contract Skill",
                "description": "FastAPI endpoints with Pydantic validation.",
                "capabilities": ["fastapi", "pydantic"],
                "activation_cues": ["FastAPI", "Pydantic"],
            },
            sort_keys=False,
        )
    )


def test_agent_run_skills_dir_dry_run_returns_route_json(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\n")
    write_fastapi_skill(skills_dir)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--dry-run",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["routing_mode"] == "capability"
    assert result["selected_skills"][0]["skill_id"] == "fastapi_contract"


def test_agent_run_skills_dir_without_dry_run_fails_clearly(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    write_fastapi_skill(skills_dir)

    assert (
        main(
            [
                "agent",
                "run",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint",
            ]
        )
        == 2
    )

    assert "compose/pass an explicit runtime" in capsys.readouterr().err
