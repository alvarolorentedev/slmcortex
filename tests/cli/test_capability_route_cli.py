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
                "capabilities": ["fastapi", "pydantic", "api endpoint creation"],
                "activation_cues": ["FastAPI", "Pydantic"],
            },
            sort_keys=False,
        )
    )


def test_route_command_emits_stable_json_contract(tmp_path, capsys):
    skills_dir = tmp_path / "skills"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("from fastapi import FastAPI\nfrom pydantic import BaseModel\n")
    write_fastapi_skill(skills_dir)

    assert (
        main(
            [
                "route",
                "--skills-dir",
                str(skills_dir),
                "--repo",
                str(repo),
                "--task",
                "Create a FastAPI endpoint with Pydantic validation",
                "--explain",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert list(result) == [
        "routing_mode",
        "skills_dir",
        "repo",
        "task",
        "repo_context",
        "selected_skills",
        "candidates",
        "fallback",
        "errors",
        "warnings",
    ]
    assert result["routing_mode"] == "capability"
    assert result["selected_skills"][0]["skill_id"] == "fastapi_contract"
    candidate = result["candidates"][0]
    assert list(candidate) == [
        "skill_id",
        "score",
        "selected",
        "compatible",
        "matched_signals",
        "negative_signals",
        "score_breakdown",
        "reason",
    ]
