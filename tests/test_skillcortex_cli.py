from skillcortex.cli import main


def test_skillcortex_cli_alias_supports_dry_run():
    assert main(["train-skill", "python_skill", "--dry-run"]) == 0
