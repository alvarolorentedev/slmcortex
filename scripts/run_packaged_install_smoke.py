import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(name: str, command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd or ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an isolated virtual environment, install Slm Cortex, and launch the Composer-first entry point.",
    )
    parser.add_argument("--package-source", default=str(ROOT))
    parser.add_argument("--workspace-root")
    parser.add_argument("--install-root")
    parsed = parser.parse_args(argv)

    install_root = (
        Path(parsed.install_root).resolve()
        if parsed.install_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-install-"))
    )
    workspace_root = (
        Path(parsed.workspace_root).resolve()
        if parsed.workspace_root
        else Path(tempfile.mkdtemp(prefix="slmcortex-installed-workspace-"))
    )
    installer_path, launcher_path, composer_launcher_path = _installer_contract(install_root)
    env = dict(os.environ)
    env["SLMCORTEX_INSTALL_ROOT"] = str(install_root)

    steps = [
        _run("install_package", installer_path + [parsed.package_source], cwd=ROOT, env=env),
        _run("launch_help", [str(launcher_path), "--help"]),
        _run("composer_launcher_help", [str(composer_launcher_path), "--help"]),
        _run(
            "doctor",
            [str(launcher_path), "doctor", "--workspace", str(workspace_root)],
        ),
    ]

    summary = {
        "status": "complete",
        "install_root": str(install_root),
        "workspace_root": str(workspace_root),
        "steps": [
            {
                "name": step["name"],
                "command": step["command"],
            }
            for step in steps
        ],
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _installer_contract(install_root: Path) -> tuple[list[str], Path, Path]:
    if os.name == "nt":
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-windows.ps1"
        return (
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            install_root / "slmcortex.cmd",
            install_root / "slmcortex-composer.cmd",
        )
    system = os.uname().sysname
    if system == "Darwin":
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-macos.sh"
    else:
        script = ROOT / "artifacts" / "installers" / "install-slmcortex-linux.sh"
    return (["sh", str(script)], install_root / "slmcortex", install_root / "slmcortex-composer")


if __name__ == "__main__":
    raise SystemExit(main())