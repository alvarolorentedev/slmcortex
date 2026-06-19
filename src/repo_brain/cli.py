from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from repo_brain import __version__
from repo_brain.indexer import index_repository
from repo_brain.repository import RepositoryError, resolve_repository
from repo_brain.skills.repo_map import repository_map


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repo-brain")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("path", nargs="?", type=Path)
    subparsers.add_parser("map")
    return parser


def _envelope(
    command: str,
    repository: str,
    data: object,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": command,
        "repository": repository,
        "data": data,
        "warnings": warnings,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = getattr(args, "path", None) or args.repo
    try:
        root = resolve_repository(selected)
    except RepositoryError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if args.command == "map":
        map_data = repository_map(root)
        if args.json:
            print(json.dumps(_envelope("map", str(root), map_data, [], []), sort_keys=True))
        else:
            print(json.dumps(map_data, indent=2, sort_keys=True))
        return 0
    result = index_repository(root)
    data: dict[str, Any] = asdict(result)
    warnings = list(result.warnings)
    errors = list(result.errors)
    data.pop("warnings")
    data.pop("errors")
    if args.json:
        print(json.dumps(_envelope("index", str(root), data, warnings, errors), sort_keys=True))
    elif errors:
        print("\n".join(errors), file=sys.stderr)
    else:
        print(
            f"indexed {data['scanned']} files: {data['added']} added, "
            f"{data['updated']} updated, {data['unchanged']} unchanged, "
            f"{data['removed']} removed, {data['ignored']} ignored, {data['failed']} failed"
        )
    return 3 if errors else 0
