import difflib
from pathlib import Path
from typing import Any


WRITE_MODES = ("off", "confirm", "on")
SKIP_DIRS = {".git", ".venv", ".skillcortex", "__pycache__", ".pytest_cache", ".ruff_cache"}
ARTIFACT_DIR_PREFIXES = ("datasets/", "runtime/", "skills/", "tmp/")
CODE_FILE_SUFFIXES = (".py", ".pyi", ".ts", ".tsx", ".js", ".jsx")
TEXT_FILE_SUFFIXES = CODE_FILE_SUFFIXES + (".md", ".txt", ".json", ".yaml", ".yml")


class ToolSandbox:
    def __init__(self, repo: Path, writes_mode: str):
        self.repo = repo.resolve()
        self.writes_mode = writes_mode

    def list_files(self, *, limit: int = 200) -> list[str]:
        files = []
        for path in sorted(self.repo.rglob("*")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file():
                files.append(path.relative_to(self.repo).as_posix())
            if len(files) >= limit:
                break
        return files

    def read_file(self, relative_path: str, *, max_chars: int = 4000) -> str:
        path = self._resolve(relative_path)
        return path.read_text()[:max_chars]

    def materialize_action(self, action: dict[str, Any], *, review_path: Path | None = None) -> dict[str, Any]:
        kind = action.get("kind") or "proposed_diff"
        if kind == "no_change":
            return {
                "kind": kind,
                "write_status": "skipped",
                "files_changed": [],
                "diff": "",
                "review_artifact_path": None,
                "summary": action.get("summary") or "No change proposed.",
            }
        if kind == "proposed_diff":
            diff = action.get("diff") or ""
            files_changed = _files_from_unified_diff(diff)
            artifact_path = None
            write_status = "proposed"
            if self.writes_mode == "on":
                _apply_unified_diff(self.repo, diff)
                write_status = "applied"
            elif self.writes_mode == "confirm":
                artifact_path = _write_review_artifact(review_path, diff)
                write_status = "review_required"
            return {
                "kind": kind,
                "write_status": write_status,
                "files_changed": files_changed,
                "diff": diff,
                "review_artifact_path": str(artifact_path) if artifact_path is not None else None,
                "summary": action.get("summary") or "Proposed diff.",
            }
        if kind != "file_replace":
            raise ValueError(f"unknown action kind: {kind}")
        relative_path = action.get("path")
        content = action.get("content")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError("file_replace action requires a non-empty path")
        if not isinstance(content, str):
            raise ValueError("file_replace action requires string content")
        path = self._resolve(relative_path)
        before = path.read_text() if path.exists() else ""
        if _looks_like_truncating_file_replace(before, content):
            raise ValueError(
                f"file_replace for {relative_path} appears to truncate an existing file; use proposed_diff or include the full updated file"
            )
        diff = _unified_diff(before, content, relative_path)
        artifact_path = None
        write_status = "proposed"
        if self.writes_mode == "on":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            write_status = "applied"
        elif self.writes_mode == "confirm":
            artifact_path = _write_review_artifact(review_path, diff)
            write_status = "review_required"
        return {
            "kind": kind,
            "write_status": write_status,
            "files_changed": [relative_path],
            "diff": diff,
            "review_artifact_path": str(artifact_path) if artifact_path is not None else None,
            "summary": action.get("summary") or f"Replace {relative_path}.",
        }

    def materialize_actions(self, actions: list[dict[str, Any]], *, review_path: Path | None = None) -> dict[str, Any]:
        if not actions:
            return {
                "kind": "no_change",
                "write_status": "skipped",
                "files_changed": [],
                "diff": "",
                "review_artifact_path": None,
                "summary": "No actions proposed.",
                "actions": [],
            }
        parts = []
        files_changed: list[str] = []
        write_status = "skipped"
        summaries: list[str] = []
        action_results: list[dict[str, Any]] = []
        review_artifact_path = None
        for action in actions:
            result = self.materialize_action(action)
            action_results.append(result)
            if result["diff"]:
                parts.append(result["diff"])
            for path in result["files_changed"]:
                if path not in files_changed:
                    files_changed.append(path)
            if result["summary"]:
                summaries.append(result["summary"])
            write_status = _merge_write_status(write_status, result["write_status"])
        diff = "\n".join(part.rstrip("\n") for part in parts if part).strip()
        if diff:
            diff += "\n"
        if self.writes_mode == "confirm" and review_path is not None and diff:
            artifact = _write_review_artifact(review_path, diff)
            review_artifact_path = str(artifact) if artifact is not None else None
            write_status = "review_required"
        return {
            "kind": "action_list" if len(actions) > 1 else action_results[0]["kind"],
            "write_status": write_status,
            "files_changed": files_changed,
            "diff": diff,
            "review_artifact_path": review_artifact_path,
            "summary": " ".join(summaries).strip() or f"Materialized {len(actions)} action(s).",
            "actions": actions,
        }

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.repo / relative_path).resolve()
        if self.repo not in (candidate, *candidate.parents):
            raise ValueError(f"path escapes repo root: {relative_path}")
        return candidate


def _write_review_artifact(review_path: Path | None, diff: str) -> Path | None:
    if review_path is None or not diff:
        return None
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(diff)
    return review_path.resolve()


def _merge_write_status(current: str, new: str) -> str:
    priority = {
        "skipped": 0,
        "not_applicable": 0,
        "proposed": 1,
        "review_required": 2,
        "approval_required": 2,
        "applied": 3,
    }
    return new if priority.get(new, 0) >= priority.get(current, 0) else current


def _files_from_unified_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path not in files:
                files.append(path)
    return files


def _looks_like_truncating_file_replace(before: str, after: str) -> bool:
    before_text = before.rstrip("\n")
    after_text = after.rstrip("\n")
    if not before_text or not after_text or before_text == after_text:
        return False
    return before_text.startswith(after_text)


def _apply_unified_diff(repo: Path, diff: str) -> None:
    if not diff.strip():
        return
    patches = _parse_unified_diff(diff)
    if not patches:
        raise ValueError("proposed_diff action requires a unified diff payload")
    for patch in patches:
        _apply_single_patch(repo, patch)


def _parse_unified_diff(diff: str) -> list[dict[str, Any]]:
    lines = diff.splitlines()
    index = 0
    patches: list[dict[str, Any]] = []
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise ValueError("malformed unified diff: missing destination header")
        from_file = lines[index][4:]
        to_file = lines[index + 1][4:]
        index += 2
        hunks: list[dict[str, Any]] = []
        while index < len(lines) and lines[index].startswith("@@"):
            header = lines[index]
            old_range, new_range = header.split("@@")[1].strip().split(" ")
            old_start, old_length = _parse_range(old_range)
            new_start, new_length = _parse_range(new_range)
            index += 1
            hunk_lines: list[tuple[str, str]] = []
            while index < len(lines):
                current = lines[index]
                if current.startswith("@@") or current.startswith("--- "):
                    break
                marker = current[0] if current else " "
                value = current[1:] if current else ""
                if marker not in {" ", "+", "-"}:
                    raise ValueError(f"unsupported diff line: {current}")
                hunk_lines.append((marker, value))
                index += 1
            hunks.append(
                {
                    "old_start": old_start,
                    "old_length": old_length,
                    "new_start": new_start,
                    "new_length": new_length,
                    "lines": hunk_lines,
                }
            )
        patches.append({"from_file": from_file, "to_file": to_file, "hunks": hunks})
    return patches


def _parse_range(token: str) -> tuple[int, int]:
    token = token[1:]
    if "," in token:
        start, length = token.split(",", 1)
        return int(start), int(length)
    return int(token), 1


def _apply_single_patch(repo: Path, patch: dict[str, Any]) -> None:
    relative_path = _normalize_diff_path(patch["to_file"])
    path = (repo / relative_path).resolve()
    if repo not in (path, *path.parents):
        raise ValueError(f"patch escapes repo root: {relative_path}")
    original_exists = path.exists()
    original_text = path.read_text() if original_exists else ""
    original_lines = original_text.splitlines()
    result_lines: list[str] = []
    source_index = 0
    for hunk in patch["hunks"]:
        target_index = max(hunk["old_start"] - 1, 0)
        result_lines.extend(original_lines[source_index:target_index])
        source_index = target_index
        for marker, value in hunk["lines"]:
            if marker == " ":
                if source_index >= len(original_lines) or original_lines[source_index] != value:
                    raise ValueError(f"diff context mismatch for {relative_path}")
                result_lines.append(value)
                source_index += 1
            elif marker == "-":
                if source_index >= len(original_lines) or original_lines[source_index] != value:
                    raise ValueError(f"diff removal mismatch for {relative_path}")
                source_index += 1
            elif marker == "+":
                result_lines.append(value)
    result_lines.extend(original_lines[source_index:])
    path.parent.mkdir(parents=True, exist_ok=True)
    trailing_newline = original_text.endswith("\n") if original_exists else True
    path.write_text("\n".join(result_lines) + ("\n" if result_lines or trailing_newline else ""))


def _normalize_diff_path(path: str) -> str:
    normalized = path.strip()
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    return normalized


def _unified_diff(before: str, after: str, relative_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )