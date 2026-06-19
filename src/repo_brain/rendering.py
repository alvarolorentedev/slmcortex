from __future__ import annotations

from repo_brain.models import EvidenceBundle


def render_evidence(bundle: EvidenceBundle, max_chars: int) -> str:
    sections = [
        f"# Task\n{bundle.task}",
        "# Repository\n"
        + "\n".join(
            f"- {key}: {value}" for key, value in sorted(bundle.repository_summary.items())
        ),
    ]
    for item in bundle.items:
        sections.append(
            f"## {item.path}:{item.line_start}-{item.line_end}\n"
            f"Reason: {item.reason}\n\n```text\n{item.content}\n```"
        )
    if bundle.suggested_tests:
        suggested_tests = "\n".join(f"- `{test}`" for test in bundle.suggested_tests)
        sections.append("# Suggested tests\n" + suggested_tests)
    if bundle.warnings:
        sections.append("# Warnings\n" + "\n".join(f"- {warning}" for warning in bundle.warnings))
    output = "\n\n".join(sections)
    if len(output) <= max_chars:
        return output
    return output[: max_chars - 1].rsplit("\n", 1)[0] + "\n"
