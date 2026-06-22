# SkillCortex Router V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `alternating_skill` behind a strict semantic gate and publish an artifact-only regression report.

**Architecture:** Keep the protected router unchanged under an explicit baseline name. Add one thin router wrapper and one deterministic report transformation.

**Tech Stack:** Python, pytest, JSON, Markdown.

---

### Task 1: Lock routing behavior

**Files:**
- Modify: `tests/test_router.py`
- Modify: `src/skill_lattice_coder/router.py`
- Modify: `src/skill_lattice_coder/inference.py`
- Modify: `src/skill_lattice_coder/schemas.py`

- [ ] Add failing tests for the baseline alias, exact alternating routes, and unchanged delegation.
- [ ] Run `pytest tests/test_router.py tests/test_inference.py -q` and confirm failure.
- [ ] Add the minimum router/schema/inference changes.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Promote config and generate the integration report

**Files:**
- Modify: `configs/skills.yaml`
- Create: `scripts/build_skillcortex_router_v1_report.py`
- Create: `tests/test_skillcortex_router_v1.py`

- [ ] Add failing tests for promoted config, artifact-only report generation, checksums, and preserved quarantine metadata.
- [ ] Run `pytest tests/test_skillcortex_router_v1.py -q` and confirm failure.
- [ ] Implement the deterministic summary transformation and Markdown output.
- [ ] Generate `artifacts/experiments/skillcortex-router-v1/summary.json` and `summary.md`.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/research-results.md`

- [ ] Record phase completion and both validated mechanisms.
- [ ] Run the smallest relevant pytest set.
- [ ] Confirm `data/eval.jsonl` and the historical experiment summary checksums are unchanged.
