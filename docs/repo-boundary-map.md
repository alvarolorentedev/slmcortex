# Repo Boundary Map

Phase 0 keeps the research engine intact and separates future product surfaces
without moving behavior.

## Canonical boundaries

- `src/skill_lattice_coder/`: current research/core engine.
- `src/skillcortex/`: product CLI façade and compatibility alias.
- `configs/`: runtime defaults and governed skill registry inputs.
- `data/`: current canonical datasets and benchmarks.
- `skills/`: skill catalog mirror and package artifacts.
- `examples/`: runnable examples and smoke snippets.
- `docs/`: specs, architecture notes, and user-facing documentation.
- `reports/`: curated human-readable summaries.
- `scripts/`: research, validation, and experiment helpers.
- `tests/`: unit, integration, and regression coverage.
- `artifacts/`: immutable generated evidence, adapters, and experiment outputs.

## Phase 0 policy

- Do not change model behavior, adapters, registry semantics, or benchmark data.
- Do not rename the research package yet.
- Keep generated artifacts immutable.
- Add new product-facing layers only as thin compatibility wrappers later.

## Current source of truth

- Core runtime: `src/skill_lattice_coder/`
- CLI entry point: `skill-lattice`
- Skill registry: `configs/skill_registry.json`
- Skill metadata: `configs/skills.yaml`
- Skill mirror: `skills/skill_registry.json`, `skills/skills.yaml`
- Datasets and benchmarks: `data/`
- Reports and provenance: `artifacts/`
