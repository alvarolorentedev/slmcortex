# Repo Boundary Map

Skill Cortex is the canonical public product surface for this repository.
The legacy research engine remains available for experimentation and internal
analysis, but public product docs should point new users at `skillcortex`
first.

## Canonical boundaries

- `src/skillcortex/`: public product CLI, package/runtime tooling, and bounded
	agent surface.
- `src/skill_lattice_coder/`: underlying research/core engine kept for legacy
	experiments and backward compatibility.
- `configs/`: runtime defaults and governed skill registry inputs.
- `data/`: current canonical datasets and benchmarks.
- `skills/`: skill catalog mirror and package artifacts.
- `examples/`: runnable examples and smoke snippets.
- `docs/`: specs, architecture notes, and user-facing documentation.
- `reports/`: curated human-readable summaries.
- `scripts/`: research, validation, and experiment helpers.
- `tests/`: unit, integration, and regression coverage.
- `artifacts/`: immutable generated evidence, adapters, and experiment outputs.

## Stability policy

- Do not change model behavior, adapters, registry semantics, or benchmark data.
- Keep `skillcortex` as the canonical public identity.
- Keep generated artifacts immutable.
- Keep public documentation product-first and move research-only guidance into
	dedicated docs.

## Current source of truth

- Public CLI entry point: `skillcortex`
- Product runtime/package implementation: `src/skillcortex/`
- Underlying research engine: `src/skill_lattice_coder/`
- Skill registry: `configs/skill_registry.json`
- Skill metadata: `configs/skills.yaml`
- Skill mirror: `skills/skill_registry.json`, `skills/skills.yaml`
- Datasets and benchmarks: `data/`
- Reports and provenance: `artifacts/`
