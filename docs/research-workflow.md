# Research Workflow

Skill Cortex is the canonical public product surface for this repository.
This document keeps the legacy research workflow available without mixing it
into the product-first README.

## Canonical identities

- Public product CLI: `skillcortex`
- Legacy research CLI: `skill-lattice`
- Product package/runtime code: `src/skillcortex/`
- Research engine: `src/skill_lattice_coder/`

## What the research surface is for

Use the research CLI when you want to:

- train or evaluate the original skill-lattice experiments
- compare base, generic, single-skill, and lattice modes
- inspect router-analysis and composition-analysis workflows
- reproduce the research artifacts documented in this repository

## Install

The repo uses the same editable install path as the public product docs:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
```

## Dry-run smoke checks

Dry runs do not download or load a model.

```bash
skill-lattice train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

## Where to go next

- [Research Results](research-results.md)
- [Five-Seed Artifact Resume](five-seed-artifact-resume.md)
- [Repo Boundary Map](repo-boundary-map.md)
- [Examples](../examples/README.md)
