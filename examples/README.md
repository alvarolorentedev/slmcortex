# Examples

## Skill Cortex v0.1

Run the no-model end-to-end demo:

```bash
python scripts/run_skillcortex_demo.py
```

For the manual command-by-command quickstart and product overview, see
[README.md](../README.md).

## Arbitrary Skill Smoke

Run the default no-model arbitrary-skill smoke flow for the tiny `fastapi_contract` fixture:

```bash
python scripts/run_skillcortex_arbitrary_skill_smoke.py
```

Run the opt-in real local training path:

```bash
python scripts/run_skillcortex_arbitrary_skill_smoke.py --real-training
```

The default mode stages a demo adapter and validates package, compose, runtime, infer dry-run, and agent dry-run without real model training. The `--real-training` mode is slow, local-only, and intentionally excluded from normal CI.

## Research Dry-Run Smoke Checks

```bash
skill-lattice train-skill python_skill --dry-run
skillcortex train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

Research workflow context lives in [docs/research-workflow.md](../docs/research-workflow.md).

## Repository Map

- [docs/repo-boundary-map.md](../docs/repo-boundary-map.md)
