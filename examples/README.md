# Examples

## Dry-run smoke checks

```bash
skill-lattice train-skill python_skill --dry-run
skillcortex train-skill python_skill --dry-run
skill-lattice train-generic --dry-run
skill-lattice infer --mode lattice --task-type debugging --prompt "Fix this Python traceback" --dry-run
skill-lattice eval --dataset data/eval.jsonl --dry-run
```

## Repository map

- [`docs/repo-boundary-map.md`](../docs/repo-boundary-map.md)
