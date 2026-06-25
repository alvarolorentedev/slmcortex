# Phase 1 Research Results

Curated summary copy of the detailed research write-up in
[`docs/research-results.md`](../docs/research-results.md).

## What it says

- Protected routing is the current default.
- The five-seed lattice result beats generic on execution pass rate, but
  narrowly misses the stricter five-point practical threshold.
- Routing quality remains the main optimization target.
- `alternating_skill` is promoted behind the strict `skillcortex_router_v1`
  gate and remains governed by registry metadata.

## Primary artifacts

- `artifacts/experiments/five-seed/summary.json`
- `artifacts/experiments/five-seed/router_analysis.json`
- `artifacts/experiments/five-seed/python_regression_analysis.json`
- `artifacts/experiments/five-seed/composition_analysis.json`

## Reproduction

```bash
python scripts/run_seeds.py --seeds 11 22 33 44 55 --output artifacts/experiments/five-seed
```
