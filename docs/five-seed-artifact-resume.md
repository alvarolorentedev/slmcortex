# Five-Seed Artifact Resume

Source artifacts:

- `artifacts/experiments/five-seed/summary.json`
- `artifacts/experiments/five-seed/router_analysis.md`
- `artifacts/experiments/five-seed/python_regression_analysis.md`

## Outcome

The five-seed experiment does not clear the strict practical threshold, but it
does show a statistically supported gain for the routed lattice over the
generic LoRA baseline.

- Generic LoRA: **40.67%** execution pass rate
- Routed lattice: **45.60%**
- Oracle lattice: **47.60%**
- Paired gain over generic: **+4.93 percentage points**
- 95% bootstrap CI: **+1.47 to +8.40 points**
- Routed lattice won in **all five seeds**
- Active adapter parameters: **427,513 average**
- Generic active adapter parameters: **933,888**
- Active-parameter reduction: **54.2%**

## Task Breakdown

| Mode | Python generation | Debugging | Test generation |
|---|---:|---:|---:|
| Base | 54.0% | 28.0% | 18.0% |
| Generic LoRA | 37.6% | 34.0% | 50.4% |
| Single skill | 38.0% | 42.0% | 52.8% |
| Routed lattice | 37.6% | 46.4% | 52.8% |
| Oracle lattice | 38.0% | 46.4% | 58.4% |

## Interpretation

- The gain comes from debugging and test generation.
- Python generation regressed relative to the frozen base.
- Oracle routing is stronger than routed lattice, so routing remains the main
  lever.
- The result supports the statistical claim, but not the stricter five-point
  practical threshold.

## Reproducibility

- Seeds: `11`, `22`, `33`, `44`, `55`
- Benchmark size: `750` executions per mode
- Benchmark families: `50`
- Models: `Qwen/Qwen2.5-Coder-1.5B-Instruct` and `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`

## Related artifacts

- FastAPI contract baseline: `artifacts/experiments/fastapi-contract-baseline/summary.md` records per-behavior diagnostics and latency comparisons for `skillcortex_router_v1`.
- Recent evaluation: `artifacts/evaluations/20260620T152056Z/report.md` contains a diagnostic report labeling its hypothesis inconclusive and reporting per-mode metrics.
