# Evaluation

The harness compares the same agent under two conditions:

- `raw`: task text only.
- `evidence`: task text plus a freshly generated repo-brain evidence bundle.

The agent command must accept formatted `{repo}`, `{prompt}`, `{patch}`, and `{metrics}`
paths and write optional JSON metrics:

```json
{
  "predicted_files": ["src/example.py"],
  "selected_tests": ["tests/test_example.py::test_case"],
  "patch_success": true,
  "iterations": 2,
  "input_tokens": 4000,
  "output_tokens": 500
}
```

Run:

```bash
uv run python -m evaluation.run examples/evaluation/tasks.jsonl \
  --agent-command 'agent --repo {repo} --prompt-file {prompt} --patch-out {patch} --metrics-out {metrics}'

uv run python -m evaluation.score evaluation-results.jsonl
```

Use at least 30 development tasks across six repositories, then 100 untouched
confirmation tasks across twenty repositories. Keep model settings, timeouts, tools, and
attempt budgets identical between arms. Review generated patches blind to arm assignment.

