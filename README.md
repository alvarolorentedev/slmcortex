# crazy-coding-llm

- [FLOP-Efficient Coding Intelligence](docs/flop-efficient-coding-intelligence.md) — technical assessment and proof-of-concept proposal, current through June 19, 2026.

## repo-brain

`repo-brain` is a local repository-intelligence CLI for coding agents. It builds a
deterministic repository index and exposes compact localization, evidence, testing, and
patch-analysis tools without requiring an LLM.

```bash
uv sync --extra dev
uv run repo-brain index .
uv run repo-brain map
uv run repo-brain localize "fix token expiry redirect"
uv run repo-brain evidence "fix token expiry redirect"
uv run repo-brain suggest-tests "fix token expiry redirect"
uv run repo-brain validate --patch change.diff
uv run repo-brain explain-failure --log test.log
uv run repo-brain score-risk --patch change.diff
uv run repo-brain trace list
```

Add `--json` for a stable machine-readable envelope. Indexes and opt-in traces stay under
`.repo-brain/` and are ignored by Git. Test execution during validation requires
`--run-tests` or an explicit `--command`.

See [evaluation/README.md](evaluation/README.md) for the controlled raw-prompt versus
evidence-bundle experiment.
