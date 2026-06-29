# Packaged Install

This guide defines the Phase 1 packaged-product contract for Slm Cortex.

The default product path is Composer-first:

1. Install the launcher
2. Run `slmcortex doctor` or open the Composer launcher
3. Point the product at a local folder
4. Compose a runtime
5. Run or export the result

Advanced Factory commands remain available, but they are optional and are not part of the normal install path.

## Supported Platform Matrix

| Target | Baseline artifact | Notes |
| --- | --- | --- |
| macOS | `artifacts/installers/install-slmcortex-macos.sh` | MLX is available on Apple Silicon; GGUF remains optional |
| Linux | `artifacts/installers/install-slmcortex-linux.sh` | Composer-first path works without training extras |
| Windows | `artifacts/installers/install-slmcortex-windows.ps1` | PowerShell installer creates a local launcher |

Each artifact expects a wheel, source distribution, or package source path and creates an isolated virtual environment plus two launchers:

- `slmcortex`: the full product CLI
- `slmcortex-composer`: the Composer App launcher that opens directly into the guided folder-to-runtime flow

## App Workspace Contract

The packaged app workspace is external to the repository checkout.

| Path | Purpose |
| --- | --- |
| `state/` | local state, copied demo repos, and future support metadata |
| `packages/` | imported or authored slm packages |
| `runtimes/` | emitted runtime bundles |
| `exports/` | export descriptors for launcher or UI handoff |
| `logs/` | compose and diagnostics logs |
| `diagnostics/` | future support bundles and environment reports |

Default roots:

- macOS: `~/Library/Application Support/SlmCortex`
- Linux: `${XDG_STATE_HOME:-~/.local/state}/slmcortex`
- Windows: `%APPDATA%\SlmCortex`

Inspect the resolved contract at any time:

```bash
slmcortex doctor
slmcortex doctor --workspace /tmp/slmcortex-app
slmcortex-composer --help
```

## Composer-First Flow

Launch the guided Composer App workflow directly from the packaged launcher:

```bash
slmcortex-composer \
  --workspace /tmp/slmcortex-app \
  --folder /path/to/repo \
  --task "Create a FastAPI endpoint with request validation" \
  --outcome export_bundle \
  --export-logs
```

The full CLI can still compose a runtime directly from a folder and the external app workspace:

```bash
slmcortex compose-folder \
  --workspace /tmp/slmcortex-app \
  --folder /path/to/repo \
  --task "Create a FastAPI endpoint with request validation" \
  --export-descriptor /tmp/slmcortex-app/exports/repo.json
```

The command returns a structured result with:

- task hints inferred from the folder scan
- the routing decision and selected packages
- runtime composition and validation status
- an optional export descriptor path
- machine-readable diagnostics and warnings

## Smoke Validation Paths

External workspace smoke:

```bash
python scripts/run_package_product_smoke.py
```

Clean-machine style install-and-launch smoke:

```bash
python scripts/run_packaged_install_smoke.py --package-source .
```

The first script validates the external workspace layout, package import, guided Composer App export, runtime validation, local-run dry-run agent flow, and log output. The second script validates that an isolated install can launch both the main CLI and the dedicated Composer launcher without relying on repository-relative runtime state.