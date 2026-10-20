# Local + hosted integration validation

Integrates PR #1 (`3a6854b`) with the c(t) implementation on main (`c8dec03`). Original commits are retained as merge ancestors. The integration uses schema 7, preserves the local engine/collections and adds c(t) to the nnsight backend and all-layer Explorer.

## Checks performed on 2026-09-30

| Area | Observed result |
| --- | --- |
| Original known-answer/hardware/export checks | 28 unittest cases passed; interpolation and tokenwise math checks passed |
| Integration contracts | 9 cases passed: every batch split, loops/stationary/nonfinite paths, float64 norms, retry/cancellation, local API persistence/exports, remote library isolation, default model and active-job restoration |
| Real local trajectory | Pythia 70M CPU, Linear/SLERP, batches 1/2/4: independent vector captures matched c, d, every segment and cumulative/total lengths; equal/stationary endpoints remained explicit |
| Local vs nnsight | Pythia 70M, every recorded layer and logits, CPU and MPS, Linear/SLERP: the original four metrics matched within `atol=rtol=2e-3`; independent nnsight tensor captures matched Python `math.dist`/`math.fsum` arc calculations within `atol=1e-9`, `rtol=1e-10` |
| Hosted dependency stack | Separate native ARM64 environment with torch 2.13.0, transformers 5.17.0 and nnsight 0.7.0: the same MPS trace/vector checks passed. This exposed and fixed GPT-NeoX's `embed_out` → `lm_head` rename |
| Local engine regressions | Pythia 70M suffix placement, fixed contexts/reversal, embedding patch, endpoint L2, offline cache load, cached decoding, injected OOM restart and cancellation/reuse passed |
| Real FastAPI HTTP | Temporary server/data directory, Pythia 70M CPU: Linear/SLERP run→poll→save→reload, both collections, selected CSV/JSONL and legacy file preservation passed |
| UI contracts | Original UI and new Explorer Node checks passed: real measured fixtures, schemas 1–7, c above d, legacy notices, per-metric validity, JSONL import, annotations, CSV quoting and classic asset routing |

The working `plateau` Conda environment and original app/data were not modified. Test dependencies were installed into separate environments. The shared model cache was read for existing Pythia weights; test runs and fixtures went to temporary directories.

## Repeat the checks

### NNsight 0.8 upgrade (2026-10-20)

- Updated the remote stack to NNsight 0.8.1 / Transformers 5.20.1 and tested with the Space's torch 2.13.0 override in a separate environment.
- The captured live NDIF status parsed as zero models with the old identifier filter. The new parser recognized 12 supported text-generation models, 11 running at the time of the check. Counts change with NDIF scheduling.
- Pythia 70M CPU and Apple MPS Linear/SLERP parity passed against the local engine for all recorded layers, arc lengths, effect metrics, natural continuations, and the three prediction columns. Matrix off/on yielded identical curves and skipped two forwards per batch when off. Loading the optional local trace model on CPU before moving it to MPS avoided a hang with direct `device_map="mps"` weight loading.
- `check_ndif_backend.py` uses a tiny random GPT-2 checkpoint generated locally. It exercises actual NNsight 0.8 trace serialization/deserialization via `remote="local"`, natural generation, both patch scopes, Linear/SLERP, and matrix off/on. It also checks meta-only loading, the new request model identity, and isolation of per-user API keys. No download or NDIF key is needed. The hosted CI job runs it alongside the API/catalog checks.
- Authenticated NDIF execution and a deployment of this upgrade have not been verified by these offline checks.

Run the new transport simulation in a separate remote test environment (use torch 2.13.0 to match the Space, 2.14.0 for the Docker requirements):

```sh
.venv-remote/bin/python -m pip install httpx==0.28.1
HF_HUB_OFFLINE=1 .venv-remote/bin/python check_ndif_backend.py
.venv-remote/bin/python check_integration.py
```

### Local engine and UI

For offline checks, install the normal local requirements and the test client dependency:

```sh
.venv/bin/python -m pip install -r requirements.txt httpx==0.28.1
.venv/bin/python -m unittest check_trajectory check_hardware check_exports
.venv/bin/python check_math.py
.venv/bin/python check_integration.py
node check_ui.js
node check_web.js
```

`.github/workflows/checks.yml` runs that set on pull requests and integration/main pushes. It needs no model download, GPU or NDIF key.

Real model checks (use a separate test environment if adding optional nnsight):

```sh
PLATEAU_DEVICE=cpu .venv/bin/python check_arc.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_inference.py --model pythia-70m --oom
PLATEAU_DEVICE=cpu .venv/bin/python check_suffix.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_l2.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_contexts.py --model pythia-70m
.venv/bin/python check_cache.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_app.py --fastapi
```

`check_backends.py` uses the remote test stack (`nnsight==0.8.1` and `transformers==5.20.1`) and a cached checkpoint locally:

```sh
PLATEAU_DEVICE=cpu .venv/bin/python check_backends.py
PLATEAU_DEVICE=mps .venv/bin/python check_backends.py  # Apple Silicon
```

The nnsight trace check captures actual tensors inside the path trace, excludes the two reference rows and verifies the cross-batch distances plus a singleton final batch. Its model calls run locally, even though they exercise the backend used for NDIF requests.

## Remaining external checks

- **Live NDIF and Space deployment:** no key was supplied and no remote inference/deployment was initiated. Transport, remote model/runtime versions, service limits and the inherited ZeroGPU startup path need a credentialed smoke test. Docker's torch 2.14.0 stack was not exercised; the Space's 2.13.0 stack was tested locally.
- **Visual browser QA:** the browser tool refused access to the isolated preview because it could not verify its admin-enforced policy. No workaround was used. HTTP and Node DOM tests do not establish visual layout, focus behavior or a complete browser journey.

For manual review, start `./start.sh --port 8767 --data-dir /absolute/path/to/test-data`. Open a preexisting schema-4 file in History, run a small Pythia example, inspect c/d overview and all-layer choices, select CSV/JSONL exports, then check `/classic/` annotations and exports. Use disposable data when checking deletion.
