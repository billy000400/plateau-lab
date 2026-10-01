# Plateau Lab development notes

Read README.md, docs/METHOD.md, docs/DATA.md and docs/HOSTING.md before changing measurement semantics.

## Runtime and structure

- `./start.sh` runs the FastAPI Explorer on 127.0.0.1:8765 with the original local engine, hardware selection, model cache and JSON collections. It supports `--port`, `--open` and `--data-dir`. The classic UI is at `/classic/`; `python app.py` remains a stdlib-server fallback.
- `requirements.txt` pins the validated local torch/transformers stack and includes `requirements-web.txt`. `requirements-remote.txt` is for a separate NDIF environment. Never upgrade the user's local environment as a side effect of hosted setup.
- `plateau/core/math.py` contains endpoint metrics and trace-safe arc segments. `trajectory.py` reexports `plateau/core/trajectory.py` for legacy callers. `plateau/core/results.py` builds schema 7 with c/d compatibility fields and an all-layer effect table.
- `engine.py` remains the local torch backend. FastAPI asks it to measure all layers. `plateau/core/experiment.py` orchestrates nnsight requests; `plateau/backends/nnsight_backend.py` supports NDIF and local trace validation.
- `server/app.py` provides jobs, token previews and local-only library/export/example/hardware APIs. `server/storage.py` owns the CSV/JSONL serializer used by both HTTP adapters. `web/records.js` handles schemas 1–7 without fabricating missing measurements.
- Hosted mode (`PLATEAU_REMOTE=1`) has per-user NDIF keys and browser-owned history. Do not expose a local collection on a shared host. Never log/persist keys; errors must redact request, shared and fallback keys.

## Measurement invariants

- c uses actual consecutive final-token vector differences in CPU float64, a previous vector across batch boundaries, compensated accumulation and one normalization over the complete run. Reference/padding rows must not enter the path.
- c and each endpoint metric have independent validity. Equal endpoints do not imply a stationary path; undefined values serialize as null. Reset all path state on OOM retry/cancellation and never save cancelled runs.
- d retains the original endpoint-distance definition and 1e-8 endpoint tolerance. Record model revision, inference precision, patch context and generation convention.
- Preserve suffix-mode patch placement and endpoint checks. Local inference retains its tested hardware policy and three-word generation; hosted inference keeps the fork's three-token convention.
- nnsight block output may be a tuple or tensor. Helpers called remotely must be registered via `ndif.register`; do not reference custom model dataclasses inside traces. Register access to module outputs in execution order.

## Validation

See docs/INTEGRATION_VALIDATION.md for commands and observed limits. Preserve the existing mathematical, real-model, suffix, context, L2, hardware, export, OOM and cancellation checks. New core checks are `check_integration.py`, `check_backends.py` and `check_web.js`. Never substitute d-derived quantities for actual arc lengths. Browser stubs are not visual validation, and local nnsight execution does not prove NDIF service availability.
