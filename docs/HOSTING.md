# Hosted Explorer and NDIF

[← README](../README.md)

The integration retains Lasse's FastAPI/NDIF service, all-layer plots, live model catalog, token previews and per-viewer keys. The default `./start.sh` runs the tested local torch engine. Hosted dependencies belong in a **separate environment**:

```sh
python3.12 -m venv .venv-remote
.venv-remote/bin/python -m pip install -r requirements-remote.txt
PLATEAU_REMOTE=1 .venv-remote/bin/uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open port 8000, enter an NDIF API key in Settings and select an available model. Keep private keys out of source control. The local launcher explicitly sets `PLATEAU_REMOTE=0`, even if your shell has an NDIF key.

| Setting | Meaning |
| --- | --- |
| `PLATEAU_REMOTE=1` | NDIF execution and browser-owned history; no local collection APIs |
| `PLATEAU_WORKERS` | Concurrent NDIF jobs, default 4; local mode always serializes inference |
| `PLATEAU_BATCH_SIZE` | Remote path samples per trace, default 32; OOM halves the batch and restarts the path |
| `PLATEAU_SHARED_NDIF_KEY` + `PLATEAU_ACCESS_CODE` | Optional lab key with a viewer access code |
| `NDIF_API_KEY` | Optional fallback key for private development; keyless visitors can use this key |
| `HF_TOKEN` | Tokenizer access for gated models, subject to their licenses |
| `PLATEAU_ALLOWED_ORIGINS` | Comma-separated hosted CORS origins; default `*` |
| `PLATEAU_DATA_DIR` | Local-only data folder; alternatively use `./start.sh --data-dir ...` |

Per-user keys travel in request headers and are not kept in jobs or saved results. The browser keeps them for the session unless the user selects Remember. Runtime error messages redact supplied, shared and fallback keys.

Hosted results use schema 7 and contain c(t), d(t), raw arc lengths, other endpoint metrics and validity per metric. Every path trace returns scalar lengths and one final vector per readout; the next trace uses that vector for the cross-batch segment. This adds data transfer proportional to one vector per measured layer/logit readout, rather than returning the full trajectory. c is normalized only after every sample has been collected. Local generation shows three words; NDIF generation shows three tokens. NDIF inference precision/model revisions can differ from local float32 runs and are recorded in each result.

## Hugging Face Space

The inherited `.github/workflows/space.yml` uploads on relevant pushes to `main` only when the repository variable `HF_SPACE` is set. Set that variable to your Space ID and provide a write-scoped `HF_TOKEN` repository secret. Runtime keys/access codes belong in Space secrets. The workflow includes all shared measurement code and `requirements-web.txt`, and overrides torch to 2.13.0 for the inherited ZeroGPU setup.

The Space entry point uses the Gradio SDK to start uvicorn on port 7860. Its inherited ZeroGPU registration stub is never used for inference; inference runs on NDIF. `Dockerfile` remains available for other container hosts and installs `requirements-remote.txt`. No Space was deployed during this integration.

The HTTP service and local nnsight traces are covered by integration checks. Live NDIF transport, hosted precision and the inherited ZeroGPU startup behavior require a credentialed Space smoke test before deployment; see [validation](INTEGRATION_VALIDATION.md).
