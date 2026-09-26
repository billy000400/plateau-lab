# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Plateau Lab: a web tool for activation-plateau experiments (after Shinkle & StefanHex, "Activation Plateaus"). Two input sequences A/B are run through a Hugging Face causal LM (GPT-2, Pythia, Qwen2.5/Qwen3 base), hidden states are interpolated at one chosen block output, and the relative distance d(t) is measured at later layers and logits. README.md holds the precise experimental definitions (distance formula, token-scope modes, fixed context, endpoint checks). Read it before changing measurement semantics.

Fork of billy000400/plateau-lab (`upstream` remote), rewritten from a local stdlib-HTTP + torch-hooks app into: static frontend + FastAPI backend, served together from one Hugging Face Docker Space → NDIF via nnsight, with each user's own NDIF key. The pre-fork engine is in upstream history (`git show upstream/main:engine.py`) and was used as the numerical reference for the port.

## Commands

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn server.app:app --reload     # serves API + web/ at http://127.0.0.1:8000
```

Without `NDIF_API_KEY` (or with `PLATEAU_REMOTE=0`) models run locally through nnsight; use GPT-2 Small (`gpt2`) on the dev MacBook Air. Other env vars: `PLATEAU_WORKERS`, `PLATEAU_DEVICE`, `PLATEAU_BATCH_SIZE`, `PLATEAU_ALLOWED_ORIGINS`, `HF_TOKEN` (gated tokenizers). Never hardcode the NDIF key.

There are no tests. The upstream `check_*.py` scripts were removed; [docs/removed-checks.md](docs/removed-checks.md) records what each verified and which are worth re-implementing.

## Architecture

- **`plateau/core/`**: backend-independent. `math.py` (interpolation, d(t), word spans; also executed *inside* nnsight traces, so torch + stdlib only), `models.py` (catalog + per-`model_type` module paths for blocks/head), `experiment.py` (`run_experiment`: validation, token-scope/patch positions, orchestration of backend calls, result record `schema_version` 5).
- **`plateau/backends/nnsight_backend.py`**: `NnsightBackend(remote=...)`. Each method is one nnsight request: `natural()` = `generate` whose first step captures the natural forward (per-layer last-token states, source states at the patch layer, logits); `path()` = one `trace` over a batch whose rows 0/1 are the patched reference endpoints and the rest the path samples. Interpolation and all registered metrics (`METRICS` in `math.py`, for every layer from the patch layer on) are computed inside the trace so only small results (metric values, argmax tokens, endpoint L2) come back. In remote mode `plateau.core.math` is shipped via `ndif.register`.
- **`server/app.py`**: FastAPI; `/api/config`, `/api/tokenize`, `/api/run` (202 + job id), `/api/jobs/{id}`, `/api/jobs/{id}/cancel`. Worker pool (`PLATEAU_WORKERS`), in-memory jobs, errors as `{"error": ...}`. Mounts `web/` at `/` when present (dev and the Space).
- **`web/`**: vanilla JS/HTML/CSS, no build step, served by the API. `config.js` sets `window.PLATEAU_API` (backend base URL; empty = same origin). Keep asset paths relative so `web/` can also be hosted under a subpath.
- **Deploy**: `.github/workflows/space.yml` (API + web/ → free HF *Gradio* Space running `deploy/hf-space/app.py`, i.e. uvicorn on 7860; var `HF_SPACE`, secret `HF_TOKEN`). `Dockerfile` (CPU torch, port 7860, `PLATEAU_REMOTE=1`) for other container hosts.
- **NDIF keys**: users send theirs in `X-NDIF-Key`; `server/app.py` passes it only to that job's `NnsightBackend(api_key=...)`, which builds a per-trace `RemoteBackend` (never the global nnsight config, since jobs run concurrently). Never store or log it; job errors go through `redact()`. Optional shared access: `X-Access-Code` matching `PLATEAU_ACCESS_CODE` runs on `PLATEAU_SHARED_NDIF_KEY` (never name it `NDIF_API_KEY`: `RemoteBackend` falls back to that env var for keyless requests).

## nnsight gotchas (verified with nnsight 0.7, transformers 5)

- Block `.output` is a plain tensor in transformers 5 but may be a tuple on NDIF's versions; always go through `_hidden()`.
- Module outputs must be accessed in execution order inside a trace (loop blocks in order, then the head).
- In `generate`, `for _ in tracer.iter[:]` loses all saved values if generation stops early (EOS). Hence `min_new_tokens=max_new_tokens` and client-side truncation of the raw per-step argmax.
- PyTorch 2.14 CPU generation bus-errors on Apple Silicon; local dev uses MPS.

## Invariants worth preserving

- In `different_suffix` mode t=0/t=1 must reproduce natural A/B; requests omitting `patch_position` default to `last_token`.
- Reference endpoints are computed in the same fixed context and batch as the path; d(t) is always measured at the last token.
- Planned extensions (more interpolation methods, distance metrics) should become registries in `plateau/core` rather than branches in the backend.
