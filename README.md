# MARS V · Plateau Lab

A web tool for collecting activation-plateau examples. Compare two sequences, generate their next three words, and inspect measured d(t) curves with **GPT-2, Pythia, or Qwen**. Inference runs through [nnsight](https://nnsight.net), either remotely on [NDIF](https://ndif.us) or locally on the server's own hardware.

Fork of [billy000400/plateau-lab](https://github.com/billy000400/plateau-lab), restructured from a local desktop tool into a web app: a static frontend and a small API, served together from one Hugging Face Space, with inference on NDIF.

## Architecture

```
web/  (static, served by the API)  ──►  server/ (FastAPI, Hugging Face Space)  ──►  NDIF via nnsight
                                   plateau/core      experiment definitions, math, result record
                                   plateau/backends  nnsight traces (remote=True on NDIF, or local)
```

The browser posts an experiment to `/api/run`, then polls `/api/jobs/{id}`. Up to `PLATEAU_WORKERS` experiments run at once (default 4 on NDIF); later ones queue. On NDIF, each user supplies their own API key (Settings ⚙ in the page), sent in the `X-NDIF-Key` header with each run and used only for that job; the server never stores or logs it, and masks it in error messages. Nothing else is stored server-side. Completed runs are saved in the browser's IndexedDB and listed under **History** (search, reopen, export/import as JSON to share with collaborators, delete one or all); the input draft is kept in `localStorage`, the key in `sessionStorage` (or `localStorage` if "Remember on this device" is checked).

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/) (or plain `venv` + `pip`):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/uvicorn server.app:app --reload
```

Open http://127.0.0.1:8000. Without an NDIF key, models are downloaded from Hugging Face and run locally (Apple MPS, then CUDA, then CPU), float32 with eager attention. GPT-2 Small is the local default and fits on a laptop. With PyTorch 2.14 on Apple Silicon, CPU generation crashes with a bus error; MPS works.

To use NDIF, set the key in the environment (e.g. in a git-ignored `.env` that you `source`), never in code:

```sh
export NDIF_API_KEY=...
.venv/bin/uvicorn server.app:app
```

| Variable | Meaning |
|---|---|
| `NDIF_API_KEY` | Fallback NDIF key for runs without a user key (local dev). Enables remote execution by default. Leave unset on a public deployment so users bring their own. |
| `PLATEAU_REMOTE` | `1` / `0` to force NDIF or local execution (the Docker image sets `1`). |
| `PLATEAU_WORKERS` | Concurrent experiments (default 4 remote, 1 local). |
| `HF_TOKEN` | Hugging Face read token; needed for gated tokenizers (Llama, Gemma). |
| `PLATEAU_SHARED_NDIF_KEY`, `PLATEAU_ACCESS_CODE` | Optional shared access: people who enter the access code (Settings ⚙) run on this NDIF key, which never leaves the server. Use a long passphrase; wrong codes are answered after a 1 s delay. Deliberately not `NDIF_API_KEY`, which nnsight would use for every keyless request. |
| `PLATEAU_DEVICE` | Local device override (`mps`, `cuda`, `cuda:N`, `cpu`). |
| `PLATEAU_BATCH_SIZE` | Path samples per forward request (default 32). |
| `PLATEAU_ALLOWED_ORIGINS` | Comma-separated CORS origins (default `*`). |

## Deploy

One free Hugging Face **Gradio** Space serves both the API and the frontend (same origin, so no CORS setup). The Gradio SDK just installs `requirements.txt` and runs `app.py`; [`deploy/hf-space/app.py`](deploy/hf-space/app.py) starts the FastAPI app with uvicorn on port 7860 (Gradio itself is unused).

1. Create a Space with the **Gradio** SDK and free CPU hardware (e.g. `lmjantsch/plateau-lab`).
2. In this GitHub repository, add the variable `HF_SPACE` (the Space id) and the secret `HF_TOKEN` (a Hugging Face token with write access). `.github/workflows/space.yml` assembles `plateau/`, `server/`, `web/`, `app.py`, a CPU-torch `requirements.txt`, and the Space metadata, and uploads them on every change to `main`.
3. In the Space settings, add the secret `HF_TOKEN` (read access, from an account that accepted the Llama and Gemma licenses) for gated tokenizers. Optionally add the secrets `PLATEAU_SHARED_NDIF_KEY` and `PLATEAU_ACCESS_CODE` to let collaborators without a key use yours via the code. Never set `NDIF_API_KEY` on a shared deployment: everyone would run on it without a code.
4. Open `https://<user>-<space>.hf.space`. Free CPU Spaces sleep after about 48 h without traffic; the page waits for the server to wake.

The `Dockerfile` runs the same app on any container host (including a Docker Space, where available).

`web/config.js` can point the frontend at a backend on another origin (`window.PLATEAU_API`), e.g. to host `web/` separately; then set `PLATEAU_ALLOWED_ORIGINS` on the backend.

## Using the workbench

1. Enter sequences A and B. The initial pair is `The house was big` / `The house was in`.
2. Drag **Interpolation start** to **After layer N** (numbered from 0; default 5, or the last layer for smaller models). Its range follows the model's layer count.
3. Choose **Tokens to interpolate**. **First difference → end** (default) requires equal token counts and interpolates the first differing position and every position after it. **Final token only** supports unequal lengths.
4. Select **Run experiment** (or **Cmd/Ctrl+Enter**). Each panel shows a greedy continuation of three words; **Input tokenization** highlights the interpolated positions; the charts show measured d(t) at the final token.

**A ↔ B · L2 distances** shows the raw source distance at the interpolation layer (for several tokens, the Frobenius norm of the state-matrix difference, plus per-token distances) and the final-token L2 distance at every layer, for the original A/B forwards and for the patched endpoints in the fixed context.

## Experimental definitions

- **Models:** Hugging Face checkpoints `openai-community/gpt2`, `gpt2-medium`, `gpt2-large`, and `gpt2-xl`; EleutherAI Pythia **70M, 160M, 410M, 1B, 1.4B, 2.8B**; Qwen2.5 base **0.5B, 1.5B, 3B**; Qwen3 base **0.6B, 1.7B**. These are base text-completion models; the input is used exactly as typed, without a chat template. Remote runs use whatever checkpoints and precision NDIF serves, so availability depends on NDIF's deployments; each record stores the dtype actually observed. Local runs use float32 weights and eager attention.
- **Word count:** Intended for English text. Words consist of letters or digits with optional internal apostrophes or hyphens. Greedy generation continues until a fourth word begins, confirming the third word boundary. A suffix extending the prompt's final word does not count as a new word. End-of-sequence or the 48-token generation limit can produce fewer than three words; the interface reports this. Expanded token details include look-ahead tokens. (Implementation: all 48 steps are generated with EOS suppressed, recording each step's unmodified argmax, then truncated at the first EOS or fourth word. This yields the same tokens as stopping early.)
- **Inputs and token scope:** Any two different sequences, up to 256 tokens each. Source states come from separate natural forwards. **First difference → end** requires equal token counts, compares token IDs position by position, and patches the full suffix beginning at the first mismatch. Matching tokens after that mismatch are included because their hidden states may already reflect the earlier change. If only the final token differs, this reduces to final-token interpolation. There is no word-level matching, padding, or alignment of unequal lengths. **Final token only** pairs the two last-token vectors and permits arbitrary prefixes and lengths.
- **Interpolation start / patch location:** Block output (`resid_post`) of layer N, before final normalization. You can select any block, including the final one, after which only final normalization and the output head remain. The selected token positions are patched together once per forward pass, at this one layer.
- **Interpolation:** Each selected token's A/B hidden-state pair is interpolated independently, using the same t at every position. SLERP interpolates that pair's unit directions and linearly interpolates its L2 norm; the suffix is not flattened into one global SLERP path. Linear interpolation is also available. Nearly parallel directions use a normalized linear direction; nearly opposite directions prompt you to select Linear because the spherical path is ambiguous. Interpolated states are constructed inside the trace, one sample batch at a time.
- **Recorded outputs:** Every block from the patch layer through the final block, plus final logits, for every registered metric (`METRICS` in `plateau/core/math.py`): **Relative L2 (Shinkle & Heimersheim 2025)** `||x−a|| / (||x−a|| + ||x−b||)` (default), **Relative L2 (Janiak et al. 2024)** `||x−a|| / ||a−b||`, and **Relative L2 projected onto A→B** `(x−a)·(b−a) / ||b−a||²`, which ignores the component perpendicular to the straight path and can leave [0, 1]. Here `a`, `b` are the patched endpoint outputs (t = 0, t = 1). Block outputs are recorded **before** final normalization. In section **02 Effect of the interpolation**, the **Effect along the path** figure plots any selection of recorded layers (default: first block after the patch, a middle block, the final block, logits), optionally overlaid with the next-token prediction along t, and lists, per layer, the endpoint L2 `||x_C(0)-x_C(1)||₂` and the **plateau score**: the Δt between the first crossings of d = 0.1 and d = 0.9, linearly interpolated between samples (d = t gives 0.8; lower means a sharper transition). A layer whose two endpoints are identical is reported as undefined; identical logits abort the run.
- **Fixed context and endpoints:** In final-token mode, choose A or B (default A). All preceding token states remain from that context, so a transferred endpoint may differ from its natural output. In suffix mode, every position before the first difference belongs to the identical causal prefix, and all remaining positions are patched. Therefore t=0 reproduces natural A and t=1 reproduces natural B, up to numerical precision; the context control is hidden. Continuation panels always show natural, unpatched A/B predictions.
- **Distance:** `d(t) = ||x_C(t)-x_C(0)||₂ / (||x_C(t)-x_C(0)||₂ + ||x_C(t)-x_C(1)||₂)`. Both reference outputs are computed under the same fixed context and patching intervention as the path, as the first two rows of every path batch. L2 distances use the complete final hidden or logit vector. Identical endpoint outputs give an explicit undefined-distance error; try a later patch location or another pair.
- **Endpoint checks:** Each record reports the max logit difference between the t=0/t=1 path samples and the patched references, and the gap between patched and natural A/B outputs. The latter should be numerically small for both endpoints in suffix mode; in final-token mode it can be substantial when prefixes differ. All patched forwards run without KV caching.
- **Slope:** Estimated from discrete `|Δd/Δt|` and dependent on sampling resolution.
- **Plots:** The dashed `d=t` line is a reference, not another model experiment. Both axes are dimensionless.

Method: [Matthew Shinkle & StefanHex, Activation Plateaus: Where and How They Emerge](https://www.lesswrong.com/posts/WMfSbt7AAcJdHzysB/activation-plateaus-where-and-how-they-emerge), especially footnotes 1–2. This implementation patches Hugging Face module outputs through nnsight; it does not reproduce TransformerLens weight transformations value for value.
