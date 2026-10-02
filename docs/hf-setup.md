# Hugging Face Space setup

[← Hosting](HOSTING.md)

The hosted Explorer runs on a Hugging Face Space. Inference runs on NDIF, so the Space needs no GPU. `.github/workflows/space.yml` uploads the app to the Space on every relevant push to `main`.

## 1. Create the Space

On huggingface.co: **New → Space**.

- **SDK:** Gradio. The workflow writes the Space's `README.md` metadata itself, so the template choice does not matter.
- **Hardware:** CPU basic (free) is enough. ZeroGPU also works: `deploy/hf-space/app.py` contains the GPU stub ZeroGPU requires, and the workflow pins torch 2.13.0 for it.
- **Visibility:** public, unless only logged-in members of your account or organization should open it.

Leave the Space empty. Every deploy replaces all of its files.

## 2. Connect the GitHub repository

Create a Hugging Face token with **write** access to the Space (Settings → Access Tokens; a fine-grained token scoped to that Space is enough). Then, in the GitHub repository under **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
| --- | --- | --- |
| Variable | `HF_SPACE` | Space ID, e.g. `your-name/plateau-lab` |
| Secret | `HF_TOKEN` | the write token |

The deploy job is skipped while `HF_SPACE` is unset. To deploy immediately, run the **Deploy** workflow manually from the Actions tab.

## 3. Space secrets (optional)

Set these in the Space under **Settings → Variables and secrets**:

| Secret | Purpose |
| --- | --- |
| `PLATEAU_SHARED_NDIF_KEY` + `PLATEAU_ACCESS_CODE` | Lab members enter the access code instead of their own NDIF key; runs use the shared key. |
| `HF_TOKEN` | Read token for tokenizers of gated models (e.g. Llama). Accept each model's license with that account first. |
| `PLATEAU_WORKERS`, `PLATEAU_BATCH_SIZE` | Concurrent NDIF jobs (default 4) and path samples per trace (default 32). |

Do **not** set `NDIF_API_KEY`: visitors without a key would run on it.

## 4. Check the deployment

After the workflow's upload, the Space builds for about two minutes. Then:

- `https://<owner>-<space>.hf.space/` shows the Explorer.
- `/api/config` reports `"remote": true` and lists the models currently running on NDIF.
- One experiment with an NDIF key (or the access code) completes.

If the Space shows **Build error**, open its **Logs → Build** tab. The pip install step only sees the generated `requirements.txt`, so it must not include other files with `-r`.
