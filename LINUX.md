# Running Plateau Lab on Linux and GPUs

The same application supports Apple GPU acceleration, NVIDIA CUDA, AMD ROCm through PyTorch's CUDA API, and CPU. No code changes are needed when moving machines. The current model menu and saved examples work on each backend.

## Install on a new Linux machine

Use Python 3.10–3.12 (3.12 is tested here). Copy the app without its Mac `.venv`; Python environments are specific to the operating system. You may copy `.model-cache` to reuse downloaded weights, and `data` to bring your examples. Preserve the cache's symbolic links when copying it.

Install a GPU driver and a compatible PyTorch build. Hardware detection cannot add GPU support to a CPU-only PyTorch installation or install drivers. The project pins PyTorch 2.8.0 and Transformers 4.51.3.

For example, on an NVIDIA Linux machine compatible with the CUDA 12.8 build:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python hardware.py
./start.sh
```

PyTorch 2.8 also offers CUDA 12.6 (`cu126`), CUDA 12.9 (`cu129`), ROCm 6.4 (`rocm6.4`, supported AMD Linux machines), and CPU (`cpu`) wheels. Choose the build supported by the GPU and driver using the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v280). The app only needs `torch`; `torchvision` and `torchaudio` are not required.

For a fresh directory without `.venv`, the launcher can install the selected wheel as well:

```sh
PLATEAU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 ./start.sh
```

That setting is used only when creating an environment. To change an existing installation, install the appropriate PyTorch wheel in that environment directly.

## Automatic behavior

- Probes visible GPUs with a small float32 computation. Among working CUDA/ROCm GPUs, chooses the one with the most free memory. Otherwise selects Apple MPS when available, then CPU. Selection is rechecked when a different model is loaded.
- Uses one device per experiment. It does not split a model across GPUs. `CUDA_VISIBLE_DEVICES` and the equivalent visibility controls of the installed ROCm runtime are respected by PyTorch.
- Chooses a CUDA/ROCm path batch from available memory, input length, and model dimensions, up to 32 by default. If curve computation runs out of memory, halves the batch and restarts the measurement, discarding partial samples. Endpoint references and the last padded batch use consistent shapes.
- Keeps the tested MPS policy: batch 1 for Pythia/Qwen, up to 4 for GPT-2. CPU uses up to 4 examples per path batch and up to 8 available processor threads.
- If a model's weights cannot fit on the selected GPU, automatic mode can load it on CPU and reports the fallback. If even a single-example forward cannot fit, use a smaller model or explicitly select CPU.
- Uses a KV cache only for natural three-word continuations: it processes the prefix once, then one new token per step. All activation-patching forwards recompute the full selected context with KV caching disabled.
- Keeps float32 weights, eager attention, and float32 distances. CUDA TF32 is disabled. No automatic quantization, mixed precision, compilation, or Flash Attention is applied to the measured curves. Floating-point results can still differ slightly across hardware.
- Shows the selected device in the interface and saves device name, backend, precision, actual batch size, and cache usage with each new result. Existing examples are preserved.

## Overrides and diagnostics

Normally no overrides are needed:

```sh
.venv/bin/python hardware.py
PLATEAU_DEVICE=cpu ./start.sh
PLATEAU_DEVICE=cuda:1 ./start.sh
PLATEAU_BATCH_SIZE=8 ./start.sh
```

`PLATEAU_DEVICE` accepts `auto` (default), `cpu`, `mps`, `cuda` (visible GPU 0), or `cuda:N`. An unavailable explicitly requested device gives a clear error. Explicit selection disables model-loading fallback to CPU. `PLATEAU_BATCH_SIZE` accepts `auto` or 1–64; the MPS batch-1 policy still applies to Pythia/Qwen, and memory failures can still reduce a requested batch.

## Open a remote Linux server from your Mac

From the Plateau Lab folder on your Mac, a single command copies the code, starts the remote server and opens an SSH tunnel:

```sh
./start-remote.sh your-user@your-server
```

Then open [remote Plateau Lab](http://127.0.0.1:8766) in your Mac's browser once the server reports it is ready. Keep that Terminal window open. SSH-config aliases work too, including configured keys, ports, and jump hosts.

The destination is `~/plateau-lab` on the server. It needs Python 3.10–3.12 with `venv`, `rsync`, and internet access for the first package/model downloads. The helper copies source, documentation and web assets. Your server's virtual environment, downloaded models and saved examples persist between launches. Mac models and examples are not uploaded automatically. Stop an earlier remote app session with Ctrl+C before relaunching to apply code updates; the helper does not terminate existing processes.

For a new NVIDIA environment using a compatible CUDA 12.8 driver/build, you can select that PyTorch wheel during first setup:

```sh
./start-remote.sh your-user@your-server cu128
```

The optional wheel choices are `auto`, `cu126`, `cu128`, `cu129`, `rocm6.4`, and `cpu`; they affect first environment creation only. Use the setup guidance above to match the server's hardware. If local port 8766 is occupied, choose another with `PLATEAU_LOCAL_PORT=8767 ./start-remote.sh your-user@your-server`.

For a server that is already running, you can connect manually instead:

Run `./start.sh` on the server. It listens on `127.0.0.1:8765`. From your Mac, forward another local port (8766 avoids colliding with your Mac's local Plateau Lab):

```sh
ssh -L 8766:127.0.0.1:8765 your-user@your-server
```

Then open [Plateau Lab through the tunnel](http://127.0.0.1:8766). Inference and model storage are on the Linux server. This is personal remote access; public hosting and per-user accounts are separate work.

## Validation on the destination GPU

```sh
.venv/bin/python check_hardware.py
.venv/bin/python prepare_model.py pythia-160m
.venv/bin/python check_cache.py --model pythia-160m
.venv/bin/python check_inference.py --model pythia-160m --oom
.venv/bin/python check_contexts.py --model pythia-160m
```

The selection-policy tests simulate CUDA/ROCm APIs and do not prove GPU execution. The other checks load a real checkpoint on the automatically selected device, verify cached versus uncached greedy tokens, test memory-retry cleanup, and check curve endpoints and context symmetry. They do not add records to your collections. Run `hardware.py` first to confirm the expected GPU is selected.
