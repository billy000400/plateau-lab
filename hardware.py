"""Device selection and conservative float32 inference settings for PyTorch 2.8."""
from __future__ import annotations

import json
import os
import platform
import re

import torch


def is_out_of_memory(error):
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower())


def cpu_threads():
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, min(8, count))


def probe_device(device):
    # Availability flags alone do not catch an incompatible driver or GPU binary.
    with torch.inference_mode():
        sample = torch.ones((2, 2), device=device, dtype=torch.float32)
        if (sample @ sample).sum().item() != 8:
            raise RuntimeError("The device failed a small computation check.")


class Hardware:
    def __init__(self, environ=None):
        env = os.environ if environ is None else environ
        self.requested = env.get("PLATEAU_DEVICE", "auto").strip().lower() or "auto"
        if not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", self.requested):
            raise ValueError("PLATEAU_DEVICE must be auto, cpu, mps, cuda, or cuda:N.")
        batch = env.get("PLATEAU_BATCH_SIZE", "auto").strip().lower() or "auto"
        try:
            self.batch_override = None if batch == "auto" else int(batch)
            if self.batch_override is not None and not 1 <= self.batch_override <= 64:
                raise ValueError()
        except ValueError:
            raise ValueError("PLATEAU_BATCH_SIZE must be auto or an integer from 1 to 64.") from None
        self.notes = []
        self.candidates = []
        self.device = "cpu"
        self.name = "CPU"
        self.backend = "CPU"
        self.threads = cpu_threads()
        self._select()
        # Keep the same precision policy across machines; no implicit TF32/FP16.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(self.threads)

    def _select(self):
        if self.requested == "cpu":
            return
        wants_cuda = self.requested == "auto" or self.requested.startswith("cuda")
        if wants_cuda:
            try:
                if torch.cuda.is_available():
                    indices = range(torch.cuda.device_count()) if self.requested == "auto" else [
                        int(self.requested.split(":")[1]) if ":" in self.requested else 0]
                    for index in indices:
                        device = f"cuda:{index}"
                        try:
                            probe_device(device)
                            free, total = torch.cuda.mem_get_info(index)
                            self.candidates.append({"device": device, "name": torch.cuda.get_device_name(index),
                                                    "free_bytes": free, "total_bytes": total})
                        except (RuntimeError, AssertionError) as error:
                            self.notes.append(f"Could not use {device}: {error}")
            except (RuntimeError, AssertionError) as error:
                self.notes.append(f"GPU discovery failed: {error}")
            if self.candidates:
                best = max(self.candidates, key=lambda item: (item['free_bytes'], item['total_bytes']))
                self.device, self.name = best['device'], best['name']
                self.backend = "ROCm" if torch.version.hip else "CUDA"
                return
            if self.requested.startswith("cuda"):
                raise ValueError("The requested GPU is unavailable to PyTorch. Check the GPU driver, PyTorch GPU build, and CUDA_VISIBLE_DEVICES.")
        if self.requested in ("auto", "mps"):
            try:
                if torch.backends.mps.is_available():
                    probe_device("mps")
                    self.device, self.name, self.backend = "mps", "Apple GPU", "MPS"
                    return
            except RuntimeError as error:
                self.notes.append(f"Could not use the Apple GPU: {error}")
            if self.requested == "mps":
                raise ValueError("Apple GPU acceleration is unavailable. Use auto or cpu on this machine.")
        self.notes.append("No usable GPU is exposed by this PyTorch installation; using CPU.")

    def describe(self):
        return {"device": self.device, "name": self.name, "backend": self.backend,
                "selection": self.requested, "platform": platform.system(), "dtype": "float32",
                "attention": "eager", "tf32": False, "cpu_threads": self.threads,
                "batch_setting": self.batch_override or "auto", "notes": list(self.notes),
                "available_gpus": list(self.candidates), "torch_version": torch.__version__}

    def clear_cache(self):
        if self.device.startswith("cuda"):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def use_cpu(self, reason):
        if self.requested != "auto":
            raise RuntimeError(reason + " Choose a smaller model or set PLATEAU_DEVICE=auto to allow CPU fallback.")
        self.clear_cache()
        self.device, self.name, self.backend = "cpu", "CPU", "CPU"
        self.notes.append(reason + " Using CPU for this model.")

    def batch_size(self, config, sequence_length, steps):
        if self.device == "mps" and config.model_type in ("gpt_neox", "qwen2", "qwen3"):
            return 1  # Keep the numerically validated MPS policy for these models.
        cap = self.batch_override or (32 if self.device.startswith("cuda") else 4)
        if self.device.startswith("cuda") and self.batch_override is None:
            free, _ = torch.cuda.mem_get_info(torch.device(self.device))
            hidden, heads = config.hidden_size, config.num_attention_heads
            # Includes eager attention matrices, intermediates and the full logits tensor.
            per_example = 8 * (sequence_length * config.vocab_size + 16 * sequence_length * hidden
                               + 4 * heads * sequence_length ** 2)
            cap = min(cap, max(1, int(free * 0.65) // max(32 * 1024 ** 2, per_example)))
        cap = max(1, min(cap, steps))
        return cap if self.batch_override else 1 << (cap.bit_length() - 1)


if __name__ == "__main__":
    print(json.dumps(Hardware().describe(), indent=2))
