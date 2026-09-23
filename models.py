"""Shared model catalog and disk-cache inspection; no ML imports or network calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".model-cache"
MODELS = {
    "gpt2-large": {"label": "GPT-2 Large", "repo": "openai-community/gpt2-large", "layers": 36, "family": "GPT-2"},
    "gpt2": {"label": "GPT-2 Small", "repo": "openai-community/gpt2", "layers": 12, "family": "GPT-2"},
    "gpt2-medium": {"label": "GPT-2 Medium", "repo": "openai-community/gpt2-medium", "layers": 24, "family": "GPT-2"},
    "gpt2-xl": {"label": "GPT-2 XL", "repo": "openai-community/gpt2-xl", "layers": 48, "family": "GPT-2"},
}
for size, layers, download_gb in [
    ("70m", 6, 0.17), ("160m", 12, 0.38), ("410m", 24, 0.91),
    ("1b", 16, 2.09), ("1.4b", 24, 2.93), ("2.8b", 32, 5.68),
]:
    MODELS[f"pythia-{size}"] = {
        "label": f"Pythia {size.upper()}", "repo": f"EleutherAI/pythia-{size}",
        "layers": layers, "family": "Pythia", "download_gb": download_gb,
    }

for model_id, repo_name, label, family, layers, download_gb in [
    ("qwen2.5-0.5b", "Qwen2.5-0.5B", "Qwen2.5 0.5B Base", "Qwen2.5", 24, 0.99),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B", "Qwen2.5 1.5B Base", "Qwen2.5", 28, 3.09),
    ("qwen2.5-3b", "Qwen2.5-3B", "Qwen2.5 3B Base", "Qwen2.5", 36, 6.17),
    ("qwen3-0.6b-base", "Qwen3-0.6B-Base", "Qwen3 0.6B Base", "Qwen3", 28, 1.19),
    ("qwen3-1.7b-base", "Qwen3-1.7B-Base", "Qwen3 1.7B Base", "Qwen3", 28, 3.44),
]:
    MODELS[model_id] = {
        "label": label, "repo": f"Qwen/{repo_name}", "family": family,
        "layers": layers, "download_gb": download_gb,
    }
MODELS['pythia-2.8b']['memory_note'] = 'Uses about 11 GB for model weights in memory, plus working space.'
MODELS['qwen2.5-3b']['memory_note'] = 'Uses about 12 GB for model weights in memory, plus working space.'

DOWNLOAD_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
]


def cached_snapshot(model_id, cache=CACHE):
    """Only report ready when required files and all weight shards are present."""
    folder = Path(cache) / ("models--" + MODELS[model_id]["repo"].replace("/", "--"))
    try:
        revision = (folder / "refs" / "main").read_text().strip()
        snapshot = folder / "snapshots" / revision
        if not all((snapshot / name).is_file() for name in
                   ("config.json", "tokenizer.json", "tokenizer_config.json")):
            return None
        index = snapshot / "model.safetensors.index.json"
        weights = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        if weights and all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in weights):
            return snapshot
    except (OSError, ValueError, KeyError):
        pass
    return None


def model_catalog():
    return [{"id": model_id, **info, "cached": cached_snapshot(model_id) is not None}
            for model_id, info in MODELS.items()]
