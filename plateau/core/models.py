"""Model catalogs and per-architecture module paths; no ML imports."""
import json

MODELS = {
    "gpt2-large": {"label": "GPT-2 Large", "repo": "openai-community/gpt2-large", "layers": 36, "family": "GPT-2"},
    "gpt2": {"label": "GPT-2 Small", "repo": "openai-community/gpt2", "layers": 12, "family": "GPT-2"},
    "gpt2-medium": {"label": "GPT-2 Medium", "repo": "openai-community/gpt2-medium", "layers": 24, "family": "GPT-2"},
    "gpt2-xl": {"label": "GPT-2 XL", "repo": "openai-community/gpt2-xl", "layers": 48, "family": "GPT-2"},
}
for size, layers in [("70m", 6), ("160m", 12), ("410m", 24), ("1b", 16), ("1.4b", 24), ("2.8b", 32)]:
    MODELS[f"pythia-{size}"] = {"label": f"Pythia {size.upper()}", "repo": f"EleutherAI/pythia-{size}",
                                "layers": layers, "family": "Pythia"}
for model_id, repo_name, label, family, layers in [
    ("qwen2.5-0.5b", "Qwen2.5-0.5B", "Qwen2.5 0.5B Base", "Qwen2.5", 24),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B", "Qwen2.5 1.5B Base", "Qwen2.5", 28),
    ("qwen2.5-3b", "Qwen2.5-3B", "Qwen2.5 3B Base", "Qwen2.5", 36),
    ("qwen3-0.6b-base", "Qwen3-0.6B-Base", "Qwen3 0.6B Base", "Qwen3", 28),
    ("qwen3-1.7b-base", "Qwen3-1.7B-Base", "Qwen3 1.7B Base", "Qwen3", 28),
]:
    MODELS[model_id] = {"label": label, "repo": f"Qwen/{repo_name}", "family": family, "layers": layers}

# Attribute paths of the transformer blocks and the output head, by config.model_type.
# Block outputs are resid_post, before final normalization.
ARCHITECTURES = {
    "gpt2": {"blocks": "transformer.h", "head": "lm_head", "family": "GPT-2"},
    "gptj": {"blocks": "transformer.h", "head": "lm_head", "family": "GPT-J"},
    "gpt_neox": {"blocks": "gpt_neox.layers", "head": "embed_out", "family": "Pythia"},
    "qwen2": {"blocks": "model.layers", "head": "lm_head", "family": "Qwen"},
    "qwen3": {"blocks": "model.layers", "head": "lm_head", "family": "Qwen"},
    "llama": {"blocks": "model.layers", "head": "lm_head", "family": "Llama"},
    "gemma2": {"blocks": "model.layers", "head": "lm_head", "family": "Gemma"},
    "olmo3": {"blocks": "model.layers", "head": "lm_head", "family": "OLMo"},
}


def model_catalog(models=MODELS):
    return [{"id": model_id, **{k: v for k, v in info.items() if k != "repo"}} for model_id, info in models.items()]


def ndif_models(status):
    """Catalog of the language models NDIF currently serves, from its /status response.

    Mirrors nnsight's `ndif.status()`: deployments that are hot, warm, or scheduled.
    Models whose architecture has no entry in ARCHITECTURES are skipped. Ids are repo ids.
    """
    models = {}
    for deployment in status.get("deployments", {}).values():
        if deployment.get("deployment_level") not in ("HOT", "WARM") and "schedule" not in deployment:
            continue
        if not deployment.get("model_key", "").startswith("nnsight.modeling.language.LanguageModel:"):
            continue
        repo, config = deployment.get("repo_id"), json.loads(deployment.get("config") or "{}")
        arch = ARCHITECTURES.get(config.get("model_type"))
        layers = config.get("num_hidden_layers") or config.get("n_layer")
        if not repo or arch is None or not layers:
            continue
        running = deployment.get("application_state") == "RUNNING"
        if repo in models and (models[repo]["running"] or not running):
            continue  # several replicas; keep a running one
        models[repo] = {"label": repo.split("/")[-1], "repo": repo, "layers": layers,
                        "family": arch["family"], "running": running, "n_params": deployment.get("n_params")}
    return dict(sorted(models.items(), key=lambda item: (item[1]["family"], item[1]["n_params"] or 0)))
