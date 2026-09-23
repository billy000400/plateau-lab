"""Download one public checkpoint into this application's private cache."""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
from models import CACHE, DOWNLOAD_FILES, MODELS, cached_snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default="gpt2-large", choices=list(MODELS))
args = parser.parse_args()
if cached_snapshot(args.model):
    print(f"{args.model} is already saved on this computer. No download needed.", flush=True)
else:
    print(f"Downloading {args.model} into the app's .model-cache directory…", flush=True)
    snapshot_download(MODELS[args.model]['repo'], cache_dir=str(CACHE),
                      allow_patterns=DOWNLOAD_FILES, max_workers=2)
print("Model ready.", flush=True)
