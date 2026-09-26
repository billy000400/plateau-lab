"""FastAPI backend. Run: uvicorn server.app:app, then open http://127.0.0.1:8000.

Experiments run one at a time on a background worker; the frontend polls
/api/jobs/{id}. Configuration comes from the environment:

- NDIF_API_KEY: NDIF key (read by nnsight). Enables remote execution by default.
  In remote mode the model list mirrors NDIF's current deployments (/status).
  Gated repos (Llama, Gemma) also need HF_TOKEN for the tokenizer download.
- PLATEAU_REMOTE: "1"/"0" to force remote NDIF or local execution.
- PLATEAU_DEVICE: local device (default: mps, then cuda, then cpu).
- PLATEAU_BATCH_SIZE: path samples per request (default 32).
- PLATEAU_ALLOWED_ORIGINS: comma-separated CORS origins (default "*").
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from plateau.core.models import MODELS, model_catalog, ndif_models

REMOTE = os.environ.get("PLATEAU_REMOTE", "1" if os.environ.get("NDIF_API_KEY") else "0") == "1"
BATCH_SIZE = int(os.environ.get("PLATEAU_BATCH_SIZE", "32"))
MAX_JOBS = 100
NDIF_STATUS_URL = "https://api.ndif.us/status"
CATALOG_TTL = 60  # seconds between NDIF status fetches
WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Plateau Lab")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("PLATEAU_ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
WORKER = ThreadPoolExecutor(max_workers=1)
BACKEND = None
CATALOG = {"models": None, "fetched": 0.0}


class RunRequest(BaseModel):
    model: str | None = None  # default: first model of the catalog
    sequence_a: str = Field(max_length=4000)
    sequence_b: str = Field(max_length=4000)
    patch_layer: int = Field(0, ge=0)
    patch_position: Literal["last_token", "different_suffix"] = "last_token"
    interpolation: Literal["slerp", "linear"] = "slerp"
    context: Literal["a", "b"] = "a"
    steps: int = Field(41, ge=5, le=201)


class TokenizeRequest(BaseModel):
    model: str | None = None
    sequence_a: str = Field("", max_length=4000)
    sequence_b: str = Field("", max_length=4000)
    patch_position: Literal["last_token", "different_suffix"] = "last_token"


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    return JSONResponse({"error": f"Invalid {'.'.join(map(str, first['loc'][1:]))}: {first['msg']}"}, status_code=400)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def backend():
    global BACKEND
    if BACKEND is None:
        # Imported lazily so the config endpoint answers before torch is loaded.
        from plateau.backends.nnsight_backend import NnsightBackend
        BACKEND = NnsightBackend(remote=REMOTE, device=os.environ.get("PLATEAU_DEVICE"))
    return BACKEND


def catalog():
    """Models offered to the client: NDIF's current deployments in remote mode, else the local list."""
    if not REMOTE:
        return MODELS
    with LOCK:
        if CATALOG["models"] is not None and time.time() - CATALOG["fetched"] < CATALOG_TTL:
            return CATALOG["models"]
    try:
        with urllib.request.urlopen(NDIF_STATUS_URL, timeout=15) as response:
            models = ndif_models(json.load(response))
    except Exception as exc:
        if CATALOG["models"] is not None:
            return CATALOG["models"]  # keep serving the last known list while NDIF is unreachable
        raise HTTPException(503, f"Cannot load the NDIF model list: {exc}")
    # Running models first, so the default (first entry) is one that answers immediately.
    models = dict(sorted(models.items(), key=lambda item: not item[1]["running"]))
    with LOCK:
        CATALOG.update(models=models, fetched=time.time())
    return models


@lru_cache(maxsize=8)
def tokenizer(repo):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo)


def update(job_id, **values):
    with LOCK:
        JOBS[job_id].update(values)


def run_job(job_id, request, models):
    from plateau.core.experiment import run_experiment
    with LOCK:
        if JOBS[job_id].get("cancelled"):
            return
        JOBS[job_id].update(status="running", message="Preparing the model…", progress=None)
    try:
        result = run_experiment(backend(), request, BATCH_SIZE,
                                progress=lambda message, fraction: update(job_id, message=message, progress=fraction),
                                cancelled=lambda: JOBS[job_id].get("cancelled", False), models=models)
        result["id"] = job_id
        update(job_id, status="done", progress=1, message="Complete.", result=result)
    except InterruptedError as exc:
        update(job_id, status="cancelled", message=str(exc))
    except ValueError as exc:  # invalid input or undefined curve
        update(job_id, status="error", message=str(exc))
    except Exception as exc:
        traceback.print_exc()
        update(job_id, status="error", message=str(exc))


@app.get("/api/config")
def config():
    models = catalog()
    default = "openai-community/gpt2" if REMOTE else "gpt2"
    return {"models": model_catalog(models), "remote": REMOTE,
            "default_model": default if default in models else next(iter(models), None)}


@app.post("/api/tokenize")
def tokenize(request: TokenizeRequest):
    """Tokenize both inputs with the selected model's tokenizer (cheap; used for the live preview)."""
    from plateau.core.experiment import tokenize_preview
    models = catalog()
    model_id = request.model or next(iter(models))
    if model_id not in models:
        raise HTTPException(400, "This model is not available right now. Reload the page to update the model list.")
    return {"model": model_id, **tokenize_preview(tokenizer(models[model_id]["repo"]), request.model_dump())}


@app.post("/api/run", status_code=202)
def run(request: RunRequest):
    models = catalog()
    if request.model is not None and request.model not in models:
        raise HTTPException(400, "This model is not available right now. Reload the page to update the model list.")
    job_id = uuid.uuid4().hex
    with LOCK:
        finished = [key for key, job in JOBS.items() if job["status"] in ("done", "error", "cancelled")]
        for key in sorted(finished, key=lambda key: JOBS[key]["created"])[:max(0, len(JOBS) - MAX_JOBS + 1)]:
            del JOBS[key]
        queued = sum(job["status"] in ("queued", "running") for job in JOBS.values())
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": None, "created": time.time(),
                        "message": f"Waiting for {queued} earlier experiment{'s' if queued != 1 else ''}…" if queued else "Starting…"}
    WORKER.submit(run_job, job_id, request.model_dump(), models)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found.")
        return dict(JOBS[job_id])


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    with LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "Job not found.")
        if JOBS[job_id]["status"] == "queued":
            JOBS[job_id].update(cancelled=True, status="cancelled", message="Stopped.")
        elif JOBS[job_id]["status"] == "running":
            JOBS[job_id].update(cancelled=True, message="Stopping after the current request…")
    return {"ok": True}


# Local development convenience: serve the frontend from the same origin.
if WEB.is_dir():
    class RevalidatedStaticFiles(StaticFiles):
        """Local dev: make browsers revalidate so edited web/ files show up on reload."""
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/", RevalidatedStaticFiles(directory=WEB, html=True), name="web")
