"""FastAPI backend. Run: uvicorn server.app:app, then open http://127.0.0.1:8000.

Experiments run on a small pool of background workers; the frontend polls
/api/jobs/{id}. In remote mode each user sends their own NDIF key in the
X-NDIF-Key header of /api/run; it is used for that job only and never stored.
Configuration comes from the environment:

- NDIF_API_KEY: fallback NDIF key for requests without one (local dev). If unset
  in remote mode, users must bring their own key.
- PLATEAU_REMOTE: "1"/"0" to force remote NDIF or local execution (default:
  remote if NDIF_API_KEY is set).
  In remote mode the model list mirrors NDIF's current deployments (/status).
  Gated repos (Llama, Gemma) also need HF_TOKEN for the tokenizer download.
- PLATEAU_WORKERS: concurrent experiments (default 4 remote, 1 local).
- PLATEAU_SHARED_NDIF_KEY + PLATEAU_ACCESS_CODE: lets people without their own key
  run on a shared key by entering the access code (X-Access-Code header). The key
  never leaves the server. Deliberately not named NDIF_API_KEY: nnsight falls back
  to that variable for every keyless request, which would bypass the access code.
- PLATEAU_DEVICE: local hardware selection (auto by default; see hardware.py).
- PLATEAU_BATCH_SIZE: remote samples per request (default 32); local hardware policy otherwise.
- PLATEAU_ALLOWED_ORIGINS: comma-separated CORS origins (default "*").
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import urllib.request
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from plateau.core.models import MODELS, model_catalog, ndif_models
from server.storage import Library, re_full_id

REMOTE = os.environ.get("PLATEAU_REMOTE", "1" if os.environ.get("NDIF_API_KEY") else "0") == "1"
BATCH_SIZE = int(os.environ.get("PLATEAU_BATCH_SIZE", "32").replace("auto", "32"))
if BATCH_SIZE < 1:
    raise ValueError("PLATEAU_BATCH_SIZE must be positive.")
MAX_JOBS = 100
NDIF_STATUS_URL = "https://api.ndif.us/status"
CATALOG_TTL = 60  # seconds between NDIF status fetches
WEB = Path(__file__).resolve().parent.parent / "web"
ROOT = WEB.parent
LIBRARY = Library(os.environ.get("PLATEAU_DATA_DIR", ROOT / "data"))
LOCAL_ENGINE = None
LOCAL_HARDWARE = None

app = FastAPI(title="Plateau Lab")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("PLATEAU_ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-NDIF-Key", "X-Access-Code"])

LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
# Remote jobs only wait on NDIF, so several can run at once; local jobs share one device.
WORKERS = int(os.environ.get("PLATEAU_WORKERS", "4")) if REMOTE else 1
WORKER = ThreadPoolExecutor(max_workers=WORKERS)
SERVER_KEY = bool(os.environ.get("NDIF_API_KEY"))
SHARED_KEY = os.environ.get("PLATEAU_SHARED_NDIF_KEY", "").strip()
ACCESS_CODE = os.environ.get("PLATEAU_ACCESS_CODE", "").strip()
SHARED_ACCESS = bool(REMOTE and SHARED_KEY and ACCESS_CODE)
NDIF_KEY_FORMAT = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CATALOG = {"models": None, "fetched": 0.0}


class RunRequest(BaseModel):
    model: str | None = None  # default: first model of the catalog
    sequence_a: str = Field(max_length=4000)
    sequence_b: str = Field(max_length=4000)
    patch_layer: int = Field(0, ge=-1)
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


def backend(api_key=None):
    # Imported lazily so the config endpoint answers before torch is loaded.
    from plateau.backends.nnsight_backend import NnsightBackend
    return NnsightBackend(remote=REMOTE, device=os.environ.get("PLATEAU_DEVICE"), api_key=api_key)


@app.middleware("http")
async def local_origin(request: Request, call_next):
    if not REMOTE and request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"error": "Use the local Plateau Lab page for this action."}, status_code=403)
    return await call_next(request)


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
    if not REMOTE:
        from models import MODELS, cached_snapshot
        model_id = next((key for key, info in MODELS.items() if info["repo"] == repo), None)
        snapshot = cached_snapshot(model_id) if model_id else None
        if snapshot:
            return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    return AutoTokenizer.from_pretrained(repo)


def update(job_id, **values):
    with LOCK:
        JOBS[job_id].update(values)


def run_job(job_id, request, models, api_key):
    global LOCAL_ENGINE
    with LOCK:
        if job_id not in JOBS or JOBS[job_id].get("cancelled"):
            return
        JOBS[job_id].update(status="running", message="Preparing the model…", progress=None)
    try:
        progress = lambda message, fraction: update(job_id, message=message, progress=fraction)
        cancelled = lambda: JOBS[job_id].get("cancelled", False)
        if REMOTE:
            from plateau.core.experiment import run_experiment
            result = run_experiment(backend(api_key), request, BATCH_SIZE, progress, cancelled, models=models)
        else:
            from engine import Engine
            if LOCAL_ENGINE is None:
                LOCAL_ENGINE = Engine()
            result = LOCAL_ENGINE.run(request, progress, cancelled, all_layers=True)
        result["id"] = job_id
        with LOCK:
            if cancelled():
                raise InterruptedError("Stopped. This experiment was not saved.")
            if not REMOTE:
                LIBRARY.write("runs", result)
            JOBS[job_id].update(status="done", progress=1, message="Complete.", result=result)
    except InterruptedError as exc:
        update(job_id, status="cancelled", message=str(exc))
    except ValueError as exc:  # invalid input or undefined curve
        update(job_id, status="error", message=redact(str(exc), api_key))
    except Exception as exc:
        # Do not print raw exception tracebacks: upstream errors may embed a key.
        update(job_id, status="error", message=redact(str(exc), api_key))


def redact(message, api_key):
    # nnsight echoes malformed keys in its error messages; never expose a key in job status.
    for secret in (api_key, SHARED_KEY, os.environ.get("NDIF_API_KEY")):
        if secret:
            message = message.replace(secret, "•••")
    return message


@app.get("/api/config")
def config():
    models = catalog()
    default = "openai-community/gpt2" if REMOTE else "gpt2"
    extra = {}
    if not REMOTE:
        from models import CACHE, model_catalog as local_catalog
        with LOCK:
            active_job = next((key for key, value in JOBS.items() if value["status"] in ("queued", "running")), None)
        extra = {"models": local_catalog(), "data_path": str(LIBRARY.root), "model_cache_path": str(CACHE),
                 "active_job": active_job}
    return {"models": model_catalog(models), "remote": REMOTE, "requires_key": REMOTE and not SERVER_KEY, "shared_access": SHARED_ACCESS,
            "local_library": not REMOTE, "default_model": default if default in models else next(iter(models), None), **extra}


@app.post("/api/tokenize")
def tokenize(request: TokenizeRequest):
    """Tokenize both inputs with the selected model's tokenizer (cheap; used for the live preview)."""
    from plateau.core.experiment import tokenize_preview
    models = catalog()
    model_id = request.model or next(iter(models))
    if model_id not in models:
        raise HTTPException(400, "This model is not available right now. Reload the page to update the model list.")
    tok = tokenizer(models[model_id]["repo"])
    if not REMOTE:
        # Match Engine.validate exactly; supported local models do not require BOS.
        class LocalTokenizer:
            def encode(self, text, **kwargs):
                return tok.encode(text, add_special_tokens=False)
            def decode(self, ids):
                return tok.decode(ids)
        tok = LocalTokenizer()
    return {"model": model_id, **tokenize_preview(tok, request.model_dump())}


@app.post("/api/run", status_code=202)
def run(request: RunRequest, x_ndif_key: str | None = Header(None, max_length=200),
        x_access_code: str | None = Header(None, max_length=200)):
    api_key = (x_ndif_key or "").strip() or None
    code = (x_access_code or "").strip()
    if api_key and not NDIF_KEY_FORMAT.fullmatch(api_key):
        raise HTTPException(400, "That is not an NDIF API key (format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)."
                                 + (" To use the lab key, enter the lab access code instead." if SHARED_ACCESS else ""))
    if not api_key and code and SHARED_ACCESS:
        if not hmac.compare_digest(code.encode(), ACCESS_CODE.encode()):
            time.sleep(1)  # slow down guessing
            raise HTTPException(403, "Wrong lab access code.")
        api_key = SHARED_KEY
    if REMOTE and not api_key and not SERVER_KEY:
        raise HTTPException(401, "Add your NDIF API key (or the lab access code) in Settings to run experiments.")
    models = catalog()
    if REMOTE and request.patch_layer < 0:
        raise HTTPException(400, "The NDIF backend supports block outputs; choose layer 0 or later.")
    if not models:
        raise HTTPException(503, "No models are available right now. Try again when a deployment is running.")
    if request.model is not None and request.model not in models:
        raise HTTPException(400, "This model is not available right now. Reload the page to update the model list.")
    job_id = uuid.uuid4().hex
    with LOCK:
        if not REMOTE and any(job["status"] in ("queued", "running") for job in JOBS.values()):
            raise HTTPException(409, "Another experiment is running. Wait for it to finish or stop it first.")
        finished = [key for key, job in JOBS.items() if job["status"] in ("done", "error", "cancelled")]
        for key in sorted(finished, key=lambda key: JOBS[key]["created"])[:max(0, len(JOBS) - MAX_JOBS + 1)]:
            del JOBS[key]
        ahead = max(0, sum(job["status"] in ("queued", "running") for job in JOBS.values()) - WORKERS + 1)
        payload = request.model_dump(exclude_none=True)
        default = "openai-community/gpt2" if REMOTE else "gpt2"
        payload.setdefault("model", default if default in models else next(iter(models)))
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": None, "created": time.time(), "request": payload,
                        "message": f"Waiting for {ahead} earlier experiment{'s' if ahead != 1 else ''}…" if ahead else "Starting…"}
    # The key travels only with the submitted task; it is not kept in JOBS.
    WORKER.submit(run_job, job_id, payload, models, api_key)
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


def require_local():
    if REMOTE:
        raise HTTPException(404, "This collection is only available on a local server.")


@app.get("/api/hardware")
def hardware():
    global LOCAL_HARDWARE
    require_local()
    from hardware import Hardware
    if LOCAL_ENGINE is not None:
        return LOCAL_ENGINE.hardware.describe()
    if LOCAL_HARDWARE is None:
        LOCAL_HARDWARE = Hardware()
    return LOCAL_HARDWARE.describe()


@app.get("/api/library")
def library():
    require_local()
    with LIBRARY.lock:
        return {"history": LIBRARY.records("runs"), "examples": LIBRARY.records("examples")}


@app.post("/api/examples")
def save_example(data: dict):
    require_local()
    try:
        record = LIBRARY.example(data.get("id"), data.get("tag", "Unclassified"), data.get("notes", ""))
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from None
    return {"ok": True, "id": record["id"]}


def export_response(format, scope, ids=None):
    require_local()
    body, status, content_type, download = LIBRARY.export(format, scope, ids)
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{download}"'
    return Response(body, status_code=status, media_type=content_type, headers=headers)


@app.get("/api/export")
def export_all(format: str = "jsonl", scope: str = "examples"):
    return export_response(format, scope)


@app.post("/api/export")
def export_selected(data: dict):
    return export_response(data.get("format", "jsonl"), data.get("scope", "examples"), data.get("ids") or [])


@app.post("/api/history/delete")
def delete_history(data: dict):
    require_local()
    ids = data.get("ids")
    if not isinstance(ids, list) or not ids or any(not re_full_id(key) for key in ids):
        raise HTTPException(400, "Select valid history records to delete.")
    with LIBRARY.lock:
        for key in ids:
            (LIBRARY.root / "runs" / (key + ".json")).unlink(missing_ok=True)
    return {"ok": True}


# Serve the frontend from the same origin (local dev and the Docker image).
if WEB.is_dir():
    class RevalidatedStaticFiles(StaticFiles):
        """Local dev: make browsers revalidate so edited web/ files show up on reload."""
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "no-cache"
            return response

    @app.get("/classic", include_in_schema=False)
    @app.get("/classic/{path:path}", include_in_schema=False)
    def classic_bookmark(request: Request, path: str = ""):
        return RedirectResponse(request.scope.get("root_path", "") + "/" + path.lstrip("/"), status_code=307)

    app.mount("/", RevalidatedStaticFiles(directory=WEB, html=True), name="web")
