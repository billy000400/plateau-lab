#!/usr/bin/env python3
"""Local GUI server. Run: python app.py, then open http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from models import CACHE, model_catalog

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
LOCK = threading.RLock()
JOBS = {}
ENGINE = None
ACTIVE = None
HARDWARE_PREVIEW = None
HARDWARE_LOCK = threading.Lock()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def records(folder):
    return [read_json(p) for p in sorted((DATA / folder).glob("*.json"), reverse=True)]


def run_job(job_id, request):
    global ENGINE, ACTIVE
    def progress(message, fraction):
        with LOCK:
            JOBS[job_id].update(message=message, progress=fraction, hardware=ENGINE.hardware.describe())
    try:
        from engine import Engine
        if ENGINE is None:
            ENGINE = Engine()
        result = ENGINE.run(request, progress, lambda: JOBS[job_id].get("cancelled", False))
        result["id"] = job_id
        with LOCK:
            if JOBS[job_id].get("cancelled"):
                raise InterruptedError("Stopped. This experiment was not saved.")
            write_json(DATA / "runs" / f"{job_id}.json", result)
            JOBS[job_id].update(status="done", progress=1, message="Complete. Results were saved automatically to History.", result=result)
    except InterruptedError as exc:
        with LOCK:
            JOBS[job_id].update(status="cancelled", message=str(exc))
    except Exception as exc:
        traceback.print_exc()
        with LOCK:
            JOBS[job_id].update(status="error", message=str(exc))
    finally:
        with LOCK:
            ACTIVE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/jobs/"):
            return
        super().log_message(fmt, *args)

    def send(self, data, status=200, content_type="application/json; charset=utf-8", download=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def export(self, format="jsonl", scope="examples", ids=None):
        if format not in ("jsonl", "csv") or scope not in ("examples", "history"):
            return self.send({"error": "Choose JSONL or CSV and Examples or History."}, 400)
        if ids is not None and (not isinstance(ids, list) or not ids or
                                any(not isinstance(item, str) or not re_full_id(item) for item in ids)):
            return self.send({"error": "Select at least one valid record to export."}, 400)
        with LOCK:
            data = records("runs" if scope == "history" else "examples")
        if ids is not None:
            available = {r["id"]: r for r in data}
            if any(item not in available for item in ids):
                return self.send({"error": "A selected record is unavailable in this collection. Refresh Examples and select again."}, 404)
            data = [available[item] for item in dict.fromkeys(ids)]
        suffix = "-selected" if ids is not None else ""
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["id", "model", "sequence_a", "sequence_b", "continuation_a", "continuation_b",
                             "tag", "notes", "patch_layer", "interpolation", "curve", "t", "d",
                             "fixed_context", "endpoint_reference", "device", "hardware_name",
                             "dtype", "batch_size", "prediction_cache", "patch_position",
                             "patch_start_a", "patch_start_b", "patch_count", "interpolation_unit", "measurement_position"])
            for r in data:
                for curve in r["curves"]:
                    for t, d in zip(curve["t"], curve["d"]):
                        writer.writerow([r["id"], r["model"], r["sequence_a"], r["sequence_b"],
                                         r["predictions"][0]["continuation"], r["predictions"][1]["continuation"],
                                         r.get("tag", ""), r.get("notes", ""), r["settings"]["patch_layer"],
                                         r["settings"]["interpolation"], curve["title"], t, d,
                                         r["settings"].get("context", "a"),
                                         r["settings"].get("endpoint_reference", "natural_matching_prefix"),
                                         r.get("device", ""), r.get("hardware", {}).get("name", ""),
                                         r.get("dtype", ""), r["settings"].get("batch_size", ""),
                                         r["settings"].get("prediction_cache", False),
                                         r["settings"].get("patch_position", "last_token"),
                                         r["settings"].get("patch_start_a", len(r["input_tokens"][0])-1),
                                         r["settings"].get("patch_start_b", len(r["input_tokens"][1])-1),
                                         r["settings"].get("patch_count", 1),
                                         r["settings"].get("interpolation_unit", "per_token_shared_t"),
                                         r["settings"].get("measurement_position", "last_token")])
            return self.send(output.getvalue().encode("utf-8-sig"), content_type="text/csv; charset=utf-8", download=f"plateau-{scope}{suffix}.csv")
        output = "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in data)
        return self.send((output + ("\n" if output else "")).encode(), content_type="application/x-ndjson", download=f"plateau-{scope}{suffix}.jsonl")

    def do_GET(self):
        global HARDWARE_PREVIEW
        route = urlparse(self.path)
        path = route.path
        try:
            if path == "/api/config":
                # Keep the UI immediately available, even during the first torch import.
                return self.send({"models": model_catalog(), "active_job": ACTIVE,
                                  "data_path": str(DATA), "model_cache_path": str(CACHE)})
            if path == "/api/hardware":
                if ENGINE is not None:
                    return self.send(ENGINE.hardware.describe())
                # Detection imports torch only in this request, keeping static pages fast.
                from hardware import Hardware
                with HARDWARE_LOCK:
                    if HARDWARE_PREVIEW is None:
                        HARDWARE_PREVIEW = Hardware().describe()
                    return self.send(HARDWARE_PREVIEW)
            if path == "/api/library":
                with LOCK:
                    all_records = records("examples")
                    history = records("runs")
                key = lambda r: r.get("created_at", "")
                return self.send({"examples": sorted(all_records, key=key, reverse=True),
                                  "history": sorted(history, key=key, reverse=True)})
            if path.startswith("/api/jobs/"):
                with LOCK:
                    job = JOBS.get(path.split("/")[-1])
                    return self.send(job if job else {"error": "Job not found."}, 200 if job else 404)
            if path == "/api/export":
                query = parse_qs(route.query)
                return self.export(query.get("format", ["jsonl"])[0],
                                   query.get("scope", ["examples"])[0])
            file = (STATIC / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if file.parent != STATIC or not file.is_file():
                return self.send({"error": "Not found"}, 404)
            return self.send(file.read_bytes(), content_type=(mimetypes.guess_type(file.name)[0] or "application/octet-stream") + "; charset=utf-8")
        except (OSError, ValueError) as exc:
            self.send({"error": str(exc)}, 500)

    def do_POST(self):
        global ACTIVE
        # Accept mutations only from this local UI.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send({"error": "Use the local Plateau Lab page for this action."}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 100_000:
                raise ValueError("The request is too large.")
            data = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Invalid request format.")
            path = urlparse(self.path).path
            if path == "/api/export":
                return self.export(data.get("format", "jsonl"), data.get("scope", "examples"),
                                   data.get("ids") if data.get("ids") is not None else [])
            if path == "/api/run":
                with LOCK:
                    if ACTIVE:
                        return self.send({"error": "Another experiment is running. Wait for it to finish or stop it first."}, 409)
                    job_id = uuid.uuid4().hex
                    JOBS[job_id] = {"id": job_id, "status": "running", "progress": 0,
                                    "message": "Preparing the model…", "request": data}
                    ACTIVE = job_id
                threading.Thread(target=run_job, args=(job_id, data), daemon=True).start()
                return self.send({"id": job_id}, 202)
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                with LOCK:
                    job_id = path.split("/")[-2]
                    if job_id not in JOBS:
                        return self.send({"error": "Job not found."}, 404)
                    if JOBS[job_id]["status"] == "running":
                        JOBS[job_id].update(cancelled=True, message="Stopping after the current computation or download step…")
                    return self.send({"ok": True})
            if path == "/api/examples":
                job_id = str(data.get("id", ""))
                if not re_full_id(job_id):
                    raise ValueError("Invalid example ID.")
                with LOCK:
                    result = read_json(DATA / "runs" / f"{job_id}.json")
                    result["tag"] = str(data.get("tag", "Unclassified"))[:80]
                    result["notes"] = str(data.get("notes", ""))[:4000]
                    result["saved_at"] = datetime.now(timezone.utc).isoformat()
                    write_json(DATA / "examples" / f"{job_id}.json", result)
                return self.send({"ok": True, "id": job_id})
            return self.send({"error": "Not found"}, 404)
        except (ValueError, OSError) as exc:
            self.send({"error": str(exc)}, 400)


def re_full_id(value):
    return len(value) == 32 and all(c in "0123456789abcdef" for c in value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"MARS V · Plateau Lab → {url}\nExamples are stored in {DATA}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
