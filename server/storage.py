"""Local JSON collections and the existing selected CSV/JSONL export contract."""
import csv
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def re_full_id(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()

    def records(self, folder):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.root / folder).glob("*.json"), reverse=True)]

    def write(self, folder, record):
        if not re_full_id(record.get("id")):
            raise ValueError("Invalid record ID.")
        path = self.root / folder / (record["id"] + ".json")
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def example(self, job_id, tag, notes):
        if not re_full_id(job_id):
            raise ValueError("Invalid example ID.")
        with self.lock:
            path = self.root / "runs" / (job_id + ".json")
            if not path.is_file():
                path = self.root / "examples" / (job_id + ".json")
            record = json.loads(path.read_text(encoding="utf-8"))
            record.update(tag=str(tag)[:80], notes=str(notes)[:4000], saved_at=datetime.now(timezone.utc).isoformat())
            self.write("examples", record)
        return record

    @staticmethod
    def send(data, status=200, content_type="application/json", download=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return data, status, content_type, download

    def export(self, format="jsonl", scope="examples", ids=None):
        if format not in ("jsonl", "csv") or scope not in ("examples", "history"):
            return self.send({"error": "Choose JSONL or CSV and Examples or History."}, 400)
        if ids is not None and (not isinstance(ids, list) or not ids or
                                any(not isinstance(item, str) or not re_full_id(item) for item in ids)):
            return self.send({"error": "Select at least one valid record to export."}, 400)
        with self.lock:
            data = self.records("runs" if scope == "history" else "examples")
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
                             "patch_start_a", "patch_start_b", "patch_count", "interpolation_unit", "measurement_position",
                             "schema_version", "c", "step_length", "cumulative_length", "total_length",
                             "c_status", "d_status", "c_undefined_reason", "d_undefined_reason", "metric_definitions"])
            for r in data:
                definitions = (json.dumps(r["metric_definitions"], ensure_ascii=False, allow_nan=False)
                               if "metric_definitions" in r else "")
                for curve in r["curves"]:
                    missing = [None] * len(curve["t"])
                    for index, (t, d) in enumerate(zip(curve["t"], curve["d"])):
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
                                         r["settings"].get("measurement_position", "last_token"),
                                         r.get("schema_version", 1), curve.get("c", missing)[index],
                                         curve.get("step_lengths", missing)[index],
                                         curve.get("cumulative_length", missing)[index], curve.get("total_length"),
                                         curve.get("c_status", "not_recorded"), curve.get("d_status", "ok"),
                                         curve.get("c_undefined_reason"), curve.get("d_undefined_reason"),
                                         definitions])
            return self.send(output.getvalue().encode("utf-8-sig"), content_type="text/csv; charset=utf-8", download=f"plateau-{scope}{suffix}.csv")
        output = "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in data)
        return self.send((output + ("\n" if output else "")).encode(), content_type="application/x-ndjson", download=f"plateau-{scope}{suffix}.jsonl")
