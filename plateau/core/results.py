"""Shared schema 7: cumulative progress plus the hosted app's endpoint metrics."""
import math

from plateau.core.math import METRICS, transition_width
from plateau.core.trajectory import METRIC_DEFINITIONS, TrajectoryReadout, summarize

SCHEMA_VERSION = 7
C_METRIC = {"id": "c", "label": "Cumulative path progress c(t)", "range": [0, 1], "plateau_score": True}


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


class EffectReadout(TrajectoryReadout):
    """Local vectors share the original c/d arithmetic and add endpoint metrics."""
    def __init__(self, endpoints):
        super().__init__(endpoints)
        self.endpoint_l2 = float((endpoints[0] - endpoints[1]).norm())
        self.extra = {key: [] for key in METRICS if key != "relative_l2_shinkle"}

    def append(self, vectors):
        super().append(vectors)
        for key, values in self.extra.items():
            values.extend(METRICS[key]["fn"](vectors, *self.endpoints).cpu().tolist()
                          if self.d_status == "ok" else [None] * len(vectors))

    def finish(self):
        result = super().finish()
        result.update(endpoint_l2=self.endpoint_l2 if math.isfinite(self.endpoint_l2) else None)
        result.update({key: values if all(v is not None and math.isfinite(v) for v in values)
                       else [None] * len(values) for key, values in self.extra.items()})
        return result


def arc_from_segments(segments):
    """Accumulate backend-measured lengths once over the whole ordered run."""
    if any(value is None or not math.isfinite(value) or value < 0 for value in segments):
        return {"c": [None] * len(segments), "step_lengths": [None] * len(segments),
                "cumulative_length": [None] * len(segments), "total_length": None,
                "c_status": "nonfinite", "c_undefined_reason": "Nonfinite trajectory; c(t) undefined"}
    total = correction = 0.0
    cumulative = []
    for length in segments:
        adjusted = length - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
        cumulative.append(total)
    return {"c": [value / total for value in cumulative] if total else [None] * len(segments),
            "step_lengths": segments, "cumulative_length": cumulative, "total_length": total,
            "c_status": "ok" if total else "stationary",
            "c_undefined_reason": None if total else "No measured movement; c(t) undefined"}


def add_effect(record, readouts):
    """Attach all available readouts without inventing missing legacy measurements."""
    ts = record["curves"][0]["t"]
    descriptors = [C_METRIC] + [{"id": key, **{k: info[k] for k in ("label", "range", "plateau_score")}}
                                for key, info in METRICS.items()]
    rows = []
    for key, readout in readouts.items():
        values = {"c": readout["c"], "relative_l2_shinkle": readout["d"],
                  **{metric: readout.get(metric) for metric in METRICS if metric != "relative_l2_shinkle"}}
        defined = {metric: bool(value) and all(v is not None and math.isfinite(v) for v in value)
                   for metric, value in values.items()}
        rows.append({"key": key, "layer": None if key == "logits" else int(key),
                     "label": "Logits" if key == "logits" else f"Layer {key}",
                     "patched": key == str(record["settings"]["patch_layer"]),
                     "defined": any(defined.values()), "defined_metrics": defined,
                     "endpoint_l2": readout.get("endpoint_l2"),
                     "total_length": readout["total_length"],
                     "step_lengths": readout["step_lengths"], "cumulative_length": readout["cumulative_length"],
                     "c_status": readout["c_status"], "d_status": readout["d_status"],
                     "c_undefined_reason": readout["c_undefined_reason"],
                     "d_undefined_reason": readout["d_undefined_reason"],
                     "values": {metric: value if defined[metric] else None for metric, value in values.items()},
                     "plateau_score": {metric: transition_width(ts, value) if defined[metric] else None
                                       for metric, value in values.items()}})
    record.update(schema_version=SCHEMA_VERSION, primary_metric="c", metric_definitions=METRIC_DEFINITIONS)
    record["effect"] = {"t": ts, "position": "last_token", "representation": "resid_post",
                        "metrics": descriptors, "rows": rows,
                        "plateau_score_definition": "First-crossing width from 0.1 to 0.9; uniform progress gives 0.8"}
    record["metrics"].update(c=summarize(ts, readouts["logits"]["c"], "c"),
                              d=summarize(ts, readouts["logits"]["d"], "d"))
    return finite_json(record)
