"""Streaming measurements of sampled vector trajectories, independent per readout."""
import math

import torch


# Preserve the existing absolute endpoint tolerance for d. Arc length has no
# positive cutoff: even very small measured movement counts (in float64).
ENDPOINT_TOLERANCE = 1e-8
METRIC_DEFINITIONS = {
    "c": {
        "name": "Normalized cumulative sampled Euclidean arc length",
        "formula": "c(t_k) = sum(i=1..k, ||z_i-z_(i-1)||_2) / sum(i=1..N, ||z_i-z_(i-1)||_2)",
        "representation": "final-token resid_post for layers; final-token logits for output",
        "context": "Same fixed context, intervention and ordered t samples as d",
        "denominator": "Fixed total sampled path length for this run and readout",
        "accumulation": "CPU float64 vector differences and L2 norms; compensated scalar summation",
        "zero_length_tolerance": 0.0,
        "undefined": "Exactly zero total measured length; nonfinite vectors are also undefined",
        "reference": "c(t)=t means uniform accumulation per unit t, not necessarily a straight trajectory",
    },
    "d": {
        "name": "Relative endpoint distance",
        "formula": "d(t) = ||z(t)-z(0)||_2 / (||z(t)-z(0)||_2 + ||z(t)-z(1)||_2)",
        "endpoint_tolerance": ENDPOINT_TOLERANCE,
        "undefined": "Endpoint L2 below the absolute tolerance; does not imply a stationary path",
    },
}


def relative_distance(x, a, b):
    # Keep arithmetic and tolerance unchanged for existing, nondegenerate d.
    if torch.linalg.vector_norm(a - b).item() < ENDPOINT_TOLERANCE:
        raise ValueError("The endpoint outputs coincide within tolerance; d(t) is undefined.")
    da = torch.linalg.vector_norm(x - a, dim=-1)
    db = torch.linalg.vector_norm(x - b, dim=-1)
    return da / (da + db)


class TrajectoryReadout:
    """Retain only one previous vector plus scalars across inference batches.

    Instances belong to one measurement attempt, so retries/cancellation cannot
    leak earlier steps into a new run. Call append with real samples only.
    """
    def __init__(self, endpoints):
        self.endpoints = endpoints
        distance = torch.linalg.vector_norm(endpoints[0] - endpoints[1]).item()
        self.d_status = ("nonfinite" if not math.isfinite(distance) else
                         "coincident_endpoints" if distance < ENDPOINT_TOLERANCE else "ok")
        self.d = []
        self.previous = None
        self.steps = []
        self.cumulative = []
        self.total = self.correction = 0.0
        self.nonfinite = False

    def append(self, vectors):
        if not len(vectors):
            return
        if self.d_status == "ok":
            values = relative_distance(vectors, *self.endpoints).cpu().tolist()
            if not all(math.isfinite(value) for value in values):
                self.d_status = "nonfinite"
            self.d.extend(values)
        else:
            self.d.extend([None] * len(vectors))

        # MPS does not implement float64. Move this small readout batch to CPU
        # before subtracting; don't cast a float32 difference after the fact.
        current = vectors.detach().cpu().to(dtype=torch.float64)
        first = 0.0 if self.previous is None else torch.linalg.vector_norm(current[0] - self.previous).item()
        lengths = [first] + torch.linalg.vector_norm(current[1:] - current[:-1], dim=-1).tolist()
        self.previous = current[-1].clone()  # Do not retain the full batch via a view.
        self.nonfinite |= not bool(torch.isfinite(current).all())
        for length in lengths:
            self.nonfinite |= not math.isfinite(length)
            if self.nonfinite:
                self.steps.append(None)
                self.cumulative.append(None)
                continue
            adjusted = length - self.correction
            updated = self.total + adjusted
            self.correction = (updated - self.total) - adjusted
            self.total = updated
            self.steps.append(length)
            self.cumulative.append(self.total)

    def finish(self):
        c_status = "nonfinite" if self.nonfinite else "stationary" if self.total == 0 else "ok"
        count = len(self.d)
        return {
            "d": self.d if self.d_status == "ok" else [None] * count,
            "d_status": self.d_status,
            "d_undefined_reason": None if self.d_status == "ok" else (
                "Coincident endpoints (L2 < 1e-8); d(t) undefined" if self.d_status == "coincident_endpoints"
                else "Nonfinite endpoint distances; d(t) undefined"),
            "c": [value / self.total for value in self.cumulative] if c_status == "ok" else [None] * count,
            "c_status": c_status,
            "c_undefined_reason": None if c_status == "ok" else (
                "No measured movement; c(t) undefined" if c_status == "stationary"
                else "Nonfinite trajectory; c(t) undefined"),
            "step_lengths": self.steps if not self.nonfinite else [None] * count,
            "cumulative_length": self.cumulative if not self.nonfinite else [None] * count,
            "total_length": self.total if not self.nonfinite else None,
        }


def summarize(t, values, metric):
    deviation = "mean_uniform_progress_deviation" if metric == "c" else "mean_linear_deviation"
    if not values or any(value is None for value in values):
        return {"metric": metric, "max_abs_slope": None, "peak_t": None, deviation: None}
    slopes = [(values[i + 1] - values[i]) / (t[i + 1] - t[i]) for i in range(len(t) - 1)]
    peak = max(range(len(slopes)), key=lambda i: abs(slopes[i]))
    return {"metric": metric, "max_abs_slope": abs(slopes[peak]),
            "peak_t": (t[peak] + t[peak + 1]) / 2,
            deviation: math.fsum(abs(value - alpha) for value, alpha in zip(values, t)) / len(t)}
