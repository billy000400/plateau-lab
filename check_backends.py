"""Real cached Pythia parity and independent vectors through nnsight traces.

Runs nnsight locally: no NDIF key or remote inference is used. Pass --output to
save a schema-7 fixture outside the user library for browser/HTTP validation.
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
from engine import Engine
from models import cached_snapshot
import plateau.backends.nnsight_backend as nb
from plateau.core.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = Engine()
    engine.load("pythia-70m", lambda *_: None)
    backend = nb.NnsightBackend(remote=False, device=engine.device)
    catalog = {"pythia-70m": {"repo": str(cached_snapshot("pythia-70m")), "label": "Pythia 70M"}}
    request = dict(model="pythia-70m", sequence_a="The house was big", sequence_b="The house was in",
                   context="a", patch_layer=0, patch_position="different_suffix", steps=7, sample_token_matrix=True)
    original = nb.arc_segments
    for method in ("linear", "slerp"):
        local_calls = []
        handle = engine.model.register_forward_pre_hook(lambda *_: local_calls.append(1))
        try:
            local = engine.run(dict(request, interpolation=method), all_layers=True)
            on_calls = len(local_calls)
            local_calls.clear()
            local_off = engine.run(dict(request, interpolation=method, sample_token_matrix=False), all_layers=True)
        finally:
            handle.remove()
        assert on_calls-len(local_calls) == 2*math.ceil(request['steps']/local['settings']['batch_size'])
        assert local_off['curves']==local['curves'] and local_off['effect']==local['effect']
        assert all('token_matrix' not in p for p in local_off['path_predictions'])
        # Capture exactly the tensors the nnsight trace supplies to arc measurement;
        # calculate distances independently using Python math.dist / math.fsum.
        captured = []
        def capture(vectors, previous=None):
            captured.append(vectors.detach().cpu().double().tolist())
            return original(vectors, previous)
        with patch.object(nb, "arc_segments", capture), patch.object(backend, '_execution', wraps=backend._execution) as traces:
            remote_contract = run_experiment(backend, dict(request, interpolation=method), 3, models=catalog)
        enabled_traces = traces.call_count
        with patch.object(backend, '_execution', wraps=backend._execution) as traces:
            remote_off = run_experiment(backend, dict(request, interpolation=method, sample_token_matrix=False), 3, models=catalog)
        assert enabled_traces-traces.call_count == 2*math.ceil(request['steps']/3)
        assert remote_off['curves']==remote_contract['curves'] and remote_off['effect']==remote_contract['effect']
        assert remote_off['path_predictions']==[{k:v for k,v in p.items() if k!='token_matrix'} for p in remote_contract['path_predictions']]
        readouts = remote_contract["effect"]["rows"]
        for index, row in enumerate(readouts):
            points = [point for batch in captured[index::len(readouts)] for point in batch]
            assert len(points) == request["steps"]
            steps = [0.] + [math.dist(a, b) for a, b in zip(points, points[1:])]
            cumulative = [math.fsum(steps[:i + 1]) for i in range(len(steps))]
            for actual, expected in zip(row["step_lengths"], steps):
                assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (row["key"], actual, expected)
            assert math.isclose(row["total_length"], cumulative[-1], rel_tol=1e-10, abs_tol=1e-9)
            torch.testing.assert_close(torch.tensor(row["values"]["c"]), torch.tensor([v / cumulative[-1] for v in cumulative]))
            peer = next(r for r in local["effect"]["rows"] if r["key"] == row["key"])
            for metric in ("c", "relative_l2_shinkle", "relative_l2_janiak", "relative_l2_projected", "relative_angle"):
                torch.testing.assert_close(torch.tensor(row["values"][metric]), torch.tensor(peer["values"][metric]),
                                           atol=2e-3, rtol=2e-3)
        json.dumps(remote_contract, allow_nan=False)
        assert remote_contract['settings']['generation']=='greedy_3_words'
        for actual, expected in zip(remote_contract['predictions'],local['predictions']):
            assert actual['words']==expected['words'], (actual['words'],expected['words'])
            assert actual['continuation']==expected['continuation']
        for sample, peer in zip(remote_contract["path_predictions"], local["path_predictions"]):
            assert sample["token_id"] == peer["token_id"]
            matrix, reference = sample["token_matrix"], peer["token_matrix"]
            assert matrix["stop_reason"] == reference["stop_reason"]
            assert len(matrix["steps"]) == len(reference["steps"])
            for candidates, expected in zip(matrix["steps"], reference["steps"]):
                assert [t["id"] for t in candidates] == [t["id"] for t in expected]
                torch.testing.assert_close(torch.tensor([t["probability"] for t in candidates]),
                                           torch.tensor([t["probability"] for t in expected]), atol=2e-5, rtol=2e-3)
        print(f"PASS {engine.device} {method}: all-layer local/nnsight parity and independent trace-vector arc norms, "
              "including relative angle, matrix off/on identical curves, two skipped forwards per batch, and all three prediction columns.", flush=True)
    if args.output:
        local["id"] = "c" * 32
        args.output.write_text(json.dumps(local, allow_nan=False))


if __name__ == "__main__":
    main()
