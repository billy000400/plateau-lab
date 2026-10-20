"""Offline cross-backend measurement and FastAPI compatibility checks."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from plateau.core.experiment import run_experiment
from plateau.core.math import arc_segments, effect_metrics
from plateau.core.results import arc_from_segments
from plateau.core.models import MODELS as LOCAL_MODELS, ndif_models, remote_catalog
from server.storage import Library
import server.app as server


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, **kwargs):
        return [1 if text == "A" else 2, 3]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Backend:
    """Known vector paths; exercise real experiment orchestration and reduction."""
    def __init__(self, points, oom=False):
        self.points = torch.tensor(points, dtype=torch.float32)
        self.natural_calls = self.calls = 0
        self.oom = oom

    def load(self, *args):
        return SimpleNamespace(tokenizer=Tokenizer(), context_limit=256, blocks=[None],
                               config=SimpleNamespace(model_type="synthetic"))

    def natural(self, *args):
        side = self.natural_calls % 2
        self.natural_calls += 1
        return {"source": torch.tensor([[float(side), 1.0]]),
                "layer_states": self.points[[0 if side == 0 else -1]],
                "logits": self.points[0 if side == 0 else -1], "dtype": "float32",
                "tokens": [1, 2, 3], "probabilities": [.5, .5, .5]}

    def path(self, lm, ids, layer, start, sources, ts, method, record_layers,
             include_reference_logits=False, previous=None, cancelled=lambda: False, sample_token_matrix=False):
        self.calls += 1
        if self.oom and self.calls == 2:
            raise RuntimeError("out of memory after a partial path")
        indices = (ts * (len(self.points) - 1)).round().long()
        samples = self.points[indices]
        values, gap = effect_metrics(samples, self.points[0], self.points[-1])
        keys = [str(layer) for layer in record_layers] + ["logits"]
        arcs = {key: arc_segments(samples, (previous or {}).get(key)) for key in keys}
        return {"values": {key: {name: value.tolist() for name, value in values.items()} for key in keys},
                "gaps": {key: float(gap) for key in keys}, "endpoint_l2": [float(gap)],
                "step_lengths": {key: arc[0].tolist() for key, arc in arcs.items()},
                "last_vectors": {key: arc[1] for key, arc in arcs.items()},
                "reference_tokens": [0, 1], "tokens": [0] * len(ts),
                "candidate_ids": [[[0, 1, 2]] * len(ts)] * 3,
                "candidate_probs": [[[.5, .3, .1]] * len(ts)] * 3,
                "first_error": 0.0, "last_error": 0.0, "reference_logits": self.points[[0, -1]]}

    def describe(self):
        return {"remote": False, "device": "synthetic"}


REQUEST = dict(model="synthetic", sequence_a="A", sequence_b="B", steps=5,
               interpolation="linear", patch_layer=0)
MODELS = {"synthetic": {"repo": "no-network", "label": "Synthetic"}}
POINTS = [[0, 0], [.5, 1], [1, -1], [1.5, 1], [2, 0]]


def run(backend, batch=2, progress=lambda *_: None, cancelled=lambda: False):
    return run_experiment(backend, REQUEST, batch, progress, cancelled, MODELS)


class ModelCatalog(unittest.TestCase):
    @staticmethod
    def deployment(repo, pinned, running=True):
        return {"repo_id": repo, "model_key": "nnsight.modeling.language.LanguageModel:" + repo,
                "config": json.dumps({"model_type": "gpt2", "n_layer": 12}),
                "deployment_level": "HOT" if running else "WARM",
                "application_state": "RUNNING" if running else "STARTING", "pinned": pinned}

    def test_transformers_model_status_and_legacy_replicas(self):
        modern = {**self.deployment("remote/pinned", True),
                  "model_key": 'nnsight.modeling.transformers.TransformersModel:{"repo_id":"remote/pinned","revision":null,"task":"text-generation"}',
                  "task": "text-generation", "revision": None,
                  "config": json.dumps({"model_type": "llama", "num_hidden_layers": 32})}
        legacy = self.deployment("remote/pinned", False)
        for ordered in ([modern, legacy], [legacy, modern]):
            models = remote_catalog(ndif_models({"deployments": dict(enumerate(ordered))}))
            self.assertTrue(models["remote/pinned"]["available"])
            self.assertTrue(models["remote/pinned"]["pinned"])
            self.assertEqual(models["remote/pinned"]["layers"], 32)
            self.assertEqual(server.selected_model(models), "remote/pinned")

    def test_skip_other_tasks_revisions_and_bad_configs_without_losing_valid_models(self):
        base = {**self.deployment("remote/valid", True),
                "model_key": "nnsight.modeling.transformers.TransformersModel:{}",
                "task": "text-generation", "config": {"model_type": "gpt2", "n_layer": 12}}
        invalid = [dict(task="image-text-to-text"), dict(task="fill-mask"), dict(task=None),
                   dict(revision="experiment"), dict(config="{broken"), dict(config="[]"),
                   dict(config={"model_type": "unknown", "n_layer": 12}),
                   dict(deployment_level="COLD"), dict(model_key="another.Model:{}")]
        deployments = {str(i): {**base, "repo_id": f"remote/invalid-{i}", **extra}
                       for i, extra in enumerate(invalid)}
        deployments["valid"] = base
        self.assertEqual(list(ndif_models({"deployments": deployments})), ["remote/valid"])

    def test_running_is_not_the_same_as_access_without_hotswapping(self):
        deployments = {
            "unpinned": self.deployment("openai-community/gpt2", False),
            "pinned": self.deployment("remote/pinned", True),
            "stopped": self.deployment("remote/stopped", True, running=False),
        }
        models = remote_catalog(ndif_models({"deployments": deployments}))
        self.assertTrue(models["gpt2"]["available"])
        self.assertTrue(models["gpt2"]["requires_hotswap"])
        self.assertFalse(models["remote/pinned"]["requires_hotswap"])
        self.assertFalse(models["remote/stopped"]["available"])
        self.assertEqual(server.selected_model(models), "remote/pinned")
        # A user with hotswap permission may explicitly select the unpinned model.
        self.assertEqual(server.selected_model(models, "gpt2"), "gpt2")
        only_unpinned = remote_catalog(ndif_models({"deployments": {"unpinned": deployments["unpinned"]}}))
        self.assertEqual(server.selected_model(only_unpinned), "gpt2")

    def test_running_pinned_replica_wins_in_either_status_order(self):
        repo = "remote/shared"
        replicas = [self.deployment(repo, False), self.deployment(repo, True),
                    self.deployment(repo, True, running=False)]
        for ordered in (replicas, list(reversed(replicas))):
            with self.subTest(order=[(d["pinned"], d["application_state"]) for d in ordered]):
                models = remote_catalog(ndif_models({"deployments": dict(enumerate(ordered))}))
                self.assertTrue(models[repo]["available"])
                self.assertTrue(models[repo]["pinned"])
                self.assertFalse(models[repo]["requires_hotswap"])


class Measurements(unittest.TestCase):
    def test_actual_segments_at_every_batch_split(self):
        expected = [0.] + [math.dist(a, b) for a, b in zip(POINTS, POINTS[1:])]
        total = math.fsum(expected)
        for batch in (1, 2, 3, 5):
            result = run(Backend(POINTS), batch)
            self.assertEqual(result["schema_version"], 7)
            for row in result["effect"]["rows"]:
                for actual, target in zip(row["step_lengths"], expected):
                    self.assertAlmostEqual(actual, target, places=12)
                self.assertAlmostEqual(row["total_length"], total, places=12)
                self.assertEqual(row["values"]["c"][-1], 1)
            json.dumps(result, allow_nan=False)

    def test_loop_stationary_and_nonfinite(self):
        cases = [([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], "ok"),
                 ([[0, 0]] * 5, "stationary"),
                 ([[0, 0], [1, 0], [float("nan"), 1], [1, 0], [0, 0]], "nonfinite")]
        for points, status in cases:
            result = run(Backend(points))
            row = result["effect"]["rows"][-1]
            self.assertEqual(row["c_status"], status)
            self.assertFalse(row["defined_metrics"]["relative_l2_shinkle"])
            self.assertEqual(row["defined_metrics"]["c"], status == "ok")
            json.dumps(result, allow_nan=False)

    def test_tiny_arc_and_float64_before_subtraction(self):
        points = torch.tensor([[1e20, 0], [-1e20, 1e-20], [1e20, 0]], dtype=torch.float32)
        lengths, _ = arc_segments(points)
        self.assertTrue(torch.isfinite(lengths).all())
        tiny = arc_from_segments([0, 1e-30, 2e-30])
        self.assertEqual(tiny["c_status"], "ok")
        self.assertEqual(tiny["c"][-1], 1)

    def test_oom_discards_partial_path(self):
        result = run(Backend(POINTS, oom=True), 2)
        clean = run(Backend(POINTS), 1)
        self.assertEqual(result["settings"]["batch_retries"], 1)
        self.assertEqual(result["curves"], clean["curves"])
        self.assertEqual(result["effect"], clean["effect"])
        self.assertEqual(result["path_predictions"], clean["path_predictions"])

    def test_cancel_after_final_chunk_and_reuse(self):
        stopped = False
        def progress(message, fraction):
            nonlocal stopped
            if message.startswith("Measuring c(t)"):
                stopped = True
        backend = Backend(POINTS)
        with self.assertRaises(InterruptedError):
            run(backend, 5, progress, lambda: stopped)
        self.assertEqual(run(backend)["curves"], run(Backend(POINTS))["curves"])


class API(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.library = Library(self.directory.name)
        self.patches = [patch.object(server, "REMOTE", False), patch.object(server, "LIBRARY", self.library),
                        patch.object(server, "JOBS", {}), patch.object(server, "LOCAL_ENGINE", None)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app)
        self.record = run(Backend(POINTS))
        self.record["id"] = "a" * 32
        self.library.write("runs", self.record)

    def tearDown(self):
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def test_local_library_examples_selected_exports(self):
        self.assertTrue(self.client.get("/api/config").json()["local_library"])
        self.assertEqual(self.client.get("/classic/").status_code, 200)
        self.assertIn('src="app.js"', self.client.get("/classic/").text)
        self.assertIn("function renderCurves", self.client.get("/classic/app.js").text)
        self.assertEqual(self.client.get('/').text,self.client.get('/classic/').text)
        self.assertNotIn('id="effect-preset"', self.client.get('/').text)
        response = self.client.post("/api/examples", json={"id": self.record["id"], "tag": "loop", "notes": 'Unicode α\n"quoted", note'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.client.get("/api/library").json()["examples"]), 1)
        for fmt in ("jsonl", "csv"):
            exported = self.client.post("/api/export", json={"format": fmt, "scope": "examples", "ids": [self.record["id"]]})
            self.assertEqual(exported.status_code, 200, exported.text)
            self.assertIn("cumulative_length", exported.text)
            self.assertIn("α", exported.text)
        for ids, status in (([], 400), (["all"], 400), (["b" * 32], 404)):
            self.assertEqual(self.client.post("/api/export", json={"ids": ids}).status_code, status)
        self.assertEqual(self.client.post("/api/examples", json={"id": "../bad"}).status_code, 400)
        self.assertEqual(self.client.post("/api/examples", headers={"Origin": "https://elsewhere.example"}, json={}).status_code, 403)

    def test_remote_has_no_local_data_and_requires_key(self):
        with patch.object(server, "REMOTE", True), patch.object(server, "SERVER_KEY", False):
            page=self.client.get('/').text
            for element in ('id="l2-section"','id="c-charts"','id="charts"','id="sample-values"','id="save"','id="notes"','id="tag"','id="sample-token-matrix"'):
                self.assertIn(element,page)
            self.assertEqual(page,self.client.get('/classic/').text)
            for path in ("/api/library", "/api/export", "/api/hardware"):
                self.assertEqual(self.client.get(path).status_code, 404)
            self.assertEqual(self.client.post("/api/examples", json={}).status_code, 404)
            self.assertEqual(self.client.post("/api/run", json=REQUEST).status_code, 401)

    def test_server_cancellation_does_not_write_result(self):
        job_id = "b" * 32
        server.JOBS[job_id] = {"status": "queued"}
        def finish(*args, **kwargs):
            server.JOBS[job_id]["cancelled"] = True
            return dict(self.record)
        with patch.object(server, "LOCAL_ENGINE", SimpleNamespace(run=finish)):
            server.run_job(job_id, REQUEST, MODELS, None)
        self.assertEqual(server.JOBS[job_id]["status"], "cancelled")
        self.assertEqual(len(self.library.records("runs")), 1)

    def test_default_model_and_active_job_restore(self):
        with patch.object(server.WORKER, "submit"):
            request = dict(REQUEST)
            request.pop("model")
            response = self.client.post("/api/run", json=request)
            self.assertEqual(response.status_code, 202)
            job = self.client.get("/api/jobs/" + response.json()["id"]).json()
            self.assertEqual(job["request"]["model"], "gpt2")
            self.assertFalse(job["request"]["sample_token_matrix"])
            self.assertEqual(self.client.post("/api/run", json=request).status_code, 409)

    def test_remote_catalog_visibility_aliases_and_request_guards(self):
        deployed = {'openai-community/gpt2': {**LOCAL_MODELS['gpt2'], 'running': True},
                    'remote/extra': {'repo':'remote/extra','label':'Extra','layers':3,'family':'Test','running':True},
                    'remote/sleeping': {'repo':'remote/sleeping','label':'Sleeping','layers':3,'family':'Test','running':False}}
        models = remote_catalog(deployed)
        self.assertFalse(models['pythia-70m']['available'])
        self.assertTrue(models['gpt2']['available'])
        self.assertFalse(models['remote/sleeping']['available'])
        self.assertEqual(len(models),len(LOCAL_MODELS)+2)
        with patch.object(server,'REMOTE',True), patch.object(server,'SERVER_KEY',True), patch.object(server,'catalog',return_value=models), patch.object(server.WORKER,'submit'):
            config = self.client.get('/api/config').json()
            self.assertEqual(config['default_model'],'gpt2')
            self.assertIn('pythia-70m',[m['id'] for m in config['models']])
            for model in ('pythia-70m','remote/sleeping'):
                self.assertEqual(self.client.post('/api/run',json={**REQUEST,'model':model}).status_code,400)
                self.assertEqual(self.client.post('/api/tokenize',json={'model':model}).status_code,400)
            self.assertEqual(server.JOBS,{})
            for value in ('false',1,None):
                self.assertEqual(self.client.post('/api/run',json={**REQUEST,'sample_token_matrix':value}).status_code,400)
            response=self.client.post('/api/run',json={**REQUEST,'model':'openai-community/gpt2','sample_token_matrix':True})
            self.assertEqual(response.status_code,202)
            job=self.client.get('/api/jobs/'+response.json()['id']).json()
            self.assertEqual(job['request']['model'],'gpt2')
            self.assertTrue(job['request']['sample_token_matrix'])

    def test_remote_status_outage_keeps_collections_ui_usable(self):
        with patch.object(server,'REMOTE',True), patch.object(server,'CATALOG',{'models':None,'fetched':0}), patch.object(server.urllib.request,'urlopen',side_effect=OSError('offline')):
            response=self.client.get('/api/config')
            self.assertEqual(response.status_code,200)
            self.assertIsNone(response.json()['default_model'])
            self.assertTrue(all(not m['available'] for m in response.json()['models']))

    def test_hotswap_rejection_explains_the_account_requirement(self):
        job_id = "c" * 32
        api_key = "11111111-1111-1111-1111-111111111111"
        server.JOBS[job_id] = {"status": "queued"}
        error = RuntimeError("RemoteException: Model is not pinned and hotswapping is not supported for this API key. " + api_key)
        with patch.object(server, "REMOTE", True), patch.object(server, "backend"), \
             patch("plateau.core.experiment.run_experiment", side_effect=error):
            server.run_job(job_id, REQUEST, MODELS, api_key)
        job = self.client.get("/api/jobs/" + job_id).json()
        self.assertEqual(job["status"], "error")
        self.assertIn("your API key does not permit hotswapping", job["message"])
        self.assertIn("Pinned on NDIF", job["message"])
        self.assertIn("https://nnsight.net/status/", job["message"])
        self.assertNotIn(api_key, json.dumps(job))
        self.assertNotIn("RemoteException", job["message"])

    def test_example_remains_editable_after_history_deletion(self):
        key=self.record['id']
        self.assertEqual(self.client.post('/api/examples',json={'id':key,'notes':'first'}).status_code,200)
        self.assertEqual(self.client.post('/api/history/delete',json={'ids':[key]}).status_code,200)
        self.assertEqual(self.client.post('/api/examples',json={'id':key,'notes':'revised','tag':'Plateau'}).status_code,200)
        collections=self.client.get('/api/library').json()
        self.assertEqual(collections['history'],[])
        self.assertEqual(collections['examples'][0]['notes'],'revised')


if __name__ == "__main__":
    unittest.main(verbosity=2)
