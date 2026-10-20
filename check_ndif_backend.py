"""NNsight 0.8 execution and wire-format checks with tiny random local models.

No downloads, API keys, user collections, or NDIF compute are used. The remote
simulation serializes/deserializes our actual traces with local modules hidden.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from plateau.backends.nnsight_backend import NnsightBackend


class NDIFBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.repo = str(Path(cls.directory.name) / "tiny-model")
        torch.manual_seed(17)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=32, n_embd=16,
                                         n_layer=3, n_head=2, bos_token_id=0,
                                         eos_token_id=1, pad_token_id=1))
        model.save_pretrained(cls.repo)
        tokenizer = Tokenizer(WordLevel({"<bos>": 0, "<eos>": 1, "<unk>": 2,
                                        "The": 3, "house": 4, "was": 5, "big": 6, "small": 7},
                                       unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>",
                               eos_token="<eos>", unk_token="<unk>",
                               pad_token="<eos>").save_pretrained(cls.repo)
        cls.backend = NnsightBackend(remote=False, device="cpu")
        cls.lm = cls.backend.load("tiny", cls.repo)

    @classmethod
    def tearDownClass(cls):
        NnsightBackend._models.clear()
        cls.directory.cleanup()

    def assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_close(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_close(a, b)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, equal_nan=True)
        elif isinstance(expected, float):
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))  # undefined metrics before the patched layer
            else:
                self.assertAlmostEqual(actual, expected, places=6)
        else:
            self.assertEqual(actual, expected)

    def test_meta_model_and_per_user_remote_request_identity(self):
        first = NnsightBackend(remote=True, api_key="test-user-a")
        second = NnsightBackend(remote=True, api_key="test-user-b")
        lm = first.load("tiny", self.repo)
        self.assertIs(lm, second.load("tiny", self.repo))
        self.assertTrue(all(p.device.type == "meta" for p in lm.model.parameters()))
        # The synthetic checkpoint has no Hub identity; replace only that lookup.
        with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(id="plateau-test/tiny")):
            a = first._execution(lm)["backend"]
            b = second._execution(lm)["backend"]
        self.assertIsNot(a, b)
        self.assertEqual((a.api_key, b.api_key), ("test-user-a", "test-user-b"))
        name, info = a.model_key.split(":", 1)
        self.assertEqual(name, "nnsight.modeling.transformers.TransformersModel")
        self.assertEqual(json.loads(info), {"repo_id": "plateau-test/tiny", "revision": None, "task": "text-generation"})

    def test_natural_generation_survives_serialization(self):
        args = (self.lm, [3, 4, 5, 6], 1, 2, 3)
        expected = self.backend.natural(*args)
        with patch.object(self.backend, "_execution", return_value={"remote": "local"}):
            actual = self.backend.natural(*args)
        self.assertEqual(len(actual["tokens"]), 3)
        self.assertEqual(actual["source"].shape, (2, 16))
        self.assertEqual(actual["layer_states"].shape, (3, 16))
        self.assert_nested_close(actual, expected)

    def test_patch_metrics_and_optional_matrix_survive_serialization(self):
        for patch_start in (3, 2):
            a = self.backend.natural(self.lm, [3, 4, 5, 6], 1, patch_start, 3)
            b = self.backend.natural(self.lm, [3, 4, 5, 7], 1, patch_start, 3)
            for method in ("linear", "slerp"):
                with self.subTest(patch_start=patch_start, method=method):
                    args = (self.lm, [3, 4, 5, 6], 1, patch_start,
                            torch.stack([a["source"], b["source"]]), torch.tensor([0., .5, 1.]),
                            method, [0, 1, 2])
                    expected = self.backend.path(*args, sample_token_matrix=True)
                    with patch.object(self.backend, "_execution", return_value={"remote": "local"}) as traces:
                        actual = self.backend.path(*args, sample_token_matrix=True)
                        self.assertEqual(traces.call_count, 3)
                        traces.reset_mock()
                        without = self.backend.path(*args, sample_token_matrix=False)
                        self.assertEqual(traces.call_count, 1)
                    self.assert_nested_close(actual, expected)
                    self.assert_nested_close(without, {k: v for k, v in actual.items()
                                                       if k not in ("candidate_ids", "candidate_probs")})
                    self.assertEqual(torch.tensor(actual["candidate_ids"]).shape, (3, 3, 3))
                    self.assertTrue(all(math.isfinite(v) for v in actual["values"]["logits"]["relative_l2_shinkle"]))
                    self.assertLess(actual["first_error"], 1e-6)
                    self.assertLess(actual["last_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
