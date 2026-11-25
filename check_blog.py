"""Check delayed blog readouts against an independent, explicit model forward."""
import math
from types import SimpleNamespace
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from engine import interpolate
from rerun_blog_examples import measure_delayed_logits


class Tokenizer:
    def encode(self, text, **kwargs):
        return {'A': [1, 2, 3], 'B': [1, 2, 4], ' is': [5]}[text]

    def decode(self, ids):
        return str(ids[0])


class DelayedReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=16, n_positions=16, n_embd=8, n_layer=2, n_head=2,
            attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
            attn_implementation='eager')).eval()
        self.runner = SimpleNamespace(model=self.model, tokenizer=Tokenizer(), device='cpu',
            architecture=lambda: (self.model.transformer.h, self.model.transformer.drop))
        self.ts = [0., .2, .5, .8, 1.]

    def record(self, layer):
        return {'sequence_a': 'A', 'sequence_b': 'B',
                'settings': {'patch_layer': layer, 'interpolation': 'slerp'},
                'curves': [{'t': self.ts}]}

    def explicit_forward(self, tokens, layer, source=None):
        # This path calls individual modules, with no activation hooks.
        model = self.model.transformer
        ids = torch.tensor([tokens])
        hidden = model.drop(model.wte(ids) + model.wpe(torch.arange(len(tokens))))
        source_state = hidden[0, 2].clone() if layer == -1 else None
        if layer == -1 and source is not None:
            hidden[:, 2, :] = source
        for index, block in enumerate(model.h):
            hidden = block(hidden, use_cache=False)[0]
            if index == layer:
                source_state = hidden[0, 2].clone()
                if source is not None:
                    hidden[:, 2, :] = source
        logits = self.model.lm_head(model.ln_f(hidden))[0, -1]
        return source_state, logits

    def test_delayed_readout_positions_and_measured_distances(self):
        with torch.inference_mode():
            for layer in (-1, 0):
                with self.subTest(layer=layer):
                    source_a, _ = self.explicit_forward([1, 2, 3], layer)
                    source_b, _ = self.explicit_forward([1, 2, 4], layer)
                    vectors = [self.explicit_forward([1, 2, 3, 5], layer, state)[1]
                               for state in interpolate(source_a, source_b, self.ts)]
                    values = [row.tolist() for row in vectors]
                    steps = [0.] + [math.dist(a, b) for a, b in zip(values, values[1:])]
                    total = math.fsum(steps)
                    expected_c = [math.fsum(steps[:i+1])/total for i in range(len(steps))]
                    expected_d = [math.dist(row, values[0]) /
                                  (math.dist(row, values[0])+math.dist(row, values[-1]))
                                  for row in values]
                    record = self.record(layer)
                    actual = measure_delayed_logits(self.runner, record)
                    torch.testing.assert_close(torch.tensor(actual['c']), torch.tensor(expected_c))
                    torch.testing.assert_close(torch.tensor(actual['d']), torch.tensor(expected_d))
                    torch.testing.assert_close(torch.tensor(actual['step_lengths']), torch.tensor(steps))
                    self.assertEqual(actual['patch_position'], 2)
                    self.assertEqual(actual['measurement_position'], 3)
                    self.assertEqual(actual['forced_token_ids'], [5])
                    self.assertEqual([p['token_id'] for p in actual['predictions']],
                                     [int(row.argmax()) for row in vectors])
                    self.assertIs(record['blog_readouts']['delayed_logits'], actual)
                    self.assertFalse(any(module._forward_hooks for module in self.model.modules()))

    def test_hooks_removed_after_forward_failure(self):
        calls = [0]
        def fail_during_patch(module, args, output):
            calls[0] += 1
            if calls[0] == 5:
                raise RuntimeError('Test interruption')
        handle = self.model.register_forward_hook(fail_during_patch)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Test interruption'):
                measure_delayed_logits(self.runner, self.record(0))
            self.assertFalse(self.model.transformer.h[0]._forward_hooks)
        finally:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
