"""Real-model checks for generation caching and restart-after-OOM behavior."""
import argparse
from unittest.mock import patch

import torch
from engine import Engine
from models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pythia-160m', choices=list(MODELS))
parser.add_argument('--oom', action='store_true', help='Also inject an OOM into the real forward path.')
args = parser.parse_args()
engine = Engine()
engine.load(args.model, lambda *_: None)

with torch.inference_mode():
    for prompt in ('The house was big', ('The researcher reviewed the evidence and wrote a report. ' * 12) + 'The capital of France is'):
        ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        shapes = []
        handle = engine.model.register_forward_pre_hook(lambda _, inputs: shapes.append(inputs[0].shape[1]))
        try:
            cached = engine.predict(ids)
        finally:
            handle.remove()
        uncached = engine.predict(ids, use_cache=False)
        assert cached['continuation'] == uncached['continuation']
        assert [t['id'] for t in cached['tokens']] == [t['id'] for t in uncached['tokens']]
        assert shapes[0] == len(ids) and all(n == 1 for n in shapes[1:])
        print(f'PASS: {args.model} on {engine.device}, {len(ids)}-token prefix; cached and uncached greedy tokens agree, subsequent steps process one token.', flush=True)

if args.oom:
    # A real forward first produces a partial curve, then fails. The retry must
    # discard that partial curve and remove its patch hooks before restarting.
    request = dict(model=args.model, sequence_a='I live in London',
                   sequence_b='After many years abroad, she returned to Berlin',
                   context='a', patch_layer=0, interpolation='slerp', steps=11)
    calls = 0
    def fail_once(_, inputs):
        global calls
        if inputs[0].shape[0] == 4:
            calls += 1
            if calls == 3:
                raise torch.OutOfMemoryError('Simulated allocation failure after a partial curve')
    handle = engine.model.register_forward_pre_hook(fail_once)
    try:
        with patch.object(engine.hardware, 'batch_size', return_value=4):
            retried = engine.run(request)
    finally:
        handle.remove()
    assert retried['settings']['batch_retries'] == 1
    assert retried['settings']['batch_size'] == 2
    assert all(not module._forward_hooks for module in engine.model.modules())
    with patch.object(engine.hardware, 'batch_size', return_value=2):
        reference = engine.run(request)
    for actual, expected in zip(retried['curves'], reference['curves']):
        assert len(actual['d']) == 11
        assert torch.allclose(torch.tensor(actual['d']), torch.tensor(expected['d']), atol=1e-6)
        assert actual['d'][0] < 1e-4 and actual['d'][-1] > 1-1e-4
    for key in ('natural_l2', 'patched_l2'):
        torch.testing.assert_close(
            torch.tensor([row[key] for row in retried['l2_distances']['layers']]),
            torch.tensor([row[key] for row in reference['l2_distances']['layers']]),
            atol=1e-6, rtol=1e-6)
    print('PASS: OOM retry halves batch size, removes hooks, discards partial samples, and matches a clean run.', flush=True)
