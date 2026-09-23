"""Check suffix intervention against actual full residual states. No collection writes."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from engine import Engine, interpolate
from models import MODELS, cached_snapshot

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pythia-160m', choices=list(MODELS))
parser.add_argument('--quick', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
assert cached_snapshot(args.model), 'This check requires an already cached model.'
engine = Engine()
engine.load(args.model, lambda *_: None)
blocks, _ = engine.architecture()
kind = engine.model.config.model_type
normalization = (engine.model.transformer.ln_f if kind == 'gpt2' else
                 engine.model.gpt_neox.final_layer_norm if kind == 'gpt_neox' else engine.model.model.norm)


def next_module(layer):
    return blocks[layer+1] if layer+1 < len(blocks) else normalization


@torch.inference_mode()
def natural_states(text, layer):
    # The input to the next block (or final norm) independently identifies resid_post.
    captured = []
    handle = next_module(layer).register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0][0].clone()))
    try:
        ids = engine.tokenizer.encode(text, add_special_tokens=False)
        engine.model(torch.tensor([ids], device=engine.device), use_cache=False)
        return ids, captured[0]
    finally:
        handle.remove()


def check(a, b, layer=2, method='slerp', context='a', batch=1):
    ids_a, a_states = natural_states(a, layer)
    ids_b, b_states = natural_states(b, layer)
    assert len(ids_a) == len(ids_b)
    start = next(i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y)
    inputs_to_next = []
    original_measure = engine._measure_path

    def inspect(*params, **kwargs):
        handle = next_module(layer).register_forward_pre_hook(lambda _, inputs: inputs_to_next.append(inputs[0].detach().cpu().clone()))
        try:
            return original_measure(*params, **kwargs)
        finally:
            handle.remove()

    request = dict(model=args.model, sequence_a=a, sequence_b=b, patch_layer=layer,
                   patch_position='different_suffix', context=context, interpolation=method, steps=5)
    with patch.object(engine, '_measure_path', side_effect=inspect), patch.object(engine.hardware, 'batch_size', return_value=batch):
        result = engine.run(request)
    s = result['settings']
    assert s['patch_start_a'] == s['patch_start_b'] == start
    assert s['patch_count'] == len(ids_a)-start
    assert s['measurement_position'] == 'last_token'
    assert result['experiment']['natural_endpoints_reproduced']
    assert max(result['metrics']['patched_vs_natural_max_abs_logit_gap'].values()) < 2e-3
    assert max(result['metrics']['endpoint_max_abs_logit_error'].values()) < 2e-3
    # Skip the independent reference forwards, then collect exactly the five t samples.
    observed = torch.cat(inputs_to_next[2 if batch == 1 else 1:])[:5]
    with torch.inference_mode():
        ts = torch.linspace(0, 1, 5, device=engine.device)
        expected = torch.stack([interpolate(a_states[p], b_states[p], ts, method) for p in range(start, len(ids_a))], dim=1).cpu()
    torch.testing.assert_close(observed[:, start:], expected, atol=2e-4, rtol=2e-4)
    if start:
        torch.testing.assert_close(observed[:, :start], a_states[:start].cpu().expand(5, -1, -1), atol=2e-4, rtol=2e-4)
    difference = a_states[start:].cpu().double() - b_states[start:].cpu().double()
    d = result['l2_distances']
    assert abs(d['source_l2'] - float(difference.norm())) < 1e-4 + 1e-5*float(difference.norm())
    assert len(d['source_token_l2']) == len(ids_a)-start
    torch.testing.assert_close(torch.tensor([row['l2'] for row in d['source_token_l2']], dtype=torch.float64), difference.norm(dim=-1), atol=2e-4, rtol=2e-4)
    for row in d['layers']:
        if row['layer'] < layer:
            assert row['patched_l2'] < 1e-5
        else:
            assert abs(row['natural_l2'] - row['patched_l2']) < 4e-4 + 2e-4*row['natural_l2']
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in engine.model.modules())
    assert all(c['d'][0] < 1e-4 and c['d'][-1] > 1-1e-4 for c in result['curves'])
    print(f'PASS: {args.model}, {method}, layer {layer}, context {context.upper()}, batch {batch}: positions {start}–{len(ids_a)-1}; prefix unchanged, all suffix states verified, natural endpoints recovered.', flush=True)
    return result


a = 'The capital of France is Paris. The capital of Japan is'
b = 'The capital of France is Paris. The capital of Germany is'
result = check(a, b)
if not args.quick:
    other = check(a, b, context='b')
    reverse = check(b, a, context='b')
    for curve, same, opposite in zip(result['curves'], other['curves'], reverse['curves']):
        torch.testing.assert_close(torch.tensor(curve['d']), torch.tensor(same['d']), atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(torch.tensor(curve['d']), 1-torch.tensor(opposite['d']).flip(0), atol=3e-4, rtol=3e-4)
    check(a, b, layer=0, method='linear', batch=2)
    check(a, b, layer=len(blocks)-1)
    check('Japan is a country in Asia.', 'Germany is a country in Europe.', method='linear')
    check('The capital of Japan is Tokyo. The weather is cold today.',
          'The capital of Germany is Berlin. The weather is warm today.', method='linear')
    one = check('The house was big', 'The house was in')
    last_request = dict(model=args.model, sequence_a='The house was big', sequence_b='The house was in',
                        patch_layer=2, context='a', interpolation='slerp', steps=5)
    with patch.object(engine.hardware, 'batch_size', return_value=1):
        last = engine.run(last_request)
    for x, y in zip(one['curves'], last['curves']):
        torch.testing.assert_close(torch.tensor(x['d']), torch.tensor(y['d']), atol=1e-6, rtol=1e-6)
    with patch.object(engine, 'predict', side_effect=AssertionError('Unequal suffixes must be rejected before generation')):
        try:
            engine.run(dict(last_request, sequence_b='After the trip, the house was in', patch_position='different_suffix'))
            raise AssertionError('Unequal token counts must be rejected')
        except ValueError as error:
            assert 'equal token counts' in str(error) and 'Final token only' in str(error)
    unequal = engine.run(dict(last_request, sequence_b='After the trip, the house was in', patch_position='last_token'))
    assert unequal['settings']['patch_count'] == 1
    assert unequal['settings']['patch_start_a'] != unequal['settings']['patch_start_b']
    print('PASS: final-only special case and unequal-length fallback; no changes to saved collections.', flush=True)

if args.output:
    args.output.write_text(json.dumps(result, allow_nan=False))
