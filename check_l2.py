"""Compare displayed L2 measurements with independent hidden-state outputs; no saved data writes."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from engine import Engine
from models import MODELS, cached_snapshot


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pythia-160m', choices=list(MODELS))
parser.add_argument('--quick', action='store_true', help='Check one middle-layer intervention.')
parser.add_argument('--output', type=Path, help='Optional test result fixture outside the saved collections.')
args = parser.parse_args()
assert cached_snapshot(args.model), 'Cache the model before running this offline check.'
engine = Engine()
engine.load(args.model, lambda *_: None)
blocks, _ = engine.architecture()
kind = engine.model.config.model_type
normalization = (engine.model.transformer.ln_f if kind == 'gpt2' else
                 engine.model.gpt_neox.final_layer_norm if kind == 'gpt_neox' else engine.model.model.norm)


@torch.inference_mode()
def states(text, intervention=None):
    """Use model-provided hidden_states; recover pre-normalization final state independently."""
    final = []
    handles = [normalization.register_forward_pre_hook(lambda _, inputs: final.append(inputs[0][0, -1].clone()))]
    if intervention is not None:
        layer, vector = intervention
        def replace(_, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            value = hidden.clone()
            value[0, -1] = vector
            return (value,) + output[1:] if isinstance(output, tuple) else value
        handles.append(blocks[layer].register_forward_hook(replace))
    try:
        ids = engine.tokenizer.encode(text, add_special_tokens=False)
        output = engine.model(torch.tensor([ids], device=engine.device), output_hidden_states=True, use_cache=False)
        # hidden_states[0] is embedding; the last item has already been normalized.
        return torch.stack([h[0, -1] for h in output.hidden_states[1:-1]] + [final[0]])
    finally:
        for handle in handles:
            handle.remove()


def l2(a, b):
    # Independent double-precision reduction over the original float32 states.
    return torch.linalg.vector_norm(a.detach().cpu().double() - b.detach().cpu().double(), dim=-1)


a = 'The capital of France is Paris. The capital of Japan is'
b = 'After the geography lesson, the capital of Germany is'
natural_a, natural_b = states(a), states(b)
expected_natural = l2(natural_a, natural_b)
middle = len(blocks) // 2
cases = [(middle, 'b', 1)] if args.quick else [(0, 'a', 1), (middle, 'a', 2), (middle, 'b', 1), (len(blocks)-1, 'a', 1)]
for layer, context, batch in cases:
    request = dict(model=args.model, sequence_a=a, sequence_b=b, patch_layer=layer,
                   context=context, interpolation='linear', steps=5)
    # Exercises both reference-forward paths independently of auto batch heuristics.
    with patch.object(engine.hardware, 'batch_size', return_value=batch):
        result = engine.run(request)
    distances = result['l2_distances']
    assert distances['position'] == 'last_token' and distances['representation'] == 'resid_post'
    assert [row['layer'] for row in distances['layers']] == list(range(len(blocks)))
    actual_natural = torch.tensor([row['natural_l2'] for row in distances['layers']], dtype=torch.float64)
    actual_patched = torch.tensor([row['patched_l2'] for row in distances['layers']], dtype=torch.float64)
    text = a if context == 'a' else b
    expected_patched = l2(states(text, (layer, natural_a[layer])), states(text, (layer, natural_b[layer])))
    torch.testing.assert_close(actual_natural, expected_natural, atol=3e-4, rtol=2e-4)
    torch.testing.assert_close(actual_patched, expected_patched, atol=3e-4, rtol=2e-4)
    torch.testing.assert_close(torch.tensor(distances['source_l2'], dtype=torch.float64), expected_natural[layer], atol=3e-4, rtol=2e-4)
    assert torch.all(actual_patched[:layer] < 1e-5)
    assert abs(float(actual_patched[layer]) - distances['source_l2']) < 1e-4
    assert torch.isfinite(actual_natural).all() and torch.isfinite(actual_patched).all()
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in engine.model.modules())
    for curve in result['curves']:
        assert curve['d'][0] < 1e-4 and curve['d'][-1] > 1-1e-4
    print(f'PASS: {args.model}, layer {layer}, context {context.upper()}, batch {batch}: every layer matches independent L2; source={distances["source_l2"]:.6f}.', flush=True)

if not args.quick:
    shared = engine.run(dict(model=args.model, sequence_a='The house was big', sequence_b='The house was in',
                             patch_layer=0, context='a', interpolation='linear', steps=5))
    for row in shared['l2_distances']['layers']:
        assert abs(row['natural_l2'] - row['patched_l2']) <= 3e-4 + 2e-4*row['natural_l2']
    print('PASS: matching prefixes reproduce natural distances at and after interpolation.', flush=True)

if args.output:
    args.output.write_text(json.dumps(result, allow_nan=False))
print('L2 checks passed; saved collections were not modified.', flush=True)
