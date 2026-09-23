"""Real-model checks for arbitrary prefixes and lengths; writes no examples."""
import argparse
import json
from pathlib import Path

import torch

from engine import Engine
from models import MODELS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gpt2-large', choices=list(MODELS))
args = parser.parse_args()

engine = Engine()


def check(a, b, context="a", patch_layer=0, steps=11):
    result = engine.run(dict(model=args.model, sequence_a=a, sequence_b=b,
                             context=context, patch_layer=patch_layer,
                             interpolation="slerp", steps=steps))
    assert result["settings"]["endpoint_reference"] == "patched_in_fixed_context"
    assert result["experiment"]["fixed_context"] == context.upper()
    assert all(p["word_count"] == 3 for p in result["predictions"])
    for curve in result["curves"]:
        values = torch.tensor(curve["d"])
        assert torch.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()
        assert values[0] < 1e-4 and values[-1] > 1 - 1e-4
    assert max(result["metrics"]["endpoint_max_abs_logit_error"].values()) < 1e-3
    assert result["metrics"]["patched_vs_natural_max_abs_logit_gap"][context.upper()] < 1e-3
    assert all(not module._forward_hooks for module in engine.model.modules())
    print(f"PASS: {context.upper()} context, layer {patch_layer}, lengths {result['experiment']['source_lengths']}", flush=True)
    return result


shared = check("The house was big", "The house was in", steps=41)
assert shared["experiment"]["shared_tokenized_prefix"]
assert max(shared["metrics"]["patched_vs_natural_max_abs_logit_gap"].values()) < 1e-3
for file in (Path(__file__).parent / "data" / "runs").glob("*.json"):
    old = json.loads(file.read_text())
    if (old.get("schema_version") == 1 and old["model"] == args.model and old["sequence_a"] == shared["sequence_a"]
            and old["sequence_b"] == shared["sequence_b"] and old["settings"]["steps"] == 41):
        for before, after in zip(old["curves"], shared["curves"]):
            assert torch.allclose(torch.tensor(before["d"]), torch.tensor(after["d"]), atol=2e-4)
        print("PASS: matching-prefix curves agree with the previous implementation.", flush=True)
        break

a, b = "I live in London", "After many years abroad, she returned to Berlin"
forward = check(a, b)
assert not forward["experiment"]["shared_tokenized_prefix"]
assert len(set(forward["experiment"]["source_lengths"])) == 2
reverse = check(b, a, context="b")
for f, r in zip(forward["curves"], reverse["curves"]):
    assert torch.allclose(torch.tensor(f["d"]), 1 - torch.tensor(r["d"]).flip(0), atol=2e-4)
print("PASS: swapping source states and preserving context reverses d(t).", flush=True)
check(a, b, context="b")
check(a, b, patch_layer=-1)
check(a, b, patch_layer=MODELS[args.model]['layers'] - 2)
final_patch = check(a, b, patch_layer=MODELS[args.model]['layers'] - 1)
assert [c['key'] for c in final_patch['curves']] == [str(MODELS[args.model]['layers'] - 1), 'logits']
print("All context checks passed; existing saved results were not modified.", flush=True)
