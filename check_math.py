"""Focused checks for the experiment's numerical definitions; no download needed."""
import torch
from engine import interpolate, interpolate_tokens, relative_distance, next_word_spans

a, b = torch.tensor([1., 0., 0.]), torch.tensor([0., 2., 0.])
t = torch.linspace(0, 1, 11)
linear = interpolate(a, b, t, "linear")
assert torch.allclose(relative_distance(linear, a, b), t, atol=1e-6)
sphere = interpolate(a, b, t, "slerp")
assert torch.equal(sphere[0], a) and torch.equal(sphere[-1], b)
assert torch.allclose(sphere.norm(dim=-1), (1-t)*a.norm()+t*b.norm(), atol=1e-6)
assert torch.allclose(sphere, interpolate(b,a,1-t,"slerp"), atol=1e-6)
assert torch.allclose(relative_distance(sphere,b,a), 1-relative_distance(sphere,a,b), atol=1e-6)
try:
    relative_distance(sphere,a,a)
    raise AssertionError("Identical endpoints must not produce a curve")
except ValueError:
    pass
assert [m.group() for m in next_word_spans("It was big", "ger than my house.")] == ["than", "my", "house"]
assert [m.group() for m in next_word_spans("It was big", ", but not huge.")] == ["but", "not", "huge"]
assert [m.group() for m in next_word_spans("I said ", "don't go home yet")] == ["don't", "go", "home", "yet"]
print("PASS: LERP d(t), SLERP endpoints/norms/symmetry, undefined endpoints, complete-word boundaries.")

# Each position follows its own scalar-vector path, rather than one global
# spherical path across the flattened suffix. Include zero and parallel vectors.
torch.manual_seed(42)
source_a, source_b = torch.randn(7, 12), torch.randn(7, 12)
source_a[0] = 0
source_b[1] = 2 * source_a[1]
source_b[2] = source_a[2]
for method in ('linear', 'slerp'):
    actual = interpolate_tokens(source_a, source_b, t, method)
    expected = torch.stack([interpolate(a, b, t, method) for a, b in zip(source_a, source_b)], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert actual.shape == (len(t), 7, 12)
    assert torch.equal(actual[0], source_a) and torch.equal(actual[-1], source_b)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, interpolate_tokens(source_b, source_a, 1-t, method), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(interpolate_tokens(source_a[:1], source_b[:1], t, method)[:, 0], interpolate(source_a[0], source_b[0], t, method), atol=0, rtol=0)
try:
    interpolate_tokens(torch.eye(2), -torch.eye(2), t)
    raise AssertionError('Antiparallel token states must reject ambiguous SLERP')
except ValueError:
    pass
print('PASS: tokenwise Linear/SLERP, shared t, zero/parallel states, exact endpoints, reversal, and single-token compatibility.')
