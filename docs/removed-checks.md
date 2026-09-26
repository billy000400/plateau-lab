# Removed validation checks

The upstream repository shipped standalone `check_*.py` scripts (plain asserts / `unittest`, no pytest). They were removed from this fork while the app is restructured into a GitHub Pages frontend + NDIF-backed backend. The originals are available in upstream history (`git show upstream/main:check_math.py`, etc.). This note records what each one verified and whether it is worth re-implementing.

## Worth re-implementing

### `check_math.py`: experiment definitions (no model needed)
- Linear interpolation gives `d(t) = t` exactly.
- SLERP: exact endpoints; norm interpolates linearly; symmetric under A↔B with t→1−t; `d` flips to `1−d` when endpoints are swapped.
- Identical endpoints raise `ValueError` (undefined d).
- Tokenwise interpolation (`interpolate_tokens`) equals a per-position loop of `interpolate`, with one shared t; handles zero and parallel vectors; antiparallel vectors reject SLERP.
- `next_word_spans` word-boundary rule (suffix extending the last word isn't a new word; apostrophes stay inside words).

**Why:** cheap, fast, and pins down the core math that every result depends on. **When re-adding:** write the properties (endpoints, symmetry, finiteness, degenerate inputs) once and run them over every registered interpolation/metric, so new methods are covered automatically. Metrics should declare which properties they guarantee (bounded to [0,1], symmetric, …).

### `check_suffix.py`: the patch really does what's claimed (real model)
- In "first difference → end" mode, records the actual input to the next block and verifies every patched position equals the expected interpolated state, and the shared prefix is untouched.
- t=0 / t=1 reproduce natural A / B logits (within ~2e-3).
- Per-token and combined source L2 match an independent computation.
- Context A vs B gives the same curve; swapping A↔B mirrors it; final-token-only special case matches the old mode; unequal token counts are rejected before generation.

**Why:** it's the main guard against wrong hook placement or indexing, which yields plausible-looking but wrong curves. Very relevant for the nnsight port, where module paths / output tuple handling differ per architecture.

### `check_l2.py`: L2 per layer vs. independent reference (real model)
- Per-layer natural and patched L2 match distances computed from `output_hidden_states` (plus the pre-final-norm state), for several patch layers / contexts / batch sizes.
- Patched L2 is ~0 before the patch layer and equals source L2 at it; with a shared prefix, patched equals natural at and after the patch.

**Why:** same reason as `check_suffix`: verifies that "layer k output" means what we think for each architecture.

### `check_contexts.py`: curve sanity across contexts and layers (real model)
- d(t) finite, within [0,1], hits 0 and 1 at the endpoints; endpoint logit errors < 1e-3.
- Different prefixes/lengths; context A vs B; swapping sources with the context reverses d(t); patching the last layer yields only `[last_layer, logits]` curves.
- No hooks left on the model afterwards.
- Regression against old saved `data/runs` records (schema v1) if present.

**Why:** broad end-to-end sanity. Partly overlaps with `check_suffix`; could be merged into one backend-agnostic suite.

**Idea for all three real-model checks:** make them backend-agnostic (local torch vs nnsight) and run the local backend once to save reference outputs as fixtures, so CI can compare without a GPU or NDIF quota.

## Probably obsolete for the web app

### `check_exports.py`: HTTP export API (no model)
Ran `app.py` on a temp data dir. Selected JSONL/CSV export returns exactly the requested records in order; CSV keeps multi-line/unicode notes and mixes old/new record metadata; suffix-mode fields survive both formats; an empty/invalid/`all` selection returns 400 and never exports everything; an unknown id returns 404; bad format/scope returns 400.
**Status:** `app.py` is gone and export moves into the browser. The *cases* (subset ordering, old+new schema in one export, empty selection never exports all) are still worth testing in the frontend if export logic gets complex.

### `check_hardware.py`: device selection policy (mocked CUDA/MPS/ROCm)
Picks the GPU with most free memory, skips failing GPUs, honors `PLATEAU_DEVICE`, rejects bad overrides, labels ROCm, batch size scales with free memory, MPS batch 1 for Pythia/Qwen, OOM detection, CPU fallback only in auto mode.
**Status:** only relevant if a local-torch backend is kept for running models on the server's own hardware.

### `check_inference.py`: generation cache + OOM retry (real model)
KV-cached greedy generation matches uncached; later steps process one token; an injected OOM halves the batch, removes hooks, discards partial samples and matches a clean run.
**Status:** local-backend only. With NDIF, the equivalent concerns are request batching and retry, which are different code.

### `check_cache.py`: model download cache (real model)
Partial downloads / missing shards are not marked "Saved"; a saved model loads with network blocked.
**Status:** obsolete with NDIF (no local weights).
