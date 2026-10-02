# Cumulative arc length validation — 2026-09-27

Implementation is complete. Numerical, real-model, HTTP, persistence/export,
and frontend component checks passed. **Browser interaction and visual checks
remain blocked**, as detailed below; component checks are not a substitute.

## Changes

- `trajectory.py` streams actual final-token vectors from each displayed
  residual readout and logits. CPU float64 segment norms and compensated sums
  include batch transitions and exclude padded samples. No complete vector
  trajectory is retained in production. Inference passes are reused.
- Schema 4 adds c, per-step and cumulative lengths, total length, statuses,
  reasons and definition metadata. Existing d fields and flat d summaries
  keep their meanings. Zero measured length makes c undefined (no positive
  cutoff); the existing absolute endpoint tolerance for d remains `1e-8`.
- The GUI places c above d, with matching columns, raw totals, metric-specific
  tooltips/axes, c-first summaries, and a sample inspector for both metrics.
  The c information control supports hover, focus, click/touch and Escape.
  New library previews use c; legacy previews explicitly identify d.
- History, Examples, JSONL and CSV preserve the extended data. Old records
  remain unchanged and require a manual rerun to obtain c.
- `--data-dir` supports isolated application tests. `.gitignore` also ignores
  the existing `.venv` symlink to the named `plateau` Conda environment.

## Commands run successfully

Working directory: `/Users/billyli/plateau-lab`. The existing ARM64 Python 3.12,
PyTorch 2.8.0 and Transformers 4.51.3 environment was reused. No dependencies
were reinstalled or upgraded. No checkpoint was initially cached, so the small
Pythia 70M checkpoint was downloaded once with:

```sh
.venv/bin/python prepare_model.py pythia-70m
```

All of the following completed successfully (affected checks were rerun after fixes):

```sh
.venv/bin/python check_trajectory.py
.venv/bin/python check_math.py
.venv/bin/python check_exports.py
.venv/bin/python check_hardware.py
.venv/bin/python check_cache.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_arc.py --model pythia-70m --output /Users/billyli/.codex/tmp/plateau-arc-validation/real-result.json
PLATEAU_DEVICE=cpu .venv/bin/python check_arc.py --model pythia-70m
PLATEAU_DEVICE=mps .venv/bin/python check_arc.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_inference.py --model pythia-70m --oom
PLATEAU_DEVICE=cpu .venv/bin/python check_suffix.py --model pythia-70m
PLATEAU_DEVICE=cpu .venv/bin/python check_l2.py --model pythia-70m
.venv/bin/python check_app.py --model pythia-70m
node --check web/app.js
node check_ui.js
node check_ui.js /Users/billyli/.codex/tmp/plateau-arc-validation/real-result.json
.venv/bin/python -m py_compile engine.py trajectory.py app.py check_arc.py check_trajectory.py
git diff --check
```

## Results

- **10 numerical tests:** uniform straight paths; repeated points; bends and
  backtracking; nonconstant closed paths with undefined d; stationary paths;
  tiny motion; every batch split and empty batches; compensated accumulation;
  nonfinite metric serialization; unchanged ordinary d arithmetic.
- **9 HTTP export/persistence tests** and **9 hardware tests** passed, along
  with interpolation math and offline cache reuse checks.
- **Real-model independent capture:** all displayed readouts for Linear and
  SLERP, CPU batches 1/2/4 (including padding) and production MPS batch 1.
  Expected lengths use Python `math.dist` and `math.fsum` on separately captured
  vectors, compared to c/raw lengths at `atol=1e-9, rtol=1e-10`. Cross-batch
  model results agree within `2e-4`; independent ordinary d agrees within
  `2e-6`. Equal endpoints and mixed stationary/valid readouts complete safely.
- **Pre-change comparison:** a separate Python check loaded the original
  engine saved at `/Users/billyli/.codex/tmp/plateau-arc-validation/engine_before.py`
  and the changed engine. Pythia 70M, CPU, 21 samples, both Linear and SLERP:
  d arrays, predictions, path tokens and endpoint-L2 records matched exactly.
- **OOM and cancellation:** failure injected after partial measurement;
  retries remove hooks, restart sums and previous vectors, and match clean runs
  for c, d, step lengths and cumulative/total lengths.
- **Real CLI/HTTP app:** `check_app.py` starts a subprocess on a dynamic port
  with a temporary collection and runs Pythia 70M on MPS. Both interpolation
  methods passed run/poll/save/reload and selected CSV/JSONL checks. Legacy
  bytes were preserved and server logs had no tracebacks. Temporary collections
  were cleaned up; the user's collection directory was not populated or edited.
- **Frontend component checks:** c/d markup, labels, sample values, raw totals,
  summaries, legacy and undefined rendering, help event handlers, and full
  result restoration from real-model/legacy fixtures passed in a Node VM with
  a DOM stub. This does not exercise a browser rendering engine.

The MPS check initially caught a combined CPU/float64 conversion that MPS
rejects. It was fixed by moving tensors to CPU before converting to float64;
CPU and MPS checks then passed. An identical-source test was corrected to
recognize actual floating-point movement rather than assuming exact stationarity.

## Browser checks — blocked, not passed

An isolated app was launched with:

```sh
./start.sh --port 8767 --data-dir /Users/billyli/.codex/tmp/plateau-arc-validation/ui-data
```

The documented browser tool attempted to open `http://127.0.0.1:8767/` and
refused access because **the admin-enforced browser policy could not be
verified**. No security workaround was attempted. The isolated server was
stopped after backend testing, and the normal app was restarted.

These requested checks still need a browser session with working policy verification:

1. Run and visually confirm the c row above matching d charts.
2. Show the definition by actual mouse hover, keyboard focus, and touch/click.
3. Inspect samples and point tooltips; verify labels and raw totals.
4. Save, reload, reopen History/Examples, and download CSV/JSONL through the UI.
5. Open a legacy d-only record and verify its notice and preview visually.
6. Check narrow-screen wrapping/order and browser console errors.

Only Pythia 70M was used for real-model validation of this change. The existing
GPT-2 and Qwen model choices and intervention logic were not changed.
