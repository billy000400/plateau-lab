# One Workbench, local and hosted

Both default servers serve `web/` at `/`. The classic Workbench layout is the base of this shared interface. `/classic/` bookmarks redirect to `/`; no second GUI is maintained or deployed.

| Workflow | Local | Hosted |
| --- | --- | --- |
| Prompts, swap, starter pairs, layer slider, SLERP/Linear, token scope, fixed context and sample count | Same controls | Same controls |
| Natural continuations | Three words, with word-boundary look-ahead | Same convention and 48-token limit |
| Tokenization preview and generated token details | Available | Available |
| Raw source L2, per-token source distances, original/patched per-layer plot and table, L2 CSV | Available | Available |
| Representative c(t) above d(t), definitions, summaries and raw path lengths | Available | Available |
| Inspect sample, c/d values, cumulative L2 and 3 × 3 token matrix | Available | Available |
| All-layer metric selection, presets, token overlay and across-layer plot | Available | Available |
| Examples, categories, notes, updates, search and keyboard/card selection | Disk collections plus browser imports | Private browser collections |
| Automatic History, reload/reopen, cancellation and active-job recovery | Available | Available for this browser tab's job |
| Selected JSONL, classic CSV, all-layer CSV, full collection JSON and imports | Available | Available |
| Legacy results and unavailable models | Inspect/export original records; choose an available model to rerun | Same |

The execution environment determines the model catalog, precision and compute label. Hosted inference uses an NDIF key or lab access code. Hosted storage is private to the browser and survives reloads when browser storage is enabled; export JSON/JSONL for backups and moving devices. Existing hosted History is preserved during the browser database upgrade. A storage failure displays a session-only warning and keeps new records exportable until the page is closed.

The stdlib fallback `python app.py` also serves this interface, with its original representative-only engine and no live token-preview endpoint. Use `./start.sh` for the full local Workbench.

Validation: `npm test` exercises the classic plots/help/inspector, all-layer rendering, remote run/poll/autosave, annotations, import/export, hidden selections, unavailable-model restoration, credential isolation and IndexedDB migration/failure behavior. Python checks cover routing in both modes, local collection compatibility, shared word boundaries, independent arc measurements, and local/nnsight parity with cached Pythia. DOM harness checks do not verify browser layout; live authenticated NDIF transport requires a credentialed test.
