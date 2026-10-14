# One Workbench, local and hosted

Both default servers serve `web/` at `/`. The classic Workbench layout is the base of this shared interface. `/classic/` bookmarks redirect to `/`; no second GUI is maintained or deployed.

| Workflow | Local | Hosted |
| --- | --- | --- |
| Prompts, swap, starter pairs, layer slider, SLERP/Linear, token scope, fixed context and sample count | Same controls | Same controls |
| Natural continuations | Three words, with word-boundary look-ahead | Same convention and 48-token limit |
| Tokenization preview and generated token details | Available | Available |
| Raw source L2, per-token source distances, original/patched per-layer plot and table, L2 CSV | Available | Available |
| Representative c(t) above d(t), definitions, summaries and raw path lengths | Available | Available |
| Inspect sample, c/d values, cumulative L2 and original next token | Available | Available |
| Next 3 tokens × top 3 candidates | Opt-in before running; off by default | Same; no extra continuation forwards when off |
| Additional all-layer metrics, including relative angle | Recorded and exportable | Recorded and exportable |
| Examples, categories, notes, updates, search and keyboard/card selection | Disk collections plus browser imports | Private browser collections |
| Automatic History, reload/reopen, cancellation and active-job recovery | Available | Available for this browser tab's job |
| Selected JSONL, classic CSV, all-layer CSV, full collection JSON and imports | Available | Available |
| Legacy results and unavailable models | Inspect/export original records; choose an available model to rerun | Same |

Both modes use the classic representative c(t) row above the matching d(t) row; there is no manual display-layer selector. The interpolation start-layer slider is retained. Hosted mode combines the local model catalog with NDIF deployments, marking unsupported or stopped choices **Unavailable on remote**. Shared checkpoints use the same model IDs; older hosted repository IDs remain accepted. Precision and compute depend on the execution backend. Hosted inference uses an NDIF key or lab access code. Hosted storage is private to the browser and survives reloads when browser storage is enabled; export JSON/JSONL for backups and moving devices. Existing hosted History is preserved during the browser database upgrade. A storage failure displays a session-only warning and keeps new records exportable until the page is closed.

The stdlib fallback `python app.py` also serves this interface, with its original representative-only engine and no live token-preview endpoint. Use `./start.sh` for the full local Workbench.

Validation: `npm test` exercises the classic plots/help/inspector, matrix opt-in and saved matrices, remote run/poll/autosave, annotations, import/export, hidden selections, disabled and legacy models, credential isolation and IndexedDB migration/failure behavior. Python checks cover routing, model availability, local collection compatibility, shared word boundaries, independent arc measurements, and local/nnsight parity with cached Pythia. Matrix off/on checks count actual forwards/traces and compare c/d measurements. DOM harness checks do not verify browser layout; live authenticated NDIF transport requires a credentialed test.
