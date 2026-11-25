# Built-in blog examples

The top-of-page selector and the **Blog examples** sidebar button expose a curated, read-only collection, independent of personal Examples and History. Selecting an entry previews real saved measurements without inference, credentials, tokenizer requests, or model downloads. The gallery loads from static files even when the inference API is unavailable. No write-up images or reconstructed curves are bundled.

**Copy example link** creates a URL such as `/#example=copying-vs-recall`. Opening it selects that preview without adding anything to the reader's collections. **Open saved experiment** restores the complete record in the Workbench once the Workbench finishes connecting. The model does not have to be available for inference. **Save example** then keeps an annotated private copy. **Download original JSON** downloads the bundled record. User edits never change the bundled source.

## Included measurements

The catalog contains six complete examples. It was assembled from the supplied 31-page `Write-up.pdf` and three original H6 exports. Source fingerprints, record IDs, asset hashes, and page references are in `web/blog-examples.json`; original local filesystem paths are not published.

- **Figure 1, page 2:** two new GPT-2 Large measurements of `The house is big` versus `The house is in` and `The house is large`. Both use SLERP after block 0, final-token interpolation, fixed context A, and 101 samples. The preview defaults to the final residual readout. Figure 1 is a conceptual illustration; these measurements do not claim to recover data behind that illustration.
- **Figure 11 / H5, page 14:** GPT-2 Large with interpolation after block 0, comparing `The capital of France is Paris. The capital of Japan` with `The capital of France is Paris. The capital of Germany`. The run uses SLERP, fixed context A, 101 samples, a three-token prediction matrix, and a separately measured delayed-logit curve after the forced continuation ` is`. This is the only H5 entry in the gallery.
- **Figure 12a / H6, page 16:** copying vs. factual recall, Gemma-2-9B-IT, after layer 10, original record `925f106eb4534a57ad2324a8f29d6c55`.
- **Figure 12b / H6, page 16:** opposite counting operations, Gemma-2-9B-IT, after layer 3, original record `24293194ccee4bb3bc006d5deeeafa56`.
- **Figure 12c / H6, page 16:** Hamlet vs. Macbeth, Gemma-2-9B-IT, after layer 0, original record `8d29ecddf6a146e9bd6c3772aedded48`.

The three H6 records use 41 SLERP samples, suffix interpolation, and fixed context A. Their original objects are preserved, including tokenization, predictions, all-layer effects, c/d arrays, L2 distances, runtime metadata, annotations, and timestamps. The three displayed local reruns retain their model revision, device, dtype, settings, and measured arrays. The gallery applies no smoothing, resampling, or reconstruction of c from d.

## Immediate and delayed readouts

The original prompts end at the country token. Their ordinary Workbench curves measure that final input position, whose next-token prediction is ` is`. A separate `blog_readouts.delayed_logits` object records a second forward experiment: append the shared token ` is`, patch only the original country position, and measure the following-token logits. The gallery defaults to this delayed readout, and the full JSON retains both measurements.

Delayed c(t) uses actual successive logit-vector distances, accumulated on CPU in float64. Delayed d(t) references the measured endpoints of the same fixed-context intervention. The record also stores top-1 tokens and probabilities at each sample, patch and measurement positions, forced token IDs, and differences between patched and natural endpoint logits. Patching block 0 only at the country position leaves the appended ` is` position's earlier computation in context A; its t=1 endpoint is therefore not assumed to equal a natural context-B forward pass. The recorded endpoint gap makes this distinction explicit.

## GPT-2 Large verification of Figure 11

The subsequent GPT-2 Large block-0 rerun matches the displayed values in Figure 11 and the rounded endpoint JSD values in the prose:

- First sampled c >= 0.1 at t = 0.36, first c >= 0.9 at t = 0.62: sampled transition width **0.26**. Linearly interpolating the crossing locations gives **0.261298**. These are two threshold conventions applied to the same measured curve.
- First sampled c >= 0.5 at t = **0.49**, with c = **0.529592**; c at t = 0.50 is **0.581129**. The figure labels these 0.53 and 0.58.
- Last Tokyo at t = **0.49**, first Berlin at t = **0.50**, matching the annotated token switch.
- Natural endpoint JSD: immediate **0.001445393** bits and delayed **0.994498880** bits, rounding to the prose's **0.0014 / 0.9945**.

This is strong evidence that the figure describes **GPT-2 Large, block-0 output**, rather than Qwen3 at its embedding. It verifies the published scalar annotations; the original c sample array was not available, so equality of every unpublished sample cannot be established. The fixed-context B endpoint differs from a natural context-B forward by about 0.01848 in maximum absolute logit value; the record preserves that intervention detail. JSD values above compare natural endpoints, as in the archived script, not patched endpoints.

## Archived reproduction checks

The Qwen and GPT-2 embedding comparisons are excluded from the gallery. Their JSON records remain available for provenance and can be rerun explicitly; they are not H5 presentation options. Qwen's interpolated c widths are approximately 0.479 (block 0) and 0.496 (embedding), and neither reproduces Figure 11. The GPT-2 token-plus-position embedding comparison has interpolated c width **0.236523** and first Berlin at t = **0.49**. GPT-2 adds a learned positional embedding before block 0. Plateau Lab's saved embedding convention patches the sum at `transformer.drop`; it differs from the old archive's raw `transformer.wte` intervention.

A separate rerun of the original archived raw-token-embedding script, using the same cached GPT-2 Large checkpoint with float32 inference on MPS, reproduces the archive's **d width 0.28** and all 101 immediate/delayed top-1 tokens. Its maximum absolute d difference from the archived JSON is **4.77e-7**. Measuring c directly from those new logits gives interpolated c width **0.235630**. The old d result is not the current figure's c result. The measured arrays and source fingerprints for this cross-check are in [gpt2-raw-embedding-check.json](validation/gpt2-raw-embedding-check.json).

## Reproduce the local cases

From the repository, using its local inference environment:

```sh
.venv/bin/python rerun_blog_examples.py --cache-dir .model-cache
```

The default runs the three curated local examples. Use repeated `--case` flags to select a subset, `--steps` to change the sample count, and `--output` to write elsewhere. For H5 alone, use `--case japan-germany-gpt2-block0`. Country-prompt cases include the separate delayed-logit measurements. Explicit comparison cases save their JSON records without adding entries to the gallery. Curated runs update complete records and catalog hashes; the script does not write to personal collections. Cached checkpoints are reused. A new run can differ with model revision, hardware, or numerical precision.

## Llama experiments awaiting access

The following Llama 3.1 8B cases are extracted as runnable configurations but are **not yet measured or included in the gallery**:

- H4 / page 13: soccer vs. geography, `The city of Hamburg`.
- H4 / page 13: art vs. geography, `Florence`, highlighted in the 50-pair figure.
- Appendix / page 30: movies vs. geography, `The town of Hollywood`.

They require a suitable remote backend and model access. Once a Plateau Lab API is available, supply its URL and credentials through the environment (never commit credentials):

```sh
python rerun_blog_examples.py --api-url https://YOUR-PLATEAU-LAB-HOST \
  --case hamburg-context --case florence-context --case hollywood-context
```

The API accepts `PLATEAU_BLOG_NDIF_KEY` or `PLATEAU_BLOG_ACCESS_CODE` from the environment. The script sends it only in request headers. These configurations follow the general method of SLERP after block 0 over the differing suffix. Individual reruns do not reproduce the whole 50-pair aggregate. Remote execution has not been verified without restored access.

## Add a complete example

1. Export the intended public experiment as JSON and check its prompts and annotations.
2. Save the complete record as `web/blog-record-<stable-slug>.json`, preserving its measured arrays and settings.
3. Add a `kind: "record"` entry to `web/blog-examples.json`, including a stable ID, title, reference, summary, record filename, source record ID, provenance, and SHA-256 of the exact JSON bytes. `preview_readout` may select a recorded curve or extra blog readout.
4. Update the catalog assertions in `check_blog.js` for the newly curated entries. Run `npm test` and `python -m unittest check_blog`, then inspect the preview, direct link, and Workbench result.

The Space workflow and Dockerfile already copy all of `web/`. No private `data/` directory, model weights, PDF, or screenshot is bundled. The stdlib server, FastAPI server, and static frontend serve the same flat, relative asset paths. Preparing these files does not by itself publish a deployment.
