#!/usr/bin/env python3
"""Measure the blog's prompt pairs and bundle complete results for offline viewing."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
CASES = {
    'big-in': {
        'title': 'A sharp transition: big → in', 'reference': 'Figure 1 · Introduction', 'source_page': 2,
        'summary': 'A measured rerun of the introductory big → in prompt pair.', 'preview_readout': '35',
        'request': {'model': 'gpt2-large', 'sequence_a': 'The house is big', 'sequence_b': 'The house is in',
                    'patch_layer': 0, 'patch_position': 'last_token'},
        'note': 'New measurement of the prompt pair illustrated conceptually in Figure 1. GPT-2 Large, after block 0; these are measured curves, not the schematic figure.'},
    'big-large': {
        'title': 'A semantic neighbor: big → large', 'reference': 'Figure 1 · Introduction', 'source_page': 2,
        'summary': 'A measured rerun of the introductory big → large prompt pair.', 'preview_readout': '35',
        'request': {'model': 'gpt2-large', 'sequence_a': 'The house is big', 'sequence_b': 'The house is large',
                    'patch_layer': 0, 'patch_position': 'last_token'},
        'note': 'New measurement of the prompt pair illustrated conceptually in Figure 1. The observed curve is retained even if it differs from the conceptual illustration.'},
    'japan-germany': {
        'title': 'Japan → Germany: Qwen3, block 0', 'reference': 'Figure 11 · H5 comparison', 'source_page': 14,
        'summary': 'Japan → Germany with the next-three-token prediction matrix recorded at every interpolation sample.',
        'request': {'model': 'qwen3-0.6b-base',
                    'sequence_a': 'The capital of France is Paris. The capital of Japan',
                    'sequence_b': 'The capital of France is Paris. The capital of Germany',
                    'patch_layer': 0, 'patch_position': 'last_token', 'sample_token_matrix': True},
        'note': 'Qwen3 0.6B Base comparison after block 0. This rerun does not reproduce Figure 11: its delayed c transition width is about 0.479, versus 0.26 in the figure. The GPT-2 Large block-0 comparison matches the reported values. Main Workbench curves measure the final input position; select delayed logits here for the readout after " is".'},
    'japan-germany-embedding': {
        'title': 'Japan → Germany: Qwen3, embedding', 'reference': 'H5 · Prose comparison', 'source_page': 14,
        'summary': 'The same country-token experiment at the embedding location specified in the prose.',
        'request': {'model': 'qwen3-0.6b-base',
                    'sequence_a': 'The capital of France is Paris. The capital of Japan',
                    'sequence_b': 'The capital of France is Paris. The capital of Germany',
                    'patch_layer': -1, 'patch_position': 'last_token', 'sample_token_matrix': True},
        'note': 'Qwen3 0.6B Base comparison at the embedding specified in the prose. This rerun does not reproduce Figure 11: its delayed c transition width is about 0.496, versus 0.26 in the figure. The GPT-2 Large block-0 comparison matches the reported values. The Workbench slider still selects hidden layers for a new run.'},
}

for slug, layer, location in [
    ('japan-germany-gpt2-block0', 0, 'block 0'),
    ('japan-germany-gpt2-embedding', -1, 'token + position embedding'),
]:
    CASES[slug] = {
        'title': f'Japan → Germany: GPT-2 Large, {location}',
        'reference': 'Figure 11 · Model comparison', 'source_page': 14,
        'summary': 'A GPT-2 Large comparison using the same country-token prompts and delayed-logit readout.',
        'request': {**CASES['japan-germany']['request'], 'model': 'gpt2-large', 'patch_layer': layer},
        'note': f'GPT-2 Large comparison at {location}, using SLERP in fixed context A. '
                'The delayed curve patches only the original country position and measures logits after a forced " is". '
                'This is a model comparison, not a Qwen result.'
    }
    if layer == 0:
        CASES[slug]['note'] += ' The 101-sample check matches Figure 11 at its displayed precision: c width 0.26, first Berlin at t=0.50, and natural endpoint JSDs 0.0014 / 0.9945 bits.'
    else:
        CASES[slug]['note'] += ' This embedding includes the positional embedding, unlike the raw token embedding in the older archived script. Its measured c width is about 0.237; it does not match Figure 11.'

# Keep reproduction diagnostics available without adding them to the public gallery.
for slug in ('japan-germany', 'japan-germany-embedding', 'japan-germany-gpt2-embedding'):
    CASES[slug]['comparison_only'] = True

CASES['japan-germany-gpt2-block0'].update(
    reference='Figure 11 · H5',
    summary='Both prompts predict “ is” next, while their following predictions switch from Tokyo to Berlin.',
    note='Interpolate the final country-token residual state after block 0 of GPT-2 Large using SLERP. '
         'After appending “ is”, the delayed logits show a sampled c transition width of 0.26, '
         'with the first Berlin prediction at t=0.50.')

for slug, title, a, b, page in [
    ('hamburg-context', 'Soccer vs. geography: Hamburg',
     "Let's talk about soccer: The city of Hamburg", "Let's talk about geography: The city of Hamburg", 13),
    ('florence-context', 'Art vs. geography: Florence',
     'In the context of art, Florence', 'In the context of geography, Florence', 13),
    ('hollywood-context', 'Movies vs. geography: Hollywood',
     "Let's talk about movies: The town of Hollywood", "Let's talk about geography: The town of Hollywood", 30),
]:
    CASES[slug] = {'title': title, 'reference': 'H4 · Contextual clues', 'source_page': page,
        'summary': 'An earlier contextual clue changes while the final token remains the same.',
        'requires_remote': True,
        'request': {'model': 'meta-llama/Llama-3.1-8B', 'sequence_a': a, 'sequence_b': b,
                    'patch_layer': 0, 'patch_position': 'different_suffix'},
        'note': 'New individual-pair measurement on Llama 3.1 8B, using the general methodology: SLERP after the first block, patching the entire differing suffix. This is not the original 50-pair aggregate.'}


def run_remote(api_url, request):
    """Use an explicitly supplied Plateau Lab API; credentials never enter records."""
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get('PLATEAU_BLOG_NDIF_KEY')
    code = os.environ.get('PLATEAU_BLOG_ACCESS_CODE')
    if key:
        headers['X-NDIF-Key'] = key
    elif code:
        headers['X-Access-Code'] = code
    def call(path, body=None):
        req = urllib.request.Request(api_url.rstrip('/')+path,
            data=None if body is None else json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    job_id = call('/api/run', request)['id']
    try:
        for _ in range(1800):
            job = call('/api/jobs/'+job_id)
            if job['status'] == 'done':
                return job['result']
            if job['status'] in ('error', 'cancelled'):
                raise RuntimeError('Remote experiment did not complete; inspect its status in the Workbench.')
            time.sleep(1)
        raise TimeoutError('Remote experiment exceeded 30 minutes.')
    except (KeyboardInterrupt, TimeoutError):
        call('/api/jobs/'+job_id+'/cancel', {})
        raise


def measure_delayed_logits(runner, result):
    """Measure the next readout after a forced ' is', patching only the country token."""
    import torch
    import math
    from engine import interpolate
    from plateau.core.trajectory import TrajectoryReadout
    from plateau.core.math import transition_width
    tokenizer, model = runner.tokenizer, runner.model
    prompts = [result['sequence_a'], result['sequence_b']]
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    suffix = tokenizer.encode(' is', add_special_tokens=False)
    assert len(suffix) == 1 and len(ids[0]) == len(ids[1]) and ids[0][:-1] == ids[1][:-1]
    blocks, embedding = runner.architecture()
    layer = result['settings']['patch_layer']
    module = embedding if layer == -1 else blocks[layer]
    position = len(ids[0])-1
    sources = []
    def capture(_, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sources.append(hidden[0, position].detach().clone())
    def forward(tokens):
        return model(torch.tensor([tokens], device=runner.device), use_cache=False).logits[0, -1].float()
    with torch.inference_mode():
        handle = module.register_forward_hook(capture)
        try:
            immediate = [forward(tokens).cpu() for tokens in ids]
        finally:
            handle.remove()
        natural = [forward(tokens+suffix).cpu() for tokens in ids]
        def patch(_, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            hidden[:, position, :] = current
            return (hidden,)+output[1:] if isinstance(output, tuple) else hidden
        ts = result['curves'][0]['t']
        handle = module.register_forward_hook(patch)
        try:
            vectors = []
            for t in ts:
                current = interpolate(sources[0], sources[1], [t], result['settings']['interpolation'])[0]
                vectors.append(forward(ids[0]+suffix).cpu())
        finally:
            handle.remove()
        values = torch.stack(vectors)
        readout = TrajectoryReadout(values[[0, -1]])
        readout.append(values)
        measured = readout.finish()
        assert measured['c_status'] == 'ok'
        top_ids = values.argmax(-1)
        probabilities = torch.softmax(values, -1).gather(1, top_ids[:, None])[:, 0]
        def jsd_bits(logits):
            logp, logq = [torch.log_softmax(vector.double(), -1) for vector in logits]
            logm = torch.logaddexp(logp, logq)-math.log(2)
            return float(((logp.exp()*(logp-logm)).sum() +
                          (logq.exp()*(logq-logm)).sum())/(2*math.log(2)))
    crossings = {str(q): next(t for t, c in zip(ts, measured['c']) if c >= q) for q in (.1, .5, .9)}
    measured.update(key='delayed_logits', title='Logits after forced “ is”', t=ts,
                    forced_continuation=' is', forced_token_ids=suffix, patch_position=position,
                    measurement_position=position+1, context='a', patch_layer=layer,
                    definition='Patch only the original country position; append the shared token " is" and measure the next-token logits. Reference endpoints use the same fixed context A and intervention.',
                    patched_vs_natural_max_abs_logit_gap={'A':float((values[0]-natural[0]).abs().max()),
                                                        'B':float((values[-1]-natural[1]).abs().max())},
                    natural_endpoint_jsd_bits={'immediate': jsd_bits(immediate), 'delayed': jsd_bits(natural)},
                    c_first_crossings=crossings,
                    transition_width_sample_grid=crossings['0.9']-crossings['0.1'],
                    predictions=[{'t':t, 'token_id':int(token), 'token':tokenizer.decode([int(token)]),
                                  'probability':float(p)} for t,token,p in zip(ts,top_ids,probabilities)],
                    transition_width=transition_width(ts,measured['c']))
    result['blog_readouts'] = {'delayed_logits': measured}
    return measured


def save_record(slug, case, result, output):
    """Write actual results and their provenance; never infer measurements from images."""
    result['id'] = uuid.uuid4().hex
    result['tag'] = 'Blog example'
    result['notes'] = case['note']
    result['blog_reproduction'] = {'example': slug, 'reference': case['reference'],
                                   'source_document': 'Write-up.pdf', 'source_page': case['source_page'],
                                   'note': case['note']}
    if result['model'].startswith('gpt2'):
        layer = result['settings']['patch_layer']
        result['blog_reproduction']['patch_module'] = (
            'transformer.drop output (token plus position embedding)' if layer == -1
            else f'transformer.h.{layer} output')
    filename = f'blog-record-{slug}.json'
    content = (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    (output/filename).write_bytes(content)
    if case.get('comparison_only'):
        return filename
    entry = {key: case[key] for key in ['title', 'reference', 'source_page', 'summary']}
    entry.update(id=slug, kind='record', record_file=filename, source_record_id=result['id'],
                 sha256=hashlib.sha256(content).hexdigest(), note=case['note'], provenance='rerun')
    if case.get('preview_readout'):
        entry['preview_readout'] = case['preview_readout']
    if 'blog_readouts' in result:
        entry['preview_readout'] = 'delayed_logits'
    manifest_path = output/'blog-examples.json'
    catalog = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'version': 1, 'examples': []}
    existing = next((i for i, e in enumerate(catalog['examples']) if e['id'] == slug), None)
    if existing is None:
        catalog['examples'].append(entry)
    else:
        catalog['examples'][existing] = entry
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(manifest_path)
    return filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=CASES, help='Repeat to select cases; default: curated local examples. Comparison cases save JSON only.')
    parser.add_argument('--steps', type=int, default=101)
    parser.add_argument('--cache-dir', type=Path, help='Reuse an existing model cache.')
    parser.add_argument('--api-url', help='Use an existing Plateau Lab API for remote cases. Keys come from PLATEAU_BLOG_NDIF_KEY or PLATEAU_BLOG_ACCESS_CODE.')
    parser.add_argument('--output', type=Path, default=ROOT/'web')
    args = parser.parse_args()
    selected = args.case or [slug for slug, case in CASES.items()
                             if not case.get('requires_remote') and not case.get('comparison_only')]
    if not args.api_url and any(CASES[slug].get('requires_remote') for slug in selected):
        parser.error('The Llama cases require an existing GPU-backed Plateau Lab API supplied with --api-url.')
    runner = None
    if not args.api_url:
        import engine
        import models
        if args.cache_dir:
            engine.CACHE = args.cache_dir.resolve()
            engine.cached_snapshot = partial(models.cached_snapshot, cache=engine.CACHE)
        runner = engine.Engine()
    for slug in selected:
        case = CASES[slug]
        request = {'steps': args.steps, 'interpolation': 'slerp', 'context': 'a', **case['request']}
        last = [-1]
        def progress(message, fraction):
            bucket = int((fraction or 0)*10)
            if fraction is None or bucket != last[0]:
                print(f'{slug}: {message}', flush=True)
                last[0] = bucket
        result = run_remote(args.api_url, request) if args.api_url else runner.run(request, progress, all_layers=True)
        if runner and slug.startswith('japan-germany'):
            delayed = measure_delayed_logits(runner, result)
            print(f'{slug}: measured delayed logits; c transition width {delayed["transition_width"]:.4f}', flush=True)
        if not all(curve.get('c_status') == 'ok' for curve in result['curves']):
            raise RuntimeError(f'{slug}: a representative c(t) curve is undefined; review before bundling.')
        filename = save_record(slug, case, result, args.output)
        print(f'Saved {filename}; {result["elapsed_seconds"]} seconds; endpoint predictions: '
              f'{result["experiment"]["natural_endpoint_next_tokens"]}', flush=True)


if __name__ == '__main__':
    main()
