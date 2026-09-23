"""Verify incomplete-cache handling and a fresh, network-free model load."""
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from engine import Engine
from models import MODELS, cached_snapshot

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pythia-160m', choices=list(MODELS))
args = parser.parse_args()

with tempfile.TemporaryDirectory() as directory:
    cache = Path(directory)
    repo = cache / 'models--EleutherAI--pythia-160m'
    snapshot = repo / 'snapshots' / 'test-revision'
    snapshot.mkdir(parents=True)
    (repo / 'refs').mkdir()
    (repo / 'refs' / 'main').write_text('test-revision')
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        (snapshot / name).write_text('{}')
    assert cached_snapshot('pythia-160m', cache) is None
    (snapshot / 'model.safetensors.incomplete').write_bytes(b'partial')
    assert cached_snapshot('pythia-160m', cache) is None
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'a': 'model-00001.safetensors', 'b': 'model-00002.safetensors'}}))
    (snapshot / 'model-00001.safetensors').write_bytes(b'a')
    assert cached_snapshot('pythia-160m', cache) is None
    (snapshot / 'model-00002.safetensors').write_bytes(b'b')
    assert cached_snapshot('pythia-160m', cache) == snapshot
print('PASS: partial downloads and missing shards are not marked Saved.', flush=True)

assert cached_snapshot(args.model), f'Download {args.model} before running this check.'
engine = Engine()
messages = []
with patch('engine.snapshot_download', side_effect=AssertionError('Attempted download')), \
     patch('socket.socket.connect', side_effect=AssertionError('Attempted network connection')):
    engine.load(args.model, lambda message, _: messages.append(message))
assert len(engine.architecture()[0]) == MODELS[args.model]['layers']
assert any('No download needed' in message for message in messages)
assert engine.model.config._commit_hash == cached_snapshot(args.model).name
print(f'PASS: a fresh engine loads saved {args.model} with network calls blocked.', flush=True)
