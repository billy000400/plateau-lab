"""Exercise selected exports over HTTP with temporary data; no model or user data touched."""
import csv
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import app
from trajectory import METRIC_DEFINITIONS


class QuietHandler(app.Handler):
    def log_message(self, *args):
        pass


class ExportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.original_data = app.DATA
        app.DATA = Path(cls.directory.name)
        cls.ids = [f'{i:032x}' for i in range(1, 4)]
        for folder in ('examples', 'runs'):
            for index, record_id in enumerate(cls.ids):
                record = {
                    'id': record_id, 'model': 'gpt2-large',
                    'sequence_a': 'The capital of Japan is', 'sequence_b': 'The capital of Germany is',
                    'input_tokens': [[{'id': 318, 'text': ' is'}]] * 2,
                    'predictions': [{'continuation': ' Tokyo.'}, {'continuation': ' Berlin.'}],
                    'settings': {'patch_layer': index, 'interpolation': 'linear'},
                    'notes': f'{folder}: line one, "quoted"\n第二行',
                    'curves': [{'title': 'Logits', 't': [0, .5, 1], 'd': [0, .2, 1]}],
                }
                # Include both older and newer record metadata in each selection.
                if index:
                    record['schema_version'] = 4
                    record['primary_metric'] = 'c'
                    record['metric_definitions'] = METRIC_DEFINITIONS
                    record['curves'][0].update(c=[0, .4, 1], step_lengths=[0, 2, 3],
                                              cumulative_length=[0, 2, 5], total_length=5,
                                              c_status='ok', d_status='ok',
                                              c_undefined_reason=None, d_undefined_reason=None)
                    record['settings'].update(context='b', prediction_cache=True, batch_size=1)
                    record['hardware'] = {'name': 'Test CPU'}
                    record['l2_distances'] = {
                        'source_l2': 2.5, 'patch_layer': index,
                        'layers': [{'layer': 0, 'natural_l2': 1.25, 'patched_l2': 0.0},
                                   {'layer': index, 'natural_l2': 2.5, 'patched_l2': 2.5}],
                    }
                if index == 2:
                    record['curves'].extend([
                        dict(title='Stationary readout', t=[0, .5, 1], d=[None]*3, c=[None]*3,
                             step_lengths=[0, 0, 0], cumulative_length=[0, 0, 0], total_length=0,
                             c_status='stationary', d_status='coincident_endpoints',
                             c_undefined_reason='No measured movement; c(t) undefined'),
                        dict(title='Closed path', t=[0, .5, 1], d=[None]*3, c=[0, .5, 1],
                             step_lengths=[0, 2, 2], cumulative_length=[0, 2, 4], total_length=4,
                             c_status='ok', d_status='coincident_endpoints'),
                    ])
                    record['settings'].update(patch_position='different_suffix', patch_start_a=0,
                                              patch_start_b=0, patch_count=2,
                                              interpolation_unit='per_token_shared_t', measurement_position='last_token')
                    record['input_tokens'] = [[{'id': 1, 'text': 'Japan'}, {'id': 318, 'text': ' is'}],
                                              [{'id': 2, 'text': 'Germany'}, {'id': 318, 'text': ' is'}]]
                app.write_json(app.DATA / folder / f'{record_id}.json', record)
        cls.before = {p: p.read_bytes() for p in app.DATA.rglob('*.json')}
        cls.server = app.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        try:
            assert cls.before == {p: p.read_bytes() for p in app.DATA.rglob('*.json')}
        finally:
            app.DATA = cls.original_data
            cls.directory.cleanup()

    def request(self, data=None, query=''):
        request = Request(self.url + '/api/export' + query,
                          data=None if data is None else json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.headers, response.read()

    def test_jsonl_subset_and_annotation_scope(self):
        for scope, folder in [('examples', 'examples'), ('history', 'runs')]:
            status, headers, body = self.request({'scope': scope, 'format': 'jsonl',
                                                 'ids': [self.ids[1], self.ids[0], self.ids[1]]})
            self.assertEqual(status, 200)
            records = [json.loads(line) for line in body.splitlines()]
            self.assertEqual([r['id'] for r in records], self.ids[1::-1])
            self.assertEqual(records, [app.read_json(app.DATA / folder / f'{i}.json') for i in self.ids[1::-1]])
            self.assertIn(f'plateau-{scope}-selected.jsonl', headers['Content-Disposition'])

    def test_csv_subset_preserves_curves_and_multiline_notes(self):
        status, headers, body = self.request({'format': 'csv', 'scope': 'examples', 'ids': self.ids[:2]})
        self.assertEqual(status, 200)
        self.assertIn('text/csv', headers['Content-Type'])
        rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
        self.assertEqual(len(rows), 6)
        self.assertEqual({r['id'] for r in rows}, set(self.ids[:2]))
        self.assertEqual([float(r['d']) for r in rows[:3]], [0, .2, 1])
        self.assertEqual(rows[0]['notes'], 'examples: line one, "quoted"\n第二行')
        self.assertEqual(rows[0]['hardware_name'], '')
        self.assertEqual(rows[3]['hardware_name'], 'Test CPU')

    def test_empty_or_invalid_selection_never_exports_everything(self):
        for ids in (None, [], '', 'all', [False], ['../outside'], [self.ids[0], 2]):
            with self.subTest(ids=ids):
                status, _, body = self.request({'ids': ids})
                self.assertEqual(status, 400)
                self.assertIn('error', json.loads(body))
        self.assertEqual(self.request({})[0], 400)

    def test_arc_lengths_definitions_and_undefined_metrics_survive_exports(self):
        status, _, body = self.request({'format': 'csv', 'ids': self.ids})
        self.assertEqual(status, 200)
        rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
        self.assertEqual(rows[0]['c'], '')
        self.assertEqual(rows[0]['total_length'], '')
        self.assertEqual(rows[0]['c_status'], 'not_recorded')
        self.assertEqual([float(row['c']) for row in rows[3:6]], [0, .4, 1])
        self.assertEqual([float(row['step_length']) for row in rows[3:6]], [0, 2, 3])
        self.assertEqual([float(row['cumulative_length']) for row in rows[3:6]], [0, 2, 5])
        self.assertEqual(json.loads(rows[3]['metric_definitions']), METRIC_DEFINITIONS)
        stationary = [row for row in rows if row['curve'] == 'Stationary readout']
        self.assertEqual(len(stationary), 3)
        self.assertTrue(all(row['c'] == row['d'] == '' and row['total_length'] == '0' for row in stationary))
        closed = [row for row in rows if row['curve'] == 'Closed path']
        self.assertEqual([float(row['c']) for row in closed], [0, .5, 1])
        self.assertTrue(all(row['d'] == '' for row in closed))
        status, _, body = self.request({'format': 'jsonl', 'ids': self.ids})
        self.assertEqual(status, 200)
        self.assertNotIn(b'NaN', body)
        self.assertNotIn(b'Infinity', body)
        exported = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(exported, [app.read_json(app.DATA / 'examples' / f'{i}.json') for i in self.ids])

    def test_save_and_library_reload_preserve_both_metrics(self):
        record = app.read_json(app.DATA / 'runs' / f'{self.ids[2]}.json')
        record['id'] = record_id = 'e' * 32
        run_path = app.DATA / 'runs' / f'{record_id}.json'
        example_path = app.DATA / 'examples' / f'{record_id}.json'
        try:
            app.write_json(run_path, record)
            request = Request(self.url + '/api/examples', data=json.dumps(
                dict(id=record_id, tag='Plateau', notes='Arc round trip')).encode(),
                headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
            with urlopen(self.url + '/api/library', timeout=5) as response:
                library = json.load(response)
            for scope in ('examples', 'history'):
                restored = next(r for r in library[scope] if r['id'] == record_id)
                self.assertEqual(restored['curves'], record['curves'])
                self.assertEqual(restored['metric_definitions'], record['metric_definitions'])
            self.assertEqual(app.read_json(example_path)['notes'], 'Arc round trip')
            self.assertEqual(app.read_json(run_path), record)
        finally:
            run_path.unlink(missing_ok=True)
            example_path.unlink(missing_ok=True)

    def test_suffix_scope_survives_csv_and_jsonl(self):
        status, _, body = self.request({'format': 'csv', 'ids': [self.ids[2]]})
        self.assertEqual(status, 200)
        rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
        for row in rows:
            self.assertEqual(row['patch_position'], 'different_suffix')
            self.assertEqual(row['patch_count'], '2')
            self.assertEqual(row['patch_start_a'], '0')
            self.assertEqual(row['patch_start_b'], '0')
            self.assertEqual(row['interpolation_unit'], 'per_token_shared_t')
            self.assertEqual(row['measurement_position'], 'last_token')
        status, _, body = self.request({'format': 'jsonl', 'ids': [self.ids[0], self.ids[2]]})
        self.assertEqual(status, 200)
        old, new = [json.loads(line) for line in body.splitlines()]
        self.assertNotIn('patch_position', old['settings'])
        self.assertEqual(new['settings']['patch_position'], 'different_suffix')

    def test_unknown_record_aborts_export(self):
        self.assertEqual(self.request({'ids': [self.ids[0], 'f' * 32]})[0], 404)

    def test_invalid_format_or_scope(self):
        self.assertEqual(self.request({'ids': self.ids[:1], 'format': 'pdf'})[0], 400)
        self.assertEqual(self.request({'ids': self.ids[:1], 'scope': 'other'})[0], 400)

    def test_existing_full_collection_get(self):
        status, _, body = self.request(query='?format=jsonl&scope=history')
        self.assertEqual(status, 200)
        self.assertEqual({json.loads(line)['id'] for line in body.splitlines()}, set(self.ids))


if __name__ == '__main__':
    unittest.main(verbosity=2)
