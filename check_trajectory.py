"""Independent known-answer trajectory checks; no downloads or saved data."""
import json
import math
import unittest

import torch

from trajectory import TrajectoryReadout, summarize


class TrajectoryChecks(unittest.TestCase):
    def measure(self, points, batches=None):
        vectors = torch.tensor(points, dtype=torch.float64)
        reader = TrajectoryReadout(vectors[[0, -1]])
        for batch in vectors.split(batches or len(vectors)):
            reader.append(batch)
        result = reader.finish()
        json.dumps(result, allow_nan=False)
        self.assertEqual(reader.previous.numel(), vectors.shape[-1])
        self.assertEqual(reader.previous.untyped_storage().nbytes(), reader.previous.numel() * 8)
        return result

    def check_values(self, result, cumulative, c):
        for actual, expected in zip(result['cumulative_length'], cumulative):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(result['total_length'], cumulative[-1], places=12)
        for actual, expected in zip(result['c'], c):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_uniform_straight_path(self):
        result = self.measure([[3*i, 4*i] for i in range(5)])
        self.check_values(result, [0, 5, 10, 15, 20], [0, .25, .5, .75, 1])
        self.assertEqual(result['d'], result['c'])

    def test_repeated_points_are_flat(self):
        result = self.measure([[0, 0], [0, 0], [3, 4], [3, 4], [6, 8]])
        self.check_values(result, [0, 0, 5, 5, 10], [0, 0, .5, .5, 1])

    def test_bent_and_backtracking_path(self):
        result = self.measure([[0, 0], [3, 0], [3, 4], [0, 4], [3, 4]])
        self.check_values(result, [0, 3, 7, 10, 13], [0, 3/13, 7/13, 10/13, 1])
        self.assertGreater(result['total_length'], 5)  # Straight-line endpoint L2.

    def test_closed_path_has_c_without_d(self):
        result = self.measure([[0, 0], [3, 0], [3, 4], [0, 0]])
        self.check_values(result, [0, 3, 7, 12], [0, .25, 7/12, 1])
        self.assertEqual(result['d'], [None]*4)
        self.assertEqual(result['d_status'], 'coincident_endpoints')

    def test_stationary_path_and_summaries(self):
        result = self.measure([[7, -2]]*5)
        self.assertEqual(result['c'], [None]*5)
        self.assertEqual(result['cumulative_length'], [0]*5)
        self.assertEqual(result['total_length'], 0)
        self.assertEqual(result['c_undefined_reason'], 'No measured movement; c(t) undefined')
        self.assertIsNone(summarize([0, .25, .5, .75, 1], result['c'], 'c')['max_abs_slope'])

    def test_tiny_movement_is_not_discarded(self):
        result = self.measure([[0], [1e-12], [2e-12]])
        self.assertEqual(result['c'], [0, .5, 1])
        self.assertEqual(result['d'], [None]*3)  # Existing absolute d tolerance.

    def test_every_batch_split_matches(self):
        points = [[0, 0], [3, 0], [3, 4], [3, 4], [-4, 4], [-4, -8], [0, 0]]
        expected = self.measure(points)
        for size in range(1, len(points)+1):
            with self.subTest(size=size):
                self.assertEqual(self.measure(points, size), expected)
        # Unequal batches, including an empty batch.
        vectors = torch.tensor(points, dtype=torch.float64)
        reader = TrajectoryReadout(vectors[[0, -1]])
        for lo, hi in [(0, 2), (2, 2), (2, 3), (3, 7)]:
            reader.append(vectors[lo:hi])
        self.assertEqual(reader.finish(), expected)

    def test_stable_accumulation_after_large_segment(self):
        points = [[0, 0], [1e16, 0]] + [[1e16, i] for i in range(1, 101)]
        result = self.measure(points, 7)
        self.assertEqual(result['total_length'], math.fsum([1e16]+[1]*100))
        self.assertEqual(result['c'][-1], 1)

    def test_nonfinite_readout_serializes_nulls(self):
        result = self.measure([[0, 0], [float('nan'), 1], [2, 2]])
        self.assertEqual(result['c_status'], 'nonfinite')
        self.assertEqual(result['d_status'], 'nonfinite')
        self.assertIsNone(result['total_length'])

    def test_ordinary_d_arithmetic_is_unchanged(self):
        vectors = torch.tensor([[1., 2.], [2., -1.], [-3., 6.], [4., 5.]])
        reader = TrajectoryReadout(vectors[[0, -1]])
        reader.append(vectors)
        da = (vectors-vectors[0]).norm(dim=-1)
        db = (vectors-vectors[-1]).norm(dim=-1)
        self.assertEqual(reader.finish()['d'], (da/(da+db)).tolist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
