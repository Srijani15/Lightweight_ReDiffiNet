"""Synthetic CPU checks for the result-figure selection and metric contracts."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_visualization import checked_rows, choose_case, choose_slice, same_metrics


def row(sid, baseline, refined):
    values = {'subject_id': sid}
    for prefix, number in (('baseline', baseline), ('refined', refined)):
        for region in ('WT', 'TC', 'ET', 'Mean'):
            values[f'{prefix}_{region}'] = str(number)
    return values


class ResultVisualizationTests(unittest.TestCase):
    def test_csv_validation_and_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'paired_per_case.csv'
            path.write_text(
                'subject_id,baseline_WT,baseline_TC,baseline_ET,baseline_Mean,'
                'refined_WT,refined_TC,refined_ET,refined_Mean\n'
                'a,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8\n'
                'b,0.8,0.8,0.8,0.8,0.7,0.7,0.7,0.7\n', encoding='utf-8')
            rows = checked_rows(path, ['a', 'b'])
            self.assertAlmostEqual(rows[0]['delta_Mean'], .1)
            self.assertAlmostEqual(rows[1]['delta_TC'], -.1)

    def test_selection_is_explicit_and_reproducible(self):
        rows = []
        for sid, delta in (('z', .2), ('a', .2), ('median', .003), ('other', -.1)):
            rows.append({'subject_id': sid, 'delta_Mean': delta})
        selected, reason = choose_case(rows, None, 'best-improved', .0005)
        self.assertEqual(selected['subject_id'], 'a')
        self.assertIn('largest', reason)
        selected, reason = choose_case(rows, None, 'representative', .0005)
        self.assertEqual(selected['subject_id'], 'median')
        self.assertIn('pooled', reason)
        selected, reason = choose_case(rows, 'z', 'best-improved', .0005)
        self.assertEqual(selected['subject_id'], 'z')
        self.assertIn('explicit', reason)
        with self.assertRaises(ValueError):
            choose_case(rows, 'missing', 'best-improved', .0005)

    def test_slice_priority(self):
        target = np.zeros((3, 3, 4, 5), dtype=bool)
        target[0, :, :, 1] = True
        target[0, :, :, 2] = True
        base = target.astype(np.float32)
        refined = base.copy()
        refined[0, 0, 0, 2] = 0
        selected, rule = choose_slice(target, base, refined)
        self.assertEqual(selected, 2)
        self.assertIn('largest ground-truth WT', rule)

    def test_metric_match(self):
        left = {'subject_id': 'x'}
        right = {'subject_id': 'x'}
        for prefix in ('baseline', 'refined'):
            for region in ('WT', 'TC', 'ET', 'Mean'):
                left[f'{prefix}_{region}'] = right[f'{prefix}_{region}'] = .9
        self.assertTrue(same_metrics(left, right))
        right['refined_ET'] = .8
        self.assertFalse(same_metrics(left, right))


if __name__ == '__main__':
    unittest.main()
