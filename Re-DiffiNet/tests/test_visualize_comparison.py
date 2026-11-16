"""Small synthetic CPU checks; never read research data or start training."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize_comparison import (select_cases, corrections, representative_slice,
                                  disjoint, check_rows, sample_volume, plot_case, plot_summary)
from testing_utils import DiceAccumulator, postprocess, paired_row, evaluate


def row(sid, baseline, refined):
    return paired_row(sid, dict.fromkeys(('WT', 'TC', 'ET', 'Mean'), baseline),
                      dict.fromkeys(('WT', 'TC', 'ET', 'Mean'), refined))


class VisualizationTests(unittest.TestCase):
    def test_selection_and_ties(self):
        rows = [row('z', .8, .9), row('a', .8, .9), row('same', .8, .8), row('bad', .8, .7)]
        selected, counts = select_cases(rows)
        self.assertEqual([(kind, r['subject_id']) for kind, r in selected],
                         [('best_improved', 'a'), ('unchanged', 'same'), ('most_worsened', 'bad')])
        self.assertEqual(counts['worsened_cases'], 1)

    def test_no_worsened_and_near_not_exact(self):
        selected, counts = select_cases([row('a', .8, .9), row('b', .8, .80001)])
        self.assertEqual(selected[1][0], 'near_unchanged')
        self.assertEqual(counts['worsened_cases'], 0)
        self.assertEqual(counts['exactly_unchanged_cases'], 0)

    def test_correction_categories(self):
        b = np.array([0, 0, 1, 1, 1])
        p = np.array([0, 1, 0, 2, 1])
        y = np.array([0, 1, 1, 3, 0])
        np.testing.assert_array_equal(corrections(b, p, y), [0, 1, 2, 3, 0])

    def test_slice_rule_and_fallback(self):
        y = np.zeros((4, 4, 5), np.uint8)
        y[:2, :2, 1:3] = 3
        self.assertEqual(representative_slice(y, y, y)[0], 1)
        empty = np.zeros_like(y)
        self.assertEqual(representative_slice(empty, y, empty)[0], 1)
        self.assertEqual(representative_slice(empty, empty, empty)[0], 2)

    def test_disjoint_roundtrip(self):
        from brats_data_utils_multi_label_5folds import labels_to_regions
        labels = np.arange(4, dtype=np.uint8).reshape(1, 1, 4)
        np.testing.assert_array_equal(disjoint(labels_to_regions(labels)), labels)

    def test_ranking_validation(self):
        rows = [row('a', .8, .9), row('b', .8, .8)]
        check_rows(rows, ['a', 'b'])
        with self.assertRaises(ValueError):
            check_rows([rows[0], rows[0]], ['a', 'b'])
        rows[0]['refined_WT'] = float('nan')
        with self.assertRaises(ValueError):
            check_rows(rows, ['a', 'b'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('monai'),
                         'Requires existing torch/monai environment')
    def test_cpu_sample_matches_evaluator_and_restores_rng(self):
        import torch
        class FakeModel:
            def eval(self):
                return self
            def sample(self, image):
                return torch.rand((image.shape[0], 3, *image.shape[2:]), device=image.device)
        rng = np.random.default_rng(3)
        item = dict(subject_id='synthetic', image=rng.random((7, 32, 32, 48), dtype=np.float32),
                    label=(rng.random((3, 32, 32, 48)) > .5).astype(np.float32))
        config = dict(seed=123, roi_size=[32, 32, 32], overlap=.25)
        device = torch.device('cpu')
        state = torch.get_rng_state().clone()
        volume = sample_volume(FakeModel(), item, device, None, config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        accumulator = DiceAccumulator()
        accumulator.add(postprocess(volume), item['label'])
        metrics, _ = evaluate(FakeModel(), [item], device, None,
                              config['roi_size'], config['overlap'], config['seed'])
        self.assertEqual(accumulator.scores(), metrics['refined'])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib unavailable locally')
    def test_render_synthetic_pngs(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            os.environ['MPLCONFIGDIR'] = temp
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            image = np.zeros((7, 32, 32, 16), dtype=np.float32)
            image[2] = np.linspace(0, 1, 32)[:, None, None]
            target = np.zeros((3, 32, 32, 16), dtype=np.float32)
            target[:, 8:24, 8:24, 8] = 1
            refined = target.copy()
            item = dict(subject_id='SYNTHETIC_ONLY', image=image, label=target)
            r = row(item['subject_id'], 0, 1)
            plot_case(plt, item, refined, 'best_improved', r, Path(temp)/'case.png', 150)
            metrics = dict(baseline=dict.fromkeys(('WT', 'TC', 'ET', 'Mean'), .9),
                           refined=dict.fromkeys(('WT', 'TC', 'ET', 'Mean'), .91))
            plot_summary(plt, metrics, Path(temp)/'summary.png', 150)
            for name in ('case.png', 'summary.png'):
                self.assertTrue((Path(temp)/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
