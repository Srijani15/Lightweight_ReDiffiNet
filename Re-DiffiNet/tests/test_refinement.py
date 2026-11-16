"""CPU-only contract tests with tiny synthetic data; no datasets or training runs."""
from __future__ import annotations
import ast
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brats_data_utils_multi_label_5folds import FoldExport, ExportDataset, read_case, labels_to_regions
from testing_utils import (DiceAccumulator, postprocess, fold_paths, save_json, save_csv,
                           require_verification, validate_checkpoint, atomic_write,
                           update_summary, METRIC_VERSION, FORMAT_VERSION, evaluate)
import train
import test as evaluation_cli

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_MODEL = HAS_TORCH and importlib.util.find_spec('monai') is not None


def fixture(root):
    data = root/'export'
    (data/'train').mkdir(parents=True)
    (data/'validation').mkdir()
    (data/'export_provenance.txt').write_text('Synthetic fixture, NOT a research dataset', encoding='utf-8')
    assignments = root/'unet'/'fold_0'/'fold_assignments.csv'
    ids = [f'case{i:02d}' for i in range(10)]
    shuffled = np.arange(10)
    np.random.default_rng(123).shuffle(shuffled)
    val = set(np.array_split(shuffled, 5)[0])
    rows, manifest = [], []
    for i, sid in enumerate(ids):
        split = 'validation' if i in val else 'training'
        path = data/('validation' if split == 'validation' else 'train')/(sid+'.pkl.gz')
        row = dict(source_directory='train', subject_id=sid, fold_split=split)
        rows.append(row)
        manifest.append(dict(**row, output_path=str(path)))
        shape = (32, 32, 32)
        label = np.zeros(shape, np.uint8)
        label[4:16, 5:20, 6:22] = 1
        label[9:12, 9:12, 9:12] = 3
        mri = [np.full(shape, j, np.float32) for j in range(4)]
        with gzip.open(path, 'wb') as handle:
            pickle.dump((*mri, label, label.copy()), handle)
    save_csv(assignments, rows)
    save_csv(data/'export_manifest.csv', manifest)
    return FoldExport(data, assignments, 0)


class DataMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.export = fixture(self.root)

    def test_shape_order_and_label_separation(self):
        path = self.export.paths['case00']
        image, target = read_case(path)
        self.assertEqual(image.shape, (7, 32, 32, 32))
        for channel in range(4):
            self.assertTrue((image[channel] == channel).all())
        np.testing.assert_array_equal(image[4:], target)
        with gzip.open(path, 'rb') as handle:
            values = list(pickle.load(handle))
        values[4] = np.zeros_like(values[4])
        with gzip.open(path, 'wb') as handle:
            pickle.dump(values, handle)
        altered, altered_target = read_case(path)
        np.testing.assert_array_equal(image, altered)
        self.assertFalse(altered_target.any())

    def test_regions_thresholds_hierarchy(self):
        p = np.array([[.45, .46, .9, .9], [.9, .40, .8, .1], [.9, .45, .1, .46]])[:, None, None, :]
        expected = np.array([0, 2, 1, 3], np.uint8)[None, None, :]
        np.testing.assert_array_equal(postprocess(p), labels_to_regions(expected))

    def test_pooled_counts_not_case_average(self):
        acc = DiceAccumulator()
        one = np.ones((3, 1, 1, 1), bool)
        first = acc.add(one, one)['Mean']
        truth = np.ones((3, 1, 1, 9), bool)
        second = acc.add(np.zeros_like(truth), truth)['Mean']
        self.assertAlmostEqual(acc.scores()['Mean'], (2+1e-8)/(11+1e-8))
        self.assertNotAlmostEqual(acc.scores()['Mean'], (first+second)/2)
        empty = DiceAccumulator().add(np.zeros_like(one), np.zeros_like(one))
        self.assertEqual(empty['Mean'], 1.)

    def test_duplicate_and_wrong_split_rejected(self):
        rows = self.export.rows + [self.export.rows[0]]
        save_csv(self.export.assignments_path, rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            FoldExport(self.export.root, self.export.assignments_path, 0)

    def test_wrong_manifest_rejected(self):
        with self.export.manifest_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['fold_split'] = 'other'
        save_csv(self.export.manifest_path, rows)
        with self.assertRaisesRegex(ValueError, 'disagreement'):
            FoldExport(self.export.root, self.export.assignments_path, 0)

    def test_corruption_not_silently_skipped(self):
        sid = self.export.ids['training'][0]
        self.export.paths[sid].write_bytes(b'broken')
        with self.assertRaises((OSError, ValueError, EOFError)):
            ExportDataset(self.export, 'training')[0]

    def test_verification_invalidated_by_change(self):
        path = self.root/'verification.json'
        report = dict(identity=self.export.identity(), metric_version=METRIC_VERSION,
                      approved_for_diagnostic=True, evaluated_cases=2)
        save_json(path, report)
        require_verification(path, self.export)
        self.export.provenance_path.write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            require_verification(path, self.export)

    def test_fold_checkpoint_and_tensorboard_isolation(self):
        a, b = fold_paths(self.root, 0), fold_paths(self.root, 1)
        self.assertNotEqual(a['checkpoints'], b['checkpoints'])
        self.assertNotEqual(a['tensorboard'], b['tensorboard'])
        self.assertNotEqual(a['root'], fold_paths(self.root, 0, True)['root'])
        cp = dict(format_version=FORMAT_VERSION, metric_version=METRIC_VERSION,
                  fold=0, identity=self.export.identity(), config={'model': 1})
        validate_checkpoint(cp, 0, self.export.identity(), {'model': 1})
        with self.assertRaises(ValueError):
            validate_checkpoint(cp, 1, self.export.identity())
        with self.assertRaises(ValueError):
            validate_checkpoint(cp, 0, self.export.identity(), {'model': 2})

    @unittest.skipUnless(HAS_TORCH and importlib.util.find_spec('tensorboard'), 'tensorboard unavailable')
    def test_tensorboard_scalar_roundtrip(self):
        from torch.utils.tensorboard import SummaryWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for fold in (0, 1):
            path = fold_paths(self.root, fold)['tensorboard']
            with SummaryWriter(str(path)) as writer:
                writer.add_scalar('Validation/WT_Dice', .8+fold*.1, 2)
                writer.flush()
            events = EventAccumulator(str(path)).Reload()
            self.assertEqual(events.Tags()['scalars'], ['Validation/WT_Dice'])
            self.assertEqual(events.Scalars('Validation/WT_Dice')[0].step, 2)

    def test_summary_upsert_and_partial_cv(self):
        for fold, value in ((0, .8), (1, .9), (0, .85)):
            result = update_summary(self.root, dict(fold=fold, baseline_Mean=.8, refined_Mean=value, delta_Mean=value-.8))
        self.assertEqual(result['completed_folds'], [0, 1])
        self.assertFalse(result['complete_five_fold_cv'])
        self.assertAlmostEqual(result['metrics']['refined_Mean']['mean'], .875)
        self.assertAlmostEqual(result['metrics']['refined_Mean']['std'], .025)

    def test_atomic_failure_preserves_old_file(self):
        path = self.root/'state.json'
        path.write_text('old')
        def fail(tmp):
            tmp.write_text('partial')
            raise RuntimeError('simulated interruption')
        with self.assertRaises(RuntimeError):
            atomic_write(path, fail)
        self.assertEqual(path.read_text(), 'old')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_reference_membership_and_best_checkpoint(self):
        cp = dict(fold_number=0, n_folds=5, args=dict(seed=123, fold_source='train', crop_size=[32]*3),
                  best_dice_scores=[1., 1., 1.], best_mean_dice=1., epoch=20, best_epoch=20)
        scores, shape = evaluation_cli.check_reference(cp, self.export)
        self.assertEqual(shape, (32, 32, 32))
        cp['args']['seed'] = 999
        with self.assertRaises(ValueError):
            evaluation_cli.check_reference(cp, self.export)

    def test_full_run_rejects_diagnostic_subsets(self):
        args = train.parser().parse_args(['--data-dir', 'data', '--assignments', 'assign.csv',
                                         '--verification', 'report.json', '--output-dir', 'output',
                                         '--max-train-cases', '1'])
        with self.assertRaisesRegex(ValueError, 'Subset limits'):
            train.check_args(args)

    @unittest.skipUnless(HAS_TORCH, 'torch unavailable')
    def test_actual_unet_postprocessing_and_accumulation(self):
        import torch
        path = ROOT.parent/'brats2023_updated'/'utils'/'general_utils.py'
        if not path.exists():
            self.skipTest('Reference U-Net source unavailable')
        names = {'overlapping_probs_to_preds', 'disjoint_to_overlapping'}
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        namespace = {'torch': torch}
        exec(compile(module, str(path), 'exec'), namespace)
        rng = np.random.default_rng(9)
        acc = DiceAccumulator()
        intersections = torch.zeros(3, dtype=torch.float64)
        prediction_sizes = intersections.clone()
        target_sizes = intersections.clone()
        for _ in range(3):
            p = rng.random((3, 4, 5, 6), dtype=np.float32)
            y = labels_to_regions(rng.integers(0, 4, size=(4, 5, 6)))
            ref = namespace['disjoint_to_overlapping'](namespace['overlapping_probs_to_preds'](torch.from_numpy(p[None])))
            np.testing.assert_array_equal(postprocess(p), ref[0].numpy())
            acc.add(postprocess(p), y)
            ref = ref.to(torch.float64)
            truth = torch.from_numpy(y[None]).to(torch.float64)
            intersections += (ref*truth).sum((0, 2, 3, 4))
            prediction_sizes += ref.sum((0, 2, 3, 4))
            target_sizes += truth.sum((0, 2, 3, 4))
        expected = ((2*intersections+1e-8)/(prediction_sizes+target_sizes+1e-8)).numpy()
        np.testing.assert_array_equal([acc.scores()[r] for r in ('WT', 'TC', 'ET')], expected)

    @unittest.skipUnless(HAS_TORCH, 'torch unavailable')
    def test_verification_cli_read_only(self):
        import torch
        cp = dict(fold_number=0, n_folds=5, args=dict(seed=123, fold_source='train', crop_size=[32]*3),
                  best_dice_scores=[1., 1., 1.], best_mean_dice=1., epoch=20, best_epoch=20)
        path = self.root/'unet'/'fold_0'/'model'/'best_mean_dice.pt'
        path.parent.mkdir()
        torch.save(cp, path)
        before = self.export.identity()
        result = subprocess.run([sys.executable, str(ROOT/'test.py'), 'verify-export',
                                 '--data-dir', str(self.export.root), '--assignments', str(self.export.assignments_path),
                                 '--unet-checkpoint', str(path), '--out-dir', str(self.root/'audit')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = require_verification(self.root/'audit'/'verification.json', self.export)
        self.assertEqual(report['baseline']['Mean'], 1.)
        self.assertEqual(report['checked_training_cases'], 8)
        self.assertEqual(self.export.identity(), before)


@unittest.skipUnless(HAS_MODEL, 'torch/MONAI unavailable')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def make_model(self):
        import torch
        from unet.refinement import DiffUNet
        torch.manual_seed(5)
        model = DiffUNet(width=2, sampling_steps=2)
        image = torch.randn(1, 7, 32, 32, 32)
        image[:, 4:] = (torch.rand(1, 3, 32, 32, 32) > .5).float()
        return model, image

    def test_loss_gradients_and_residual_correction_reachability(self):
        import torch
        model, image = self.make_model()
        target = 1-image[:, 4:]
        losses = model.training_losses(image, target)
        losses['Loss'].backward()
        for module in (model.gate_head, model.model.final_conv, model.embed_model):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        corrected = model.compose(image[:, 4:], target-image[:, 4:], torch.ones_like(target))
        torch.testing.assert_close(corrected, target)

    def test_ddim_calls_model_and_ranges_and_reproducibility(self):
        import torch
        model, image = self.make_model()
        model.eval()
        calls = []
        hook = model.register_forward_pre_hook(lambda module, args: calls.append(args[1].clone()))
        noise = torch.randn(1, 3, 32, 32, 32)
        result, residual, gate = model.sample(image, noise, return_parts=True)
        hook.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual([int(t[0]) for t in calls], [999, 0])
        self.assertEqual(result.shape, (1, 3, 32, 32, 32))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((residual >= -1) & (residual <= 1)).all())
        self.assertTrue(((gate > 0) & (gate < 1)).all())
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        torch.testing.assert_close(result, model.sample(image, noise), rtol=0, atol=0)

    def test_forward_diffusion_residual_scale(self):
        import torch
        model, image = self.make_model()
        residual = 1-2*image[:, 4:]
        noise = torch.randn_like(residual)
        t = torch.tensor([347])
        actual = model.diffusion.q_sample(residual, t, noise)
        alpha = float(model.diffusion.alphas_cumprod[347])
        expected = alpha**.5*residual + (1-alpha)**.5*noise
        torch.testing.assert_close(actual, expected)

    def test_primary_segmentation_loss_reaches_both_heads_and_timestep_layers(self):
        import torch
        model, image = self.make_model()
        losses = model.training_losses(image, 1-image[:, 4:])
        # Auxiliary losses must not conceal a detached final-segmentation path.
        (losses['Dice_Loss'] + losses['BCE_Loss']).backward()
        for module in (model.gate_head, model.model.final_conv, model.model.conv_0,
                       model.model.down_4, model.model.temb, model.embed_model):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_sampled_learned_heads_can_change_hard_masks(self):
        import torch
        model, image = self.make_model()
        image[:, 4:] = 0
        image[:, 4:, :, :, 16:] = 1  # nested, valid WT/TC/ET baseline
        noise = torch.randn(1, 3, 32, 32, 32)
        small = model.sample(image, noise)
        self.assertGreater(float((small-image[:, 4:]).abs().mean()), 0.)
        np.testing.assert_array_equal(postprocess(small[0].numpy()), image[0, 4:].numpy())
        with torch.no_grad():
            model.model.final_conv.weight.zero_()
            model.model.final_conv.bias.fill_(6.)
            model.gate_head.weight.zero_()
            model.gate_head.bias.fill_(6.)
        calls = []
        hook = model.register_forward_pre_hook(lambda module, args: calls.append(int(args[1][0])))
        try:
            strong = model.sample(image, noise)
        finally:
            hook.remove()
        self.assertEqual(calls, [999, 0])
        np.testing.assert_array_equal(postprocess(strong[0].numpy()), 1-image[0, 4:].numpy())

    def test_read_only_inspector_matches_existing_evaluation_and_preserves_state(self):
        import torch
        from inspect_refinement import inspect_case
        model, image = self.make_model()
        # Odd full-volume size exercises padding-free overlap stitching and edge windows.
        image = torch.nn.functional.pad(image, (0, 8, 0, 8, 0, 8))
        item = dict(subject_id='overlap-fixture', image=image[0].numpy(), label=image[0, 4:].numpy())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = inspect_case(model, item, torch.device('cpu'), None, [32]*3, .25, 123)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        metrics, rows = evaluate(model, [item], torch.device('cpu'), None, [32]*3, .25, 123)
        self.assertEqual(report['baseline'], metrics['baseline'])
        self.assertEqual(report['refined'], metrics['refined'])
        self.assertEqual(report['baseline_counts'], metrics['baseline_counts'])
        self.assertEqual(report['refined_counts'], metrics['refined_counts'])
        self.assertGreater(report['gate_mean'], 0.)
        self.assertGreater(report['abs_residual_mean'], 0.)
        self.assertGreater(report['mean_abs_prediction_change'], 0.)
        self.assertEqual(report['region_decisions']['changed'], 0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_checkpoint_serialization_and_cpu_paired_evaluation(self):
        import torch
        model, image = self.make_model()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.pt'
            atomic_write(path, lambda tmp: torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict()), tmp))
            saved = torch.load(path, map_location='cpu', weights_only=False)
            clone, _ = self.make_model()
            clone.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, clone.state_dict()[key])
        item = dict(subject_id='synthetic', image=image[0].numpy(), label=image[0, 4:].numpy())
        rng = torch.get_rng_state().clone()
        metrics, rows = evaluate(model, [item], torch.device('cpu'), None, [32]*3, .25, 123)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(metrics['cases'], 1)
        self.assertEqual(metrics['baseline']['Mean'], rows[0]['baseline_Mean'])


class RefinementStatisticsTests(unittest.TestCase):
    def test_corrected_worsened_and_wrong_to_wrong_counts(self):
        from inspect_refinement import refinement_statistics
        # First region triplet is corrected, second worsened, third wrong-to-wrong.
        b = labels_to_regions(np.array([0, 3, 1], np.uint8)[None, None, :]).astype(np.float32)
        p = labels_to_regions(np.array([3, 0, 2], np.uint8)[None, None, :]).astype(np.float32)
        y = labels_to_regions(np.array([3, 3, 3], np.uint8)[None, None, :]).astype(np.float32)
        report = refinement_statistics(b, p, p-b, np.ones_like(b), y)
        decisions = report['region_decisions']
        self.assertEqual(decisions['changed'], 7)
        self.assertEqual(decisions['corrected'], 3)
        self.assertEqual(decisions['worsened'], 4)
        self.assertAlmostEqual(decisions['corrected_percent_of_changes'], 300/7)
        spatial = report['spatial_voxels']
        self.assertEqual(spatial['changed'], 3)
        self.assertEqual(spatial['corrected_to_exact_target'], 1)
        self.assertEqual(spatial['worsened_from_exact_target'], 1)
        self.assertEqual(spatial['wrong_to_different_wrong'], 1)

    def test_subthreshold_changes_have_no_hard_changes_or_defined_percentages(self):
        from inspect_refinement import refinement_statistics
        b = np.zeros((3, 2, 2, 2), np.float32)
        gate = np.full_like(b, .1)
        residual = np.full_like(b, .2)
        report = refinement_statistics(b, gate*residual, residual, gate, b)
        self.assertAlmostEqual(report['mean_abs_prediction_change'], .02)
        self.assertEqual(report['region_decisions']['changed_fraction'], 0.)
        self.assertIsNone(report['region_decisions']['corrected_percent_of_changes'])
        self.assertIsNone(report['spatial_voxels']['worsened_percent_of_changes'])


class DeterminismTests(unittest.TestCase):
    def test_policy_saved_and_compatibility_enforced(self):
        args = train.parser().parse_args(['--data-dir', 'data', '--assignments', 'assign.csv',
                                         '--verification', 'report.json', '--output-dir', 'output'])
        self.assertEqual(args.determinism, 'warn-only')
        config = train.configuration(args)
        self.assertEqual(config['determinism'], 'warn-only')
        train.validate_diagnostic_configuration(config, config)
        checkpoint = dict(format_version=FORMAT_VERSION, metric_version=METRIC_VERSION,
                          fold=0, identity={'fixture': True}, config=config)
        validate_checkpoint(checkpoint, 0, {'fixture': True}, config)
        for policy in ('strict', 'off', None):
            altered = dict(config, determinism=policy)
            if policy is None:
                altered.pop('determinism')  # pre-policy checkpoints/diagnostics cannot silently migrate
            with self.assertRaisesRegex(ValueError, 'determinism'):
                train.validate_diagnostic_configuration(altered, config)
            with self.assertRaisesRegex(ValueError, 'Resume configuration changed'):
                validate_checkpoint(dict(checkpoint, config=altered), 0, {'fixture': True}, config)

    @unittest.skipUnless(HAS_TORCH, 'torch unavailable')
    def test_modes_and_seeded_streams_cpu(self):
        import os
        import random
        import torch
        from unittest.mock import patch
        from testing_utils import configure_determinism
        previous = (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                    random.getstate(), np.random.get_state(), torch.get_rng_state())
        try:
            with patch.dict(os.environ), patch('torch.cuda.is_available', return_value=False), \
                    patch('torch.cuda.manual_seed_all'):
                samples = []
                for policy, enabled, warn_only in [('strict', True, False), ('warn-only', True, True),
                                                   ('off', False, False), ('warn-only', True, True)]:
                    configure_determinism(123, policy)
                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn_only)
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    self.assertTrue(torch.backends.cudnn.deterministic)
                    samples.append((random.random(), float(np.random.rand()), float(torch.rand(()))))
                self.assertTrue(all(sample == samples[0] for sample in samples))
                with self.assertRaisesRegex(ValueError, 'Unknown determinism policy'):
                    configure_determinism(123, 'invalid')
        finally:
            enabled, warn_only, benchmark, deterministic, py_rng, np_rng, torch_rng = previous
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = deterministic
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(torch_rng)


class ImportTests(unittest.TestCase):
    def test_help_and_imports_without_torch(self):
        for script in ('train.py', 'test.py'):
            result = subprocess.run([sys.executable, str(ROOT/script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        code = "import sys; sys.modules['torch']=None; import train, test, testing_utils, brats_data_utils_multi_label_5folds"
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
