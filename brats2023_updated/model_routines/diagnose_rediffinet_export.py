"""Read-only audit of a baseline Fold-0 export and its reported Dice scores.

This intentionally does not write pickles or checkpoints.  It compares four
quantities on the *same* fold membership:

* the original ``train_baseline.validate`` result (dataset-level Dice),
* the same thresholded prediction after its exported 0/1/2/3 label round trip,
* the exporter-style FP32 prediction and the pickle actually on disk, and
* Re-DiffiNet's per-case, NaN-skipping Dice average.

Run from the directory above ``brats2023_updated``.  For example::

    python -m brats2023_updated.model_routines.diagnose_rediffinet_export \
      --checkpoint ~/Major_Project/UNet_Output_Light_BF16/fold_0/model/best_mean_dice.pt \
      --train-dir /path/to/BraTS2023_split/train \
      --val-dir /path/to/BraTS2023_split/validation \
      --fold-assignments ~/Major_Project/UNet_Output_Light_BF16/fold_0/fold_assignments.csv \
      --export-dir ~/Major_Project/ReDiffiNet_Data --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from ..datasets.brats_dataset import BratsDataset
from ..utils.general_utils import (
    disjoint_to_overlapping,
    one_hot_channels_to_three_labels,
    probs_to_preds,
    seg_to_one_hot_channels,
)
from .export_for_rediffinet import FoldAssignment, load_model, read_fold_assignments
from .train_baseline import targets_from_segmentation, validate as training_validate


REGION_NAMES = ('WT', 'TC', 'ET')
EPSILON = 1e-8
PICKLE_SUFFIXES = ('.pkl.gz', '.pkl')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--fold-assignments', required=True)
    parser.add_argument('--export-dir', required=True, help='Existing ReDiffiNet_Data directory; it is never modified.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers; 0 is most transparent for auditing.')
    parser.add_argument(
        '--metrics-csv', default=None,
        help='Optional metrics.csv.  Defaults to the fold directory inferred from the checkpoint.',
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Mapping[str, object]:
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f'{path} is not a checkpoint metadata mapping.')
    return value


def _checkpoint_args(checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    saved = checkpoint.get('args')
    if not isinstance(saved, Mapping):
        raise KeyError('Checkpoint lacks its saved train_baseline arguments.')
    return saved


def _checkpoint_int(checkpoint: Mapping[str, object], saved_args: Mapping[str, object], name: str) -> int:
    value = checkpoint.get(name, saved_args.get(name))
    if value is None:
        raise KeyError(f'Checkpoint lacks {name!r}.')
    return int(value)


def _checkpoint_bool(checkpoint: Mapping[str, object], name: str) -> bool:
    return bool(checkpoint.get(name, False))


def _configure_checkpoint_cuda(checkpoint: Mapping[str, object], device: torch.device) -> None:
    """Match the saved validation numerical mode without changing any model state."""
    if device.type != 'cuda':
        return
    torch.backends.cuda.matmul.allow_tf32 = _checkpoint_bool(checkpoint, 'tf32_enabled')
    torch.backends.cudnn.allow_tf32 = _checkpoint_bool(checkpoint, 'tf32_enabled')
    torch.backends.cudnn.benchmark = _checkpoint_bool(checkpoint, 'cudnn_benchmark_enabled')


def _cuda_settings(device: torch.device) -> tuple[bool, bool, bool] | None:
    """Capture the process defaults used by a standalone exporter invocation."""
    if device.type != 'cuda':
        return None
    return (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        bool(torch.backends.cudnn.benchmark),
    )


def _restore_cuda_settings(settings: tuple[bool, bool, bool] | None) -> None:
    if settings is None:
        return
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = settings


def _build_expected_fold(
    train_dir: str,
    val_dir: str,
    crop_size: tuple[int, int, int] | None,
    seed: int,
    n_folds: int,
    fold_number: int,
    fold_source: str,
) -> tuple[Dataset, list[tuple[str, str, str]], list[int]]:
    """Reproduce ``make_cross_validation_datasets`` without reading its CSV output."""
    sources: list[tuple[str, BratsDataset]] = [
        ('train', BratsDataset(train_dir, mode='train', crop_size=crop_size)),
    ]
    if fold_source == 'all':
        sources.append(('validation', BratsDataset(val_dir, mode='train', crop_size=crop_size)))
    elif fold_source != 'train':
        raise ValueError(f'Unsupported saved fold_source {fold_source!r}.')

    dataset: Dataset = sources[0][1] if len(sources) == 1 else ConcatDataset([source[1] for source in sources])
    case_ids = [
        (source_name, subject_id)
        for source_name, source_dataset in sources
        for subject_id in source_dataset.subject_list
    ]
    shuffled_indices = np.arange(len(case_ids))
    np.random.default_rng(seed).shuffle(shuffled_indices)
    validation_indices = np.array_split(shuffled_indices, n_folds)[fold_number]
    validation_set = set(validation_indices.tolist())
    assignments = [
        (source_name, subject_id, 'validation' if index in validation_set else 'training')
        for index, (source_name, subject_id) in enumerate(case_ids)
    ]
    return dataset, assignments, sorted(validation_set)


def _assignment_tuples(assignments: Iterable[FoldAssignment]) -> list[tuple[str, str, str]]:
    return [(item.source_directory, item.subject_id, item.fold_split) for item in assignments]


def _assert_assignment_reproduction(
    expected: list[tuple[str, str, str]], actual: list[FoldAssignment],
) -> None:
    actual_rows = _assignment_tuples(actual)
    if actual_rows == expected:
        print(f'Fold assignment reproduction: PASS ({len(actual_rows)} rows exactly match saved seed/fold settings).')
        return
    expected_set, actual_set = set(expected), set(actual_rows)
    missing, unexpected = expected_set - actual_set, actual_set - expected_set
    raise RuntimeError(
        'fold_assignments.csv does not reproduce the checkpoint fold. '
        f'Missing examples={sorted(missing)[:3]}; unexpected examples={sorted(unexpected)[:3]}; '
        f'expected rows={len(expected)}, CSV rows={len(actual_rows)}.'
    )


def _load_pickle(path: Path) -> tuple[object, ...]:
    if path.name.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as handle:
            value = pickle.load(handle)
    elif path.name.endswith('.pkl'):
        with path.open('rb') as handle:
            value = pickle.load(handle)
    else:
        raise ValueError(f'Unsupported pickle path: {path}')
    if not isinstance(value, tuple) or len(value) != 6:
        raise ValueError(f'{path} is not the required six-array export tuple.')
    return value


def _pickle_paths(split_dir: Path) -> dict[str, Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Export split does not exist: {split_dir}')
    paths: dict[str, Path] = {}
    for path in split_dir.iterdir():
        if not path.is_file():
            continue
        suffix = next((suffix for suffix in PICKLE_SUFFIXES if path.name.endswith(suffix)), None)
        if suffix is None:
            continue
        case_id = path.name[:-len(suffix)]
        if case_id in paths:
            raise ValueError(f'Duplicate exported case ID in {split_dir}: {case_id}')
        paths[case_id] = path
    return paths


def _assert_export_membership(assignments: list[FoldAssignment], export_dir: Path) -> dict[str, Path]:
    expected = {
        'train': {row.subject_id for row in assignments if row.fold_split == 'training'},
        'validation': {row.subject_id for row in assignments if row.fold_split == 'validation'},
    }
    actual = {
        'train': _pickle_paths(export_dir / 'train'),
        'validation': _pickle_paths(export_dir / 'validation'),
    }
    for split in ('train', 'validation'):
        actual_ids = set(actual[split])
        if actual_ids != expected[split]:
            raise RuntimeError(
                f'Export {split} membership differs from fold assignments. '
                f'missing={sorted(expected[split] - actual_ids)[:5]}; '
                f'unexpected={sorted(actual_ids - expected[split])[:5]}; '
                f'expected={len(expected[split])}, actual={len(actual_ids)}.'
            )
    if set(actual['train']) & set(actual['validation']):
        raise RuntimeError('Export has subject IDs in both train and validation.')
    print(f'Export membership: PASS (train={len(actual["train"])}, validation={len(actual["validation"])}).')
    return actual['validation']


@dataclass
class DiceAccumulator:
    intersections: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    prediction_sizes: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    target_sizes: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().to(torch.float64)
        target = target.detach().cpu().to(torch.float64)
        self.intersections += (prediction * target).sum(dim=(0, 2, 3, 4))
        self.prediction_sizes += prediction.sum(dim=(0, 2, 3, 4))
        self.target_sizes += target.sum(dim=(0, 2, 3, 4))

    def result(self) -> tuple[float, float, float]:
        values = (2.0 * self.intersections + EPSILON) / (self.prediction_sizes + self.target_sizes + EPSILON)
        return tuple(float(value) for value in values)


@dataclass
class MacroDiceAccumulator:
    """Exactly the per-case, empty/empty-NaN convention used by Re-DiffiNet."""
    values: list[list[float]] = field(default_factory=lambda: [[], [], []])
    skipped_empty: list[int] = field(default_factory=lambda: [0, 0, 0])

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction_array = prediction.detach().cpu().numpy().astype(bool, copy=False)
        target_array = target.detach().cpu().numpy().astype(bool, copy=False)
        for batch_index in range(prediction_array.shape[0]):
            for channel in range(3):
                predicted_region = prediction_array[batch_index, channel]
                target_region = target_array[batch_index, channel]
                if not predicted_region.any() and not target_region.any():
                    self.skipped_empty[channel] += 1
                    continue
                true_positive = np.count_nonzero(predicted_region & target_region)
                false_positive = np.count_nonzero(predicted_region & ~target_region)
                false_negative = np.count_nonzero(~predicted_region & target_region)
                self.values[channel].append(float(
                    2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                ))

    def result(self) -> tuple[float, float, float]:
        return tuple(float(np.mean(channel_values)) if channel_values else 0.0 for channel_values in self.values)


def _round_trip_export_labels(predicted_disjoint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exactly the exporter conversion and return label map plus WT/TC/ET reconstruction."""
    if predicted_disjoint.shape[0] != 1:
        raise ValueError('The diagnostic deliberately uses batch size 1 for pickle comparison.')
    label_map = one_hot_channels_to_three_labels(predicted_disjoint[0]).to(torch.uint8)
    reconstructed_disjoint = seg_to_one_hot_channels(label_map[None, None, ...])
    reconstructed_regions = disjoint_to_overlapping(reconstructed_disjoint)
    return label_map, reconstructed_regions


def _format_scores(scores: tuple[float, float, float]) -> str:
    return ', '.join(f'{name}={value:.9f}' for name, value in zip(REGION_NAMES, scores)) + f', mean={np.mean(scores):.9f}'


def _assert_scores_equal(name: str, actual: tuple[float, float, float], expected: tuple[float, float, float]) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-5):
        raise RuntimeError(f'{name} differs: actual [{_format_scores(actual)}]; expected [{_format_scores(expected)}].')
    print(f'{name}: PASS ({_format_scores(actual)}).')


def _metrics_row(path: Path, epoch: int) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if int(row['epoch']) == epoch:
                return row
    return None


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError('--workers cannot be negative.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'{device} was requested but CUDA is unavailable.')

    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = _torch_load(checkpoint_path)
    saved_args = _checkpoint_args(checkpoint)
    exporter_cuda_defaults = _cuda_settings(device)
    _configure_checkpoint_cuda(checkpoint, device)
    model, crop_size = load_model(checkpoint_path, device)

    seed = _checkpoint_int(checkpoint, saved_args, 'seed')
    n_folds = _checkpoint_int(checkpoint, saved_args, 'n_folds')
    fold_number = _checkpoint_int(checkpoint, saved_args, 'fold_number')
    fold_source = str(checkpoint.get('fold_source', saved_args.get('fold_source', 'all')))
    best_epoch = int(checkpoint.get('best_epoch', checkpoint.get('epoch', 0)))
    checkpoint_best = float(checkpoint['best_mean_dice'])
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Checkpoint metadata: epoch={checkpoint.get("epoch")}, best_epoch={best_epoch}, '
          f'best_mean_dice={checkpoint_best:.9f}, base_channels={model.base_channels}.')
    print(f'Saved fold settings: seed={seed}, n_folds={n_folds}, fold_number={fold_number}, fold_source={fold_source}.')
    print(f'Saved crop: {crop_size if crop_size is not None else "historical fixed crop"}.')

    source_dirs = {'train': Path(args.train_dir), 'validation': Path(args.val_dir)}
    csv_assignments = read_fold_assignments(args.fold_assignments, source_dirs)
    full_dataset, expected_assignments, validation_indices = _build_expected_fold(
        args.train_dir, args.val_dir, crop_size, seed, n_folds, fold_number, fold_source,
    )
    _assert_assignment_reproduction(expected_assignments, csv_assignments)
    validation_dataset = Subset(full_dataset, validation_indices)
    validation_rows = [row for row in csv_assignments if row.fold_split == 'validation']
    if len(validation_dataset) != len(validation_rows):
        raise RuntimeError('Recreated validation subset and fold_assignments.csv have different validation counts.')
    print(f'Fold-{fold_number} validation cases ({len(validation_rows)}):')
    for row in validation_rows:
        print(f'  {row.source_directory}/{row.subject_id}')

    export_paths = _assert_export_membership(csv_assignments, Path(args.export_dir).expanduser())

    # This is the exact function called at train_baseline.py:798, including its
    # BF16 autocast choice and its dataset-level count accumulation.
    training_amp = bool(device.type == 'cuda' and checkpoint.get('amp_enabled', False))
    amp_dtype = torch.bfloat16 if checkpoint.get('amp_dtype', 'bf16') == 'bf16' else torch.float16
    loss_args = SimpleNamespace(
        mse_weight=float(saved_args.get('mse_weight', 0.4)),
        bce_weight=float(saved_args.get('bce_weight', 0.7)),
    )
    loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )
    _, direct_training_scores = training_validate(
        model, loader, device, nn.MSELoss(), nn.BCELoss(), loss_args, training_amp, amp_dtype,
    )
    print(f'train_baseline.validate (same metric path): {_format_scores(direct_training_scores)}')
    print(f'Checkpoint best_mean_dice comparison: direct={np.mean(direct_training_scores):.9f}, '
          f'checkpoint={checkpoint_best:.9f}, delta={np.mean(direct_training_scores) - checkpoint_best:+.9f}.')

    metrics_path = Path(args.metrics_csv).expanduser() if args.metrics_csv else checkpoint_path.parent.parent / 'metrics.csv'
    metrics_row = _metrics_row(metrics_path, best_epoch)
    if metrics_row is None:
        print(f'metrics.csv epoch-{best_epoch} check: not available at {metrics_path}.')
    else:
        print('metrics.csv epoch check: '
              f'mean={float(metrics_row["mean_dice"]):.9f}, WT={float(metrics_row["dice_wt"]):.9f}, '
              f'TC={float(metrics_row["dice_tc"]):.9f}, ET={float(metrics_row["dice_et"]):.9f}.')

    bf16_direct, bf16_roundtrip, fp32_direct, fp32_roundtrip, stored_roundtrip = (
        DiceAccumulator(), DiceAccumulator(), DiceAccumulator(), DiceAccumulator(), DiceAccumulator(),
    )
    stored_macro = MacroDiceAccumulator()
    bf16_macro = MacroDiceAccumulator()
    fp32_macro = MacroDiceAccumulator()
    conversion_mismatches = {'bf16_disjoint': 0, 'bf16_regions': 0, 'fp32_disjoint': 0, 'fp32_regions': 0}
    source_ground_truth_mismatches = 0
    source_mri_mismatches = 0
    stored_fp32_prediction_mismatches = 0
    stored_bf16_prediction_mismatches = 0

    # A second deterministic pass exposes both numerical precision modes and
    # compares every on-disk pickle.  It remains read-only.
    model.eval()
    with torch.no_grad():
        for subject_names, images, segmentation in loader:
            subject_id = str(subject_names[0])
            image = torch.cat([modality.to(device, non_blocking=True) for modality in images], dim=1)
            target = targets_from_segmentation(segmentation.to(device, non_blocking=True)).cpu()

            # Training validation had explicitly configured TF32/cuDNN state.
            _configure_checkpoint_cuda(checkpoint, device)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=training_amp):
                bf16_probabilities = model(image)
            # A standalone exporter does not configure those flags or enter
            # autocast.  Restore its fresh-process defaults before matching it.
            _restore_cuda_settings(exporter_cuda_defaults)
            fp32_probabilities = model(image)

            bf16_disjoint = probs_to_preds(bf16_probabilities, 'overlapping')
            fp32_disjoint = probs_to_preds(fp32_probabilities, 'overlapping')
            bf16_regions = disjoint_to_overlapping(bf16_disjoint)
            fp32_regions = disjoint_to_overlapping(fp32_disjoint)
            bf16_labels, bf16_reconstructed = _round_trip_export_labels(bf16_disjoint)
            fp32_labels, fp32_reconstructed = _round_trip_export_labels(fp32_disjoint)

            conversion_mismatches['bf16_disjoint'] += int(torch.count_nonzero(
                bf16_disjoint != seg_to_one_hot_channels(bf16_labels[None, None, ...])
            ).item())
            conversion_mismatches['bf16_regions'] += int(torch.count_nonzero(bf16_regions != bf16_reconstructed).item())
            conversion_mismatches['fp32_disjoint'] += int(torch.count_nonzero(
                fp32_disjoint != seg_to_one_hot_channels(fp32_labels[None, None, ...])
            ).item())
            conversion_mismatches['fp32_regions'] += int(torch.count_nonzero(fp32_regions != fp32_reconstructed).item())

            bf16_direct.add(bf16_regions, target)
            bf16_roundtrip.add(bf16_reconstructed, target)
            fp32_direct.add(fp32_regions, target)
            fp32_roundtrip.add(fp32_reconstructed, target)
            bf16_macro.add(bf16_reconstructed, target)
            fp32_macro.add(fp32_reconstructed, target)

            exported = _load_pickle(export_paths[subject_id])
            exported_mris = tuple(np.asarray(value) for value in exported[:4])
            exported_ground_truth = np.asarray(exported[4]).squeeze().astype(np.uint8, copy=False)
            exported_prediction = np.asarray(exported[5]).squeeze().astype(np.uint8, copy=False)
            source_ground_truth = segmentation[0, 0].numpy().astype(np.uint8, copy=False)
            if not np.array_equal(exported_ground_truth, source_ground_truth):
                source_ground_truth_mismatches += 1
            for modality, exported_mri in zip(images, exported_mris):
                source_mri = modality[0, 0].numpy().astype(np.float32, copy=False)
                if not np.array_equal(exported_mri, source_mri):
                    source_mri_mismatches += 1
                    break
            if not np.array_equal(exported_prediction, fp32_labels.numpy()):
                stored_fp32_prediction_mismatches += 1
            if not np.array_equal(exported_prediction, bf16_labels.numpy()):
                stored_bf16_prediction_mismatches += 1

            stored_disjoint = seg_to_one_hot_channels(torch.from_numpy(exported_prediction)[None, None, ...])
            stored_regions = disjoint_to_overlapping(stored_disjoint)
            stored_roundtrip.add(stored_regions, target)
            stored_macro.add(stored_regions, target)

    _assert_scores_equal('BF16 direct vs train_baseline.validate', bf16_direct.result(), direct_training_scores)
    _assert_scores_equal('BF16 direct vs 0/1/2/3 round trip', bf16_direct.result(), bf16_roundtrip.result())
    _assert_scores_equal('FP32 direct vs 0/1/2/3 round trip', fp32_direct.result(), fp32_roundtrip.result())
    _assert_scores_equal('Stored export vs current FP32 round trip', stored_roundtrip.result(), fp32_roundtrip.result())
    print(f'BF16 dataset-level Dice: {_format_scores(bf16_direct.result())}')
    print(f'FP32 dataset-level Dice (export inference): {_format_scores(fp32_direct.result())}')
    print(f'Stored export dataset-level Dice: {_format_scores(stored_roundtrip.result())}')
    print(f'Stored export per-case macro Dice (Re-DiffiNet convention): {_format_scores(stored_macro.result())}')
    print(f'FP32 export-style per-case macro Dice: {_format_scores(fp32_macro.result())}')
    print(f'BF16 per-case macro Dice: {_format_scores(bf16_macro.result())}')
    print('Per-case macro empty/empty cases skipped (WT, TC, ET): '
          f'{tuple(stored_macro.skipped_empty)}.')
    print(f'Prediction conversion mismatched elements: {conversion_mismatches}.')
    print(f'Export/source MRI mismatched cases: {source_mri_mismatches}/{len(validation_dataset)}.')
    print(f'Export/source ground-truth mismatched cases: {source_ground_truth_mismatches}/{len(validation_dataset)}.')
    print(f'Stored labels differing from regenerated FP32 exporter labels: '
          f'{stored_fp32_prediction_mismatches}/{len(validation_dataset)}.')
    print(f'Stored labels differing from BF16 validation labels: '
          f'{stored_bf16_prediction_mismatches}/{len(validation_dataset)}.')


if __name__ == '__main__':
    main()
