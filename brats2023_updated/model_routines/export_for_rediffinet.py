"""Export a trained U-Net fold in the six-array pickle format used by Re-DiffiNet.

The fold-assignment CSV written by ``train_baseline.py`` is authoritative. Its
``source_directory`` identifies a case's original location; ``fold_split``
identifies its cross-validation output split.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..datasets.brats_dataset import BratsDataset
from ..models.unet3d import U_Net3d
from ..processing.preprocess import DEFAULT_CROP_SIZE
from ..utils.general_utils import one_hot_channels_to_three_labels, probs_to_preds


REQUIRED_ASSIGNMENT_COLUMNS = frozenset({'source_directory', 'subject_id', 'fold_split'})
VALID_SOURCE_DIRECTORIES = frozenset({'train', 'validation'})
VALID_FOLD_SPLITS = frozenset({'training', 'validation'})
EXPECTED_MODEL_CLASS = 'brats2023_updated.models.unet3d.U_Net3d'
OUTPUT_SUFFIXES = ('.pkl.gz', '.pkl')


@dataclass(frozen=True)
class FoldAssignment:
    """One source case and its trained cross-validation membership."""

    source_directory: str
    subject_id: str
    fold_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Baseline checkpoint (best_mean_dice.pt is recommended).')
    parser.add_argument('--train-dir', required=True, help='Original BraTS training split.')
    parser.add_argument('--val-dir', required=True, help='Original BraTS validation split.')
    parser.add_argument(
        '--fold-assignments', required=True,
        help='fold_assignments.csv generated for the checkpointed cross-validation fold.',
    )
    parser.add_argument('--out-dir', required=True, help='Output root; creates train/ and validation/.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--overwrite', action='store_true', help='Regenerate existing pickle files.')
    parser.add_argument(
        '--compression', choices=('none', 'gzip'), default='gzip',
        help='Lossless export compression (default: gzip).',
    )
    return parser.parse_args()


def _output_suffix(compression: str) -> str:
    if compression == 'gzip':
        return '.pkl.gz'
    if compression == 'none':
        return '.pkl'
    raise ValueError(f'Unsupported export compression: {compression!r}.')


def _output_path(out_dir: Path, output_split: str, subject_id: str, compression: str) -> Path:
    return out_dir / output_split / f'{subject_id}{_output_suffix(compression)}'


def atomic_pickle_dump(value: object, destination: Path, compression: str) -> None:
    """Write a pickle atomically so an interruption cannot leave a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        if compression == 'gzip':
            with gzip.open(temp_path, 'wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif compression == 'none':
            with temp_path.open('wb') as handle:
                pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f'Unsupported export compression: {compression!r}.')
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_manifest_write(assignments: list[FoldAssignment], out_dir: Path, compression: str) -> None:
    """Write the exact source-to-output mapping used for this export."""
    destination = out_dir / 'export_manifest.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix='.tmp', mode='w', newline='', encoding='utf-8', delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=['subject_id', 'source_directory', 'fold_split', 'output_path'],
        )
        writer.writeheader()
        for assignment in assignments:
            output_split = 'train' if assignment.fold_split == 'training' else 'validation'
            writer.writerow({
                'subject_id': assignment.subject_id,
                'source_directory': assignment.source_directory,
                'fold_split': assignment.fold_split,
                'output_path': str(_output_path(out_dir, output_split, assignment.subject_id, compression)),
            })
    try:
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalise_crop_size(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f'Checkpoint crop_size must be three integers, not {value!r}.')
    try:
        crop_size = tuple(int(item) for item in value)  # type: ignore[union-attr]
    except TypeError as error:
        raise ValueError('Checkpoint crop_size must contain three integers.') from error
    if len(crop_size) != 3 or any(item < 1 for item in crop_size):
        raise ValueError(f'Checkpoint crop_size must contain three positive integers, got {crop_size}.')
    return crop_size


def _checkpoint_crop_size(checkpoint: Mapping[str, object]) -> tuple[int, int, int] | None:
    """Use the saved training crop, retaining the original fixed crop for ``None``."""
    crop_values: list[tuple[int, int, int] | None] = []
    for key in ('args', 'run_config'):
        saved_config = checkpoint.get(key)
        if isinstance(saved_config, Mapping) and 'crop_size' in saved_config:
            crop_values.append(_normalise_crop_size(saved_config['crop_size']))
    if not crop_values:
        return None
    if any(crop_size != crop_values[0] for crop_size in crop_values[1:]):
        raise ValueError(f'Checkpoint contains conflicting crop_size values: {crop_values}.')
    return crop_values[0]


def _state_tensor(state: Mapping[str, object], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f'Checkpoint model state is missing tensor {key!r}; cannot verify the saved U-Net architecture.'
        )
    return value


def _single_metadata_value(
    checkpoint: Mapping[str, object], architecture: Mapping[str, object] | None, key: str,
) -> int | None:
    """Return one mutually consistent integer from top-level/config metadata."""
    values: list[int] = []
    for source in (architecture, checkpoint):
        if source is not None and source.get(key) is not None:
            try:
                values.append(int(source[key]))
            except (TypeError, ValueError) as error:
                raise ValueError(f'Checkpoint {key} metadata must be an integer.') from error
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f'Checkpoint contains conflicting {key} metadata: {values}.')
    return values[0]


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[U_Net3d, tuple[int, int, int] | None]:
    """Rebuild the saved U-Net from metadata and verify its model-state widths.

    A checkpoint without ``base_channels`` metadata is deliberately rejected:
    guessing would silently load Fold-0's base-32 weights into a wrong model.
    """
    checkpoint_value: Any = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint_value, Mapping):
        raise TypeError('Checkpoint must be a metadata dictionary, not a bare model state dictionary.')
    checkpoint: Mapping[str, object] = checkpoint_value
    state_value = checkpoint.get('model_state_dict', checkpoint.get('model_state', checkpoint.get('model_sd')))
    if not isinstance(state_value, Mapping):
        raise KeyError('Checkpoint does not contain a model_state_dict, model_state, or model_sd mapping.')
    state: Mapping[str, object] = state_value

    raw_architecture = checkpoint.get('architecture_config')
    if raw_architecture is not None and not isinstance(raw_architecture, Mapping):
        raise TypeError('Checkpoint architecture_config must be a mapping when present.')
    architecture = raw_architecture if isinstance(raw_architecture, Mapping) else None
    model_class = architecture.get('model_class') if architecture is not None else checkpoint.get('architecture')
    if model_class is not None and model_class != EXPECTED_MODEL_CLASS:
        raise ValueError(f'Unsupported checkpoint architecture {model_class!r}; expected {EXPECTED_MODEL_CLASS!r}.')

    base_channels = _single_metadata_value(checkpoint, architecture, 'base_channels')
    if base_channels is None:
        raise ValueError(
            'Checkpoint is missing base_channels architecture metadata. Refusing to guess a U-Net width; '
            'export a checkpoint written by the current baseline trainer.'
        )
    first_layer = _state_tensor(state, 'Conv1.conv.1.weight')
    head_layer = _state_tensor(state, 'Conv_1x11.weight')
    if first_layer.ndim != 5 or head_layer.ndim != 5:
        raise ValueError('Checkpoint U-Net convolution weights must be 5-D tensors.')
    inferred_base_channels, inferred_img_ch = int(first_layer.shape[0]), int(first_layer.shape[1])
    inferred_output_ch = int(head_layer.shape[0])
    if base_channels != inferred_base_channels:
        raise ValueError(
            'Checkpoint base_channels metadata does not match Conv1 weights: '
            f'metadata={base_channels}, weights={inferred_base_channels}.'
        )
    img_ch = _single_metadata_value(checkpoint, architecture, 'img_ch')
    output_ch = _single_metadata_value(checkpoint, architecture, 'output_ch')
    img_ch = inferred_img_ch if img_ch is None else img_ch
    output_ch = inferred_output_ch if output_ch is None else output_ch
    if img_ch != inferred_img_ch or output_ch != inferred_output_ch:
        raise ValueError('Checkpoint input/output metadata does not match saved model-state tensor shapes.')
    if img_ch != 4 or output_ch != 3:
        raise ValueError(
            'Re-DiffiNet export requires the baseline U-Net convention of four MRI inputs and '
            f'three WT/TC/ET outputs; checkpoint has img_ch={img_ch}, output_ch={output_ch}.'
        )

    model = U_Net3d(img_ch=img_ch, output_ch=output_ch, base_channels=base_channels)
    if architecture is not None and architecture.get('channel_widths') is not None:
        try:
            saved_widths = tuple(int(width) for width in architecture['channel_widths'])
        except (TypeError, ValueError) as error:
            raise ValueError('Checkpoint channel_widths metadata must be an integer sequence.') from error
        if saved_widths != model.channel_widths:
            raise ValueError(
                f'Checkpoint channel_widths {saved_widths} do not match base_channels={base_channels} '
                f'({model.channel_widths}).'
            )
    model.to(device)
    model.load_state_dict(state)  # PyTorch verifies every saved tensor key and shape.
    model.eval()
    return model, _checkpoint_crop_size(checkpoint)


def _case_inventory(source_dir: Path, source_name: str) -> set[str]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f'{source_name} source directory does not exist: {source_dir}')
    cases = {entry.name for entry in source_dir.iterdir() if entry.is_dir()}
    if not cases:
        raise RuntimeError(f'No case directories found in {source_name} source directory: {source_dir}')
    return cases


def _validate_subject_id(subject_id: str, line_number: int) -> None:
    candidate = Path(subject_id)
    if not subject_id or candidate.name != subject_id or subject_id in {'.', '..'}:
        raise ValueError(f'Invalid subject_id at CSV row {line_number}: {subject_id!r}.')


def read_fold_assignments(
    assignment_path: str | Path, source_dirs: Mapping[str, Path],
) -> list[FoldAssignment]:
    """Read and completely validate the CSV against the actual input cases."""
    assignment_path = Path(assignment_path)
    if not assignment_path.is_file():
        raise FileNotFoundError(f'Fold-assignment CSV does not exist: {assignment_path}')
    inventories = {
        source_name: _case_inventory(source_dirs[source_name], source_name)
        for source_name in VALID_SOURCE_DIRECTORIES
    }
    source_subject_overlap = inventories['train'] & inventories['validation']
    if source_subject_overlap:
        examples = ', '.join(sorted(source_subject_overlap)[:5])
        raise ValueError(
            'A subject ID is present in both original source directories, making output names ambiguous: '
            f'{examples}.'
        )

    assignments: list[FoldAssignment] = []
    with assignment_path.open('r', newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_ASSIGNMENT_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f'Fold-assignment CSV is missing required columns: {", ".join(sorted(missing_columns))}.'
            )
        for line_number, row in enumerate(reader, start=2):
            source_directory = (row.get('source_directory') or '').strip()
            subject_id = (row.get('subject_id') or '').strip()
            fold_split = (row.get('fold_split') or '').strip()
            if source_directory not in VALID_SOURCE_DIRECTORIES:
                raise ValueError(
                    f'Invalid source_directory at CSV row {line_number}: {source_directory!r}; '
                    f'expected one of {sorted(VALID_SOURCE_DIRECTORIES)}.'
                )
            if fold_split not in VALID_FOLD_SPLITS:
                raise ValueError(
                    f'Invalid fold_split at CSV row {line_number}: {fold_split!r}; '
                    f'expected one of {sorted(VALID_FOLD_SPLITS)}.'
                )
            _validate_subject_id(subject_id, line_number)
            assignments.append(FoldAssignment(source_directory, subject_id, fold_split))

    if not assignments:
        raise ValueError('Fold-assignment CSV contains no data rows.')
    subject_ids = [assignment.subject_id for assignment in assignments]
    duplicates = sorted({subject_id for subject_id in subject_ids if subject_ids.count(subject_id) > 1})
    if duplicates:
        raise ValueError('Duplicate subject IDs in fold-assignment CSV: ' + ', '.join(duplicates[:10]) + '.')

    assigned_cases = {(assignment.source_directory, assignment.subject_id) for assignment in assignments}
    actual_cases = {
        (source_name, subject_id)
        for source_name, subject_ids_for_source in inventories.items()
        for subject_id in subject_ids_for_source
    }
    missing_assignments = actual_cases - assigned_cases
    unknown_assignments = assigned_cases - actual_cases
    if missing_assignments or unknown_assignments or len(assignments) != len(actual_cases):
        details: list[str] = []
        if missing_assignments:
            details.append('missing CSV rows for ' + ', '.join(
                f'{source}/{subject}' for source, subject in sorted(missing_assignments)[:5]
            ))
        if unknown_assignments:
            details.append('CSV rows without source cases ' + ', '.join(
                f'{source}/{subject}' for source, subject in sorted(unknown_assignments)[:5]
            ))
        details.append(f'CSV rows={len(assignments)}, actual source cases={len(actual_cases)}')
        raise ValueError('Fold assignments must correspond exactly to input cases: ' + '; '.join(details) + '.')

    training_ids = {assignment.subject_id for assignment in assignments if assignment.fold_split == 'training'}
    validation_ids = {assignment.subject_id for assignment in assignments if assignment.fold_split == 'validation'}
    overlap = training_ids & validation_ids
    if overlap:
        raise ValueError('Data leakage: subject IDs appear in both output splits: ' + ', '.join(sorted(overlap)[:10]) + '.')
    return assignments


def _expected_subjects_by_output(assignments: list[FoldAssignment]) -> dict[str, set[str]]:
    return {
        'train': {assignment.subject_id for assignment in assignments if assignment.fold_split == 'training'},
        'validation': {assignment.subject_id for assignment in assignments if assignment.fold_split == 'validation'},
    }


def _output_subject_id(path: Path, suffix: str) -> str:
    if not path.name.endswith(suffix):
        raise ValueError(f'Output file does not end in {suffix!r}: {path}')
    return path.name[:-len(suffix)]


def validate_output_tree(
    out_dir: Path, assignments: list[FoldAssignment], compression: str,
) -> dict[str, set[str]]:
    """Reject stale files that could leak cases between folds during a resume."""
    expected = _expected_subjects_by_output(assignments)
    selected_suffix = _output_suffix(compression)
    existing: dict[str, set[str]] = {}
    for output_split in ('train', 'validation'):
        split_dir = out_dir / output_split
        if split_dir.exists() and not split_dir.is_dir():
            raise ValueError(f'Expected output directory but found a file: {split_dir}')
        paths = (
            [path for suffix in OUTPUT_SUFFIXES for path in split_dir.glob(f'*{suffix}')]
            if split_dir.is_dir() else []
        )
        existing[output_split] = set()
        unexpected: set[str] = set()
        for path in paths:
            suffix = next(suffix for suffix in OUTPUT_SUFFIXES if path.name.endswith(suffix))
            subject_id = _output_subject_id(path, suffix)
            if suffix != selected_suffix or subject_id not in expected[output_split]:
                unexpected.add(path.name)
            else:
                existing[output_split].add(subject_id)
        if unexpected:
            raise ValueError(
                f'Output directory {split_dir} contains unexpected export files for this fold/split: '
                + ', '.join(sorted(unexpected)[:10]) + '.'
            )
    overlap = existing['train'] & existing['validation']
    if overlap:
        raise ValueError(
            'Data leakage: pickle files already appear in both train and validation outputs: '
            + ', '.join(sorted(overlap)[:10]) + '.'
        )
    return existing


def _loader_kwargs(workers: int, device: torch.device) -> dict[str, object]:
    kwargs: dict[str, object] = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': workers,
        'pin_memory': device.type == 'cuda',
    }
    if workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def _validate_export_arrays(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], subject_id: str,
) -> None:
    """Keep the Re-DiffiNet tuple order and cropped spatial shape explicit."""
    t1c, t1n, t2f, t2w, ground_truth, baseline_prediction = values
    spatial_shape = t1c.shape
    if len(spatial_shape) != 3 or any(
        array.shape != spatial_shape for array in (t1n, t2f, t2w, ground_truth, baseline_prediction)
    ):
        raise RuntimeError(
            f'Unexpected exported shapes for {subject_id}: '
            f't1c={t1c.shape}, t1n={t1n.shape}, t2f={t2f.shape}, t2w={t2w.shape}, '
            f'ground_truth={ground_truth.shape}, baseline_prediction={baseline_prediction.shape}.'
        )
    if any(array.dtype != np.float32 for array in (t1c, t1n, t2f, t2w)):
        raise RuntimeError(f'MRI arrays for {subject_id} must be float32.')
    for name, label_array in (('ground truth', ground_truth), ('baseline prediction', baseline_prediction)):
        if label_array.dtype != np.uint8 or not np.all(np.isin(label_array, (0, 1, 2, 3))):
            raise RuntimeError(
                f'{name.title()} for {subject_id} must use Re-DiffiNet-compatible uint8 labels 0/1/2/3.'
            )


@torch.no_grad()
def export_assignments(
    assignments: list[FoldAssignment], source_dirs: Mapping[str, Path], out_dir: Path,
    model: U_Net3d, device: torch.device, workers: int, overwrite: bool,
    crop_size: tuple[int, int, int] | None, compression: str,
) -> None:
    """Export each assigned case from its original source to its fold output."""
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.fold_split, assignment.source_directory)].append(assignment.subject_id)

    datasets: dict[str, BratsDataset] = {}
    subject_indices: dict[str, dict[str, int]] = {}
    for source_name in VALID_SOURCE_DIRECTORIES:
        dataset = BratsDataset(str(source_dirs[source_name]), mode='train', crop_size=crop_size)
        datasets[source_name] = dataset
        subject_indices[source_name] = {subject_id: index for index, subject_id in enumerate(dataset.subject_list)}

    for fold_split in ('training', 'validation'):
        output_split = 'train' if fold_split == 'training' else 'validation'
        for source_name in ('train', 'validation'):
            subject_ids = grouped.get((fold_split, source_name), [])
            if not subject_ids:
                continue
            indices = [subject_indices[source_name][subject_id] for subject_id in subject_ids]
            loader = DataLoader(Subset(datasets[source_name], indices), **_loader_kwargs(workers, device))
            for subject_names, images, segmentation in loader:
                subject_id = subject_names[0]
                destination = _output_path(out_dir, output_split, subject_id, compression)
                if destination.exists() and not overwrite:
                    print(f'Skipping existing {destination.name}')
                    continue

                mri_arrays = tuple(
                    modality[0, 0].numpy().astype(np.float32, copy=False) for modality in images
                )
                model_input = torch.cat([modality.to(device, non_blocking=True) for modality in images], dim=1)
                probabilities = model(model_input)
                expected_probability_shape = (1, 3, *mri_arrays[0].shape)
                if tuple(probabilities.shape) != expected_probability_shape:
                    raise RuntimeError(
                        f'Unexpected U-Net output shape for {subject_id}: {tuple(probabilities.shape)}; '
                        f'expected {expected_probability_shape} (WT, TC, ET).'
                    )
                # Training uses overlapping output channels in WT, TC, ET order.
                # This conversion preserves their convention and emits Re-DiffiNet's
                # single-channel labels 0/1/2/3 (not BraTS label 4).
                prediction = one_hot_channels_to_three_labels(probs_to_preds(probabilities, 'overlapping')[0])
                ground_truth = segmentation[0, 0].numpy().astype(np.uint8, copy=False)
                baseline_prediction = np.asarray(prediction, dtype=np.uint8)
                exported_values = (*mri_arrays, ground_truth, baseline_prediction)
                _validate_export_arrays(exported_values, subject_id)
                atomic_pickle_dump(exported_values, destination, compression)
                print(f'Wrote {destination}')


def _print_assignment_summary(assignments: list[FoldAssignment]) -> None:
    training_count = sum(assignment.fold_split == 'training' for assignment in assignments)
    validation_count = sum(assignment.fold_split == 'validation' for assignment in assignments)
    training_ids = {assignment.subject_id for assignment in assignments if assignment.fold_split == 'training'}
    validation_ids = {assignment.subject_id for assignment in assignments if assignment.fold_split == 'validation'}
    print(f'Fold assignment rows: {len(assignments)}')
    print(f'Training exports: {training_count}')
    print(f'Validation exports: {validation_count}')
    print(f'Overlap: {len(training_ids & validation_ids)}')


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError('--workers must be at least 0.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if device.type == 'cuda' and device.index not in (None, 0):
        raise ValueError('This single-GPU exporter is intentionally restricted to cuda:0.')

    source_dirs = {'train': Path(args.train_dir), 'validation': Path(args.val_dir)}
    assignments = read_fold_assignments(args.fold_assignments, source_dirs)
    out_dir = Path(args.out_dir)
    validate_output_tree(out_dir, assignments, args.compression)
    model, crop_size = load_model(args.checkpoint, device)

    _print_assignment_summary(assignments)
    print(f'checkpoint base_channels: {model.base_channels}')
    crop_description = crop_size if crop_size is not None else f'original fixed crop {DEFAULT_CROP_SIZE}'
    print(f'crop size: {crop_description}')
    print(f'device: {device}')
    print(f'compression: {args.compression}')
    export_assignments(
        assignments, source_dirs, out_dir, model, device, args.workers, args.overwrite, crop_size, args.compression,
    )

    existing = validate_output_tree(out_dir, assignments, args.compression)
    expected = _expected_subjects_by_output(assignments)
    missing = {
        output_split: expected[output_split] - existing[output_split]
        for output_split in ('train', 'validation')
    }
    if missing['train'] or missing['validation']:
        raise RuntimeError(
            'Export ended with missing pickle files: '
            + '; '.join(
                f'{split}={", ".join(sorted(subject_ids)[:10])}'
                for split, subject_ids in missing.items() if subject_ids
            )
        )
    atomic_manifest_write(assignments, out_dir, args.compression)
    training_files = len(existing['train'])
    validation_files = len(existing['validation'])
    exported_cases = training_files + validation_files
    total_output_bytes = sum(
        _output_path(out_dir, output_split, subject_id, args.compression).stat().st_size
        for output_split, subject_ids in existing.items()
        for subject_id in subject_ids
    )
    average_size_bytes = total_output_bytes / exported_cases if exported_cases else 0
    print('Export completed')
    print(f'Training files: {training_files}')
    print(f'Validation files: {validation_files}')
    print(f'Total files: {exported_cases}')
    print(f'Exported cases: {exported_cases}')
    print(f'Total output size: {total_output_bytes / 1024 ** 3:.6f} GB')
    size_description = 'Average compressed size per case' if args.compression == 'gzip' else 'Average output size per case'
    print(f'{size_description}: {average_size_bytes / 1024 ** 2:.3f} MB')
    print(f'Overlap: {len(existing["train"] & existing["validation"])}')


if __name__ == '__main__':
    main()
