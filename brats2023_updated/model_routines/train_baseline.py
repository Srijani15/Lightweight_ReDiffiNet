"""Train the BraTS2023 baseline U-Net efficiently on one GPU and resume safely.

Run from the directory *above* ``brats2023_updated`` with ``python -m``.
Input directories must contain BraTS case folders, for example
``BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz`` and ``-seg.nii.gz``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torch.utils.tensorboard import SummaryWriter

from ..datasets.brats_dataset import BratsDataset
from ..models.unet3d import U_Net3d
from ..processing.preprocess import DEFAULT_CROP_SIZE
from ..utils.general_utils import disjoint_to_overlapping, probs_to_preds, seg_to_one_hot_channels


class DeterministicEpochSampler(Sampler[int]):
    """A reproducible shuffled sampler whose order is determined by epoch."""

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.dataset), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True, help='Folder containing training case folders.')
    parser.add_argument('--val-dir', required=True, help='Folder containing validation case folders.')
    parser.add_argument('--out-dir', required=True, help='Directory for checkpoints and CSV metrics.')
    parser.add_argument('--device', default='cuda:0', help='Single device to use (default: cuda:0).')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=6e-5)
    parser.add_argument('--decay-rate', type=float, default=0.995)
    parser.add_argument('--mse-weight', type=float, default=0.4)
    parser.add_argument('--bce-weight', type=float, default=0.7)
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Validate every N epochs (default: 5).')
    parser.add_argument('--early-stopping-patience', type=int, default=4,
                        help='Validation checks without sufficient Mean Dice improvement before stopping.')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Required Validation Mean Dice improvement.')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of deterministic cross-validation folds.')
    parser.add_argument('--fold-number', type=int, default=0,
                        help='Fold to train now (0-based). This script never loops through folds automatically.')
    parser.add_argument('--fold-source', choices=('all', 'train'), default='all',
                        help='Use train + validation labels (default), or train only, to build folds.')
    parser.add_argument('--base-channels', type=int, default=U_Net3d.ORIGINAL_BASE_CHANNELS,
                        help=(f'U-Net first encoder width; {U_Net3d.ORIGINAL_BASE_CHANNELS} exactly reproduces '
                              'the original architecture.'))
    parser.add_argument('--crop-size', nargs=3, type=int, metavar=('X', 'Y', 'Z'), default=None,
                        help=('Optional deterministic centred crop. Omit to use the original fixed '
                              f'crop {DEFAULT_CROP_SIZE} exactly.'))
    parser.add_argument('--amp', action='store_true',
                        help='Enable CUDA automatic mixed precision. It is disabled automatically on CPU.')
    parser.add_argument('--amp-dtype', choices=('fp16', 'bf16'), default='bf16',
                        help='Autocast datatype when --amp is enabled (default: bf16).')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Maximum gradient norm applied before each optimizer step (default: 1.0).')
    parser.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable TF32 CUDA matmul/cuDNN kernels when CUDA is used (default: enabled).')
    parser.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable cuDNN benchmark for fixed-size volumes when CUDA is used (default: enabled).')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Deterministically limit training cases for a smoke test only.')
    parser.add_argument('--max-val-samples', type=int, default=None,
                        help='Deterministically limit validation cases for a smoke test only.')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log Train/Batch_Loss every N optimizer steps; 0 disables batch-level logging.')
    parser.add_argument('--checkpoint-steps', type=int, default=100,
                        help='Save latest.pt after this many optimizer steps; 0 saves only after epochs.')
    parser.add_argument('--backup-interval', type=int, default=10,
                        help='Save a periodic backup checkpoint every N completed epochs.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', default=None,
                        help='Checkpoint to resume. Defaults to OUT_DIR/fold_N/model/latest.pt when it exists.')
    return parser.parse_args()


def atomic_torch_save(value: object, destination: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt latest.pt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json_save(value: dict[str, Any], destination: Path) -> None:
    """Atomically write small, human-readable run metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', mode='w', delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    try:
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def configure_cuda_performance(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    """Configure fast CUDA kernels without disabling cuDNN."""
    cuda_active = device.type == 'cuda'
    tf32_enabled = bool(cuda_active and args.tf32)
    cudnn_benchmark_enabled = bool(cuda_active and args.cudnn_benchmark)
    if cuda_active:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        # Fixed BraTS crop sizes benefit from cuDNN's algorithm search. Never
        # set torch.backends.cudnn.enabled: cuDNN stays enabled.
        torch.backends.cudnn.benchmark = cudnn_benchmark_enabled
    amp_enabled = bool(cuda_active and args.amp)
    autocast_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    return {
        'amp_enabled': amp_enabled,
        'amp_dtype': args.amp_dtype,
        'autocast_dtype': autocast_dtype,
        # BF16 has sufficient exponent range to train without loss scaling.
        'scaler_enabled': bool(amp_enabled and args.amp_dtype == 'fp16'),
        'tf32_enabled': tf32_enabled,
        'cudnn_benchmark_enabled': cudnn_benchmark_enabled,
    }


def _loader_kwargs(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'num_workers': args.workers,
        'pin_memory': device.type == 'cuda',
    }
    if args.workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return kwargs


def make_training_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DeterministicEpochSampler]:
    """Create one reusable loader; persistent workers then survive epochs."""
    sampler = DeterministicEpochSampler(dataset, args.seed)
    # Keep DataLoader worker seeding off the global torch RNG so restarting a
    # loader after a mid-epoch resume cannot perturb the restored model RNG.
    loader_generator = torch.Generator().manual_seed(args.seed + 10_001)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        generator=loader_generator,
        **_loader_kwargs(args, device),
    )
    return loader, sampler


def make_validation_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    loader_generator = torch.Generator().manual_seed(args.seed + 20_001)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=loader_generator,
        **_loader_kwargs(args, device),
    )


def targets_from_segmentation(segmentation: torch.Tensor) -> torch.Tensor:
    return disjoint_to_overlapping(seg_to_one_hot_channels(segmentation))


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mse: nn.Module,
    bce: nn.Module,
    args: argparse.Namespace,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the published MSE + BCE loss and overlapping WT/TC/ET targets."""
    mse_loss = sum(mse(output[:, channel:channel + 1], target[:, channel:channel + 1]) for channel in range(3))
    bce_loss = sum(bce(output[:, channel:channel + 1], target[:, channel:channel + 1]) for channel in range(3))
    total_loss = args.mse_weight * mse_loss + args.bce_weight * bce_loss
    if return_components:
        return total_loss, mse_loss, bce_loss
    return total_loss


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse: nn.Module,
    bce: nn.Module,
    args: argparse.Namespace,
    amp_enabled: bool,
    autocast_dtype: torch.dtype,
) -> tuple[float, tuple[float, float, float]]:
    """Validate with autocast when enabled; Dice is accumulated in float64."""
    model.eval()
    losses: list[float] = []
    intersections = torch.zeros(3, dtype=torch.float64)
    prediction_sizes = torch.zeros(3, dtype=torch.float64)
    target_sizes = torch.zeros(3, dtype=torch.float64)

    with torch.no_grad():
        for _, images, segmentation in loader:
            image = torch.cat([modality.to(device, non_blocking=True) for modality in images], dim=1)
            target = targets_from_segmentation(segmentation.to(device, non_blocking=True))
            with torch.amp.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                output = model(image)
        
            with torch.amp.autocast(
                device_type="cuda",
                enabled=False,
            ):
                loss, mse_loss, bce_loss = compute_loss(
                    output.float(),
                    target.float(),
                    mse,
                    bce,
                    args,
                    return_components=True,
                )
            losses.append(loss.item())

            # Keep the repository's original thresholds, hierarchy, and Dice convention.
            predicted_disjoint = probs_to_preds(output, 'overlapping')
            predicted = disjoint_to_overlapping(predicted_disjoint).detach().cpu().to(torch.float64)
            truth = target.detach().cpu().to(torch.float64)
            intersections += (predicted * truth).sum(dim=(0, 2, 3, 4))
            prediction_sizes += predicted.sum(dim=(0, 2, 3, 4))
            target_sizes += truth.sum(dim=(0, 2, 3, 4))

    if not losses:
        raise RuntimeError('Validation loader produced no batches.')
    dice = (2.0 * intersections + 1e-8) / (prediction_sizes + target_sizes + 1e-8)
    return float(np.mean(losses)), tuple(float(value) for value in dice)


def make_cross_validation_datasets(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, list[tuple[str, str, str]]]:
    """Build one deterministic fold without putting a case in both splits."""
    sources = [('train', BratsDataset(args.train_dir, mode='train', crop_size=args.crop_size))]
    if args.fold_source == 'all':
        sources.append(('validation', BratsDataset(args.val_dir, mode='train', crop_size=args.crop_size)))

    pool: Dataset = sources[0][1] if len(sources) == 1 else ConcatDataset([dataset for _, dataset in sources])
    case_ids: list[tuple[str, str]] = []
    for source_name, dataset in sources:
        case_ids.extend((source_name, subject_name) for subject_name in dataset.subject_list)
    if args.n_folds > len(case_ids):
        raise ValueError(f'n-folds ({args.n_folds}) cannot exceed the number of cases ({len(case_ids)}).')

    shuffled_indices = np.arange(len(case_ids))
    np.random.default_rng(args.seed).shuffle(shuffled_indices)
    validation_indices = np.array_split(shuffled_indices, args.n_folds)[args.fold_number]
    validation_set = set(validation_indices.tolist())
    training_indices = [index for index in range(len(case_ids)) if index not in validation_set]
    validation_indices = sorted(validation_set)
    assignments = [
        (source_name, subject_name, 'validation' if index in validation_set else 'training')
        for index, (source_name, subject_name) in enumerate(case_ids)
    ]
    return Subset(pool, training_indices), Subset(pool, validation_indices), assignments


def limit_dataset(dataset: Dataset, maximum: int | None, split_name: str) -> Dataset:
    """Deterministically retain the first selected fold cases for a smoke test."""
    if maximum is None:
        return dataset
    if maximum < 1:
        raise ValueError(f'max-{split_name}-samples must be at least 1 when supplied.')
    return Subset(dataset, range(min(maximum, len(dataset))))


def write_fold_assignments(path: Path, assignments: list[tuple[str, str, str]]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source_directory', 'subject_id', 'fold_split'])
        writer.writerows(assignments)


def update_cross_validation_summary(path: Path, row: dict[str, object]) -> None:
    """Upsert one fold's best result while retaining every other fold's row."""
    fieldnames = ['fold', 'best_epoch', 'wt_dice', 'tc_dice', 'et_dice', 'mean_dice']
    existing_rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline='') as handle:
            existing_rows = list(csv.DictReader(handle))
    rows = [existing for existing in existing_rows if int(existing['fold']) != int(row['fold'])]
    rows.append({field: str(row[field]) for field in fieldnames})
    rows.sort(key=lambda value: int(value['fold']))

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', mode='w', newline='', delete=False) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _as_cpu_byte_tensor(value: object, state_name: str) -> torch.Tensor:
    """Normalise torch.load CPU/CUDA RNG values to torch.ByteTensor on CPU."""
    try:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device='cpu', dtype=torch.uint8).contiguous()
        else:
            tensor = torch.as_tensor(value, dtype=torch.uint8, device='cpu').contiguous()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(f'{state_name} cannot be converted to a CPU torch.ByteTensor.') from error
    if tensor.dtype != torch.uint8 or tensor.device.type != 'cpu':
        raise TypeError(f'{state_name} must be a CPU torch.ByteTensor.')
    return tensor


def restore_rng_state(checkpoint: dict[str, object], device: torch.device) -> None:
    """Restore all available RNG state without device-mapped ByteTensor errors."""
    if checkpoint.get('python_rng_state') is not None:
        random.setstate(checkpoint['python_rng_state'])
    if checkpoint.get('numpy_rng_state') is not None:
        np.random.set_state(checkpoint['numpy_rng_state'])
    if checkpoint.get('torch_rng_state') is not None:
        torch.set_rng_state(_as_cpu_byte_tensor(checkpoint['torch_rng_state'], 'torch_rng_state'))

    cuda_states = checkpoint.get('cuda_rng_state_all')
    if device.type == 'cuda' and cuda_states is not None:
        if not isinstance(cuda_states, (list, tuple)):
            raise TypeError('cuda_rng_state_all must be a list or tuple of RNG states.')
        states = [_as_cpu_byte_tensor(state, 'cuda_rng_state_all entry') for state in cuda_states]
        torch.cuda.set_rng_state_all(states)


def architecture_config(model: U_Net3d) -> dict[str, object]:
    return {
        'model_class': 'brats2023_updated.models.unet3d.U_Net3d',
        'img_ch': model.img_ch,
        'output_ch': model.output_ch,
        'base_channels': model.base_channels,
        'channel_widths': list(model.channel_widths),
    }


def checkpoint_payload(
    model: U_Net3d,
    optimizer: Adam,
    scaler: torch.amp.GradScaler,
    runtime_config: dict[str, Any],
    next_epoch: int,
    next_batch: int,
    best_validation_loss: float,
    best_mean_dice: float,
    best_epoch: int,
    best_dice_scores: tuple[float, float, float],
    global_step: int,
    early_stopping_counter: int,
    training_time_seconds: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Return all state needed to resume the same single-GPU experiment."""
    model_config = architecture_config(model)
    return {
        'format_version': 5,
        'architecture': model_config['model_class'],
        'architecture_config': model_config,
        'base_channels': model.base_channels,
        'channel_widths': list(model.channel_widths),
        'epoch': next_epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'grad_scaler_state_dict': scaler.state_dict() if runtime_config['scaler_enabled'] else None,
        'amp_enabled': runtime_config['amp_enabled'],
        'amp_dtype': runtime_config['amp_dtype'],
        'tf32_enabled': runtime_config['tf32_enabled'],
        'cudnn_benchmark_enabled': runtime_config['cudnn_benchmark_enabled'],
        'scheduler_state_dict': None,
        'next_epoch': next_epoch,
        'next_batch': next_batch,
        'global_step': global_step,
        'fold_number': args.fold_number,
        'n_folds': args.n_folds,
        'fold_source': args.fold_source,
        'best_validation_loss': best_validation_loss,
        'best_mean_dice': best_mean_dice,
        'best_epoch': best_epoch,
        'early_stopping_counter': early_stopping_counter,
        'best_dice_scores': best_dice_scores,
        'training_time_seconds': training_time_seconds,
        'args': vars(args).copy(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
    }


def _checkpoint_base_channels(checkpoint: dict[str, object]) -> int:
    if checkpoint.get('base_channels') is not None:
        return int(checkpoint['base_channels'])
    config = checkpoint.get('architecture_config')
    if isinstance(config, dict) and config.get('base_channels') is not None:
        return int(config['base_channels'])
    # Version-3 baseline checkpoints predate width metadata and are the
    # historical model whose first encoder level has 64 channels (nf=32).
    return U_Net3d.ORIGINAL_BASE_CHANNELS


def _normalise_crop_size(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)  # type: ignore[arg-type]


def validate_resume_configuration(
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    model: U_Net3d,
    runtime_config: dict[str, Any],
) -> None:
    checkpoint_base = _checkpoint_base_channels(checkpoint)
    if checkpoint_base != model.base_channels:
        raise ValueError(
            f'Checkpoint base_channels={checkpoint_base} conflicts with requested '
            f'--base-channels {model.base_channels}. Use a matching architecture.'
        )
    expected_widths = checkpoint.get('channel_widths')
    if expected_widths is not None and tuple(int(value) for value in expected_widths) != model.channel_widths:
        raise ValueError('Checkpoint channel widths conflict with the requested U-Net architecture.')

    for key, requested in (
        ('fold_number', args.fold_number),
        ('n_folds', args.n_folds),
        ('fold_source', args.fold_source),
    ):
        saved = checkpoint.get(key)
        if saved is not None and str(saved) != str(requested):
            raise ValueError(f'Checkpoint {key}={saved!r} conflicts with requested {requested!r}.')

    checkpoint_args = checkpoint.get('args')
    if isinstance(checkpoint_args, dict):
        saved_seed = checkpoint_args.get('seed')
        if saved_seed is not None and int(saved_seed) != args.seed:
            raise ValueError('Checkpoint seed conflicts with --seed; this would change deterministic fold/order state.')
        if 'crop_size' in checkpoint_args:
            if _normalise_crop_size(checkpoint_args['crop_size']) != _normalise_crop_size(args.crop_size):
                raise ValueError('Checkpoint crop_size conflicts with --crop-size.')

    if 'amp_enabled' in checkpoint and bool(checkpoint['amp_enabled']) != runtime_config['amp_enabled']:
        raise ValueError(
            'Checkpoint AMP configuration conflicts with this run. Resume with the same --amp setting '
            '(or start a new output directory for a different precision experiment).'
        )
    saved_amp_dtype = checkpoint.get('amp_dtype')
    if saved_amp_dtype is None and isinstance(checkpoint_args, dict):
        saved_amp_dtype = checkpoint_args.get('amp_dtype')
    if saved_amp_dtype is not None and str(saved_amp_dtype) != runtime_config['amp_dtype']:
        raise ValueError(
            'Checkpoint AMP datatype conflicts with this run. Resume with the same --amp-dtype setting '
            '(or start a new output directory for a different precision experiment).'
        )


def append_metrics(path: Path, row: list[object]) -> None:
    is_new_file = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as handle:
        writer = csv.writer(handle)
        if is_new_file:
            writer.writerow([
                'epoch', 'training_loss', 'training_mse_loss', 'training_bce_loss',
                'validation_loss', 'mean_dice', 'dice_wt', 'dice_tc', 'dice_et',
            ])
        writer.writerow(row)


def build_run_config(
    args: argparse.Namespace,
    model: U_Net3d,
    device: torch.device,
    runtime_config: dict[str, Any],
    total_parameters: int,
    trainable_parameters: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        'architecture': architecture_config(model),
        'base_channels': model.base_channels,
        'total_parameters': total_parameters,
        'trainable_parameters': trainable_parameters,
        'amp_requested': bool(args.amp),
        'amp_enabled': runtime_config['amp_enabled'],
        'amp_dtype': runtime_config['amp_dtype'],
        'grad_clip': args.grad_clip,
        'tf32_enabled': runtime_config['tf32_enabled'],
        'cudnn_benchmark_enabled': runtime_config['cudnn_benchmark_enabled'],
        'batch_size': args.batch_size,
        'workers': args.workers,
        'learning_rate': args.learning_rate,
        'fold': args.fold_number,
        'n_folds': args.n_folds,
        'fold_source': args.fold_source,
        'seed': args.seed,
        'input_crop_shape': list(args.crop_size) if args.crop_size is not None else list(DEFAULT_CROP_SIZE),
        'crop_uses_original_fixed_ranges': args.crop_size is None,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'pytorch_version': torch.__version__,
        'cuda_build_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'device': str(device),
    }
    if device.type == 'cuda':
        config['gpu_name'] = torch.cuda.get_device_name(device)
    return config


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError('epochs and batch-size must be positive; workers cannot be negative.')
    if args.val_interval < 1 or args.backup_interval < 1:
        raise ValueError('val-interval and backup-interval must be positive.')
    if args.checkpoint_steps < 0 or args.log_interval < 0:
        raise ValueError('checkpoint-steps and log-interval cannot be negative.')
    if args.grad_clip <= 0:
        raise ValueError('grad-clip must be positive.')
    if args.early_stopping_patience < 1:
        raise ValueError('early-stopping-patience must be positive.')
    if args.n_folds < 2:
        raise ValueError('n-folds must be at least 2.')
    if not 0 <= args.fold_number < args.n_folds:
        raise ValueError(f'fold-number must be in [0, {args.n_folds - 1}].')
    if args.base_channels < 2 or args.base_channels % 2:
        raise ValueError('base-channels must be a positive even integer.')
    if args.crop_size is not None:
        crop_size = tuple(args.crop_size)
        if any(size < 128 or size % 64 for size in crop_size):
            raise ValueError(
                '--crop-size dimensions must be multiples of 64 and at least 128 for this six-level '
                'InstanceNorm U-Net (for example: --crop-size 128 128 128).'
            )

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available. Check the PyTorch CUDA installation.')
    if device.type == 'cuda' and device.index not in (None, 0):
        raise ValueError('This single-GPU entry point is intentionally restricted to logical cuda:0.')
    runtime_config = configure_cuda_performance(device, args)
    amp_enabled = runtime_config['amp_enabled']
    autocast_dtype = runtime_config['autocast_dtype']

    out_dir = Path(args.out_dir)
    fold_dir = out_dir / f'fold_{args.fold_number}'
    model_dir = fold_dir / 'model'
    tensorboard_dir = out_dir / 'tensorboard' / f'fold_{args.fold_number}'
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    latest_path = model_dir / 'latest.pt'
    best_loss_path = model_dir / 'best_validation_loss.pt'
    best_dice_path = model_dir / 'best_mean_dice.pt'
    metrics_path = fold_dir / 'metrics.csv'
    assignments_path = fold_dir / 'fold_assignments.csv'
    run_config_path = fold_dir / 'run_config.json'
    backup_dir = model_dir / 'backup_checkpoints'
    cross_validation_path = out_dir / 'cross_validation_summary.csv'

    set_seed(args.seed, device)
    train_dataset, validation_dataset, assignments = make_cross_validation_datasets(args)
    write_fold_assignments(assignments_path, assignments)
    train_dataset = limit_dataset(train_dataset, args.max_train_samples, 'train')
    validation_dataset = limit_dataset(validation_dataset, args.max_val_samples, 'val')
    train_loader, train_sampler = make_training_loader(train_dataset, args, device)
    validation_loader = make_validation_loader(validation_dataset, args, device)

    model = U_Net3d(base_channels=args.base_channels).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0, amsgrad=True)
    scaler = torch.amp.GradScaler(device=device.type, enabled=runtime_config['scaler_enabled'])
    mse, bce = nn.MSELoss(), nn.BCELoss()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    start_epoch, start_batch, global_step = 1, 0, 0
    # -inf makes the first scheduled validation a valid baseline checkpoint,
    # even if a smoke test obtains an all-zero Dice score.
    best_validation_loss, best_mean_dice = float('inf'), float('-inf')
    best_epoch, best_dice_scores = 0, (0.0, 0.0, 0.0)
    early_stopping_counter = 0
    training_time_seconds = 0.0

    resume_path = Path(args.resume) if args.resume else latest_path
    if resume_path.exists():
        # Keep RNG tensors on CPU when loading; restore_rng_state then handles
        # both old device-mapped files and ordinary CPU checkpoint tensors.
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f'Checkpoint {resume_path} is not a state dictionary.')
        validate_resume_configuration(checkpoint, args, model, runtime_config)
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        optimizer_state = checkpoint.get('optimizer_state_dict', checkpoint.get('optimizer_state'))
        if model_state is None or optimizer_state is None:
            raise KeyError('Checkpoint is missing model or optimizer state.')
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        if runtime_config['scaler_enabled'] and checkpoint.get('grad_scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        start_epoch = int(checkpoint['next_epoch']) if 'next_epoch' in checkpoint else int(checkpoint['epoch']) + 1
        start_batch = int(checkpoint.get('next_batch', 0))
        global_step = int(checkpoint.get('global_step', 0))
        best_validation_loss = float(checkpoint.get('best_validation_loss', best_validation_loss))
        best_mean_dice = float(checkpoint.get('best_mean_dice', best_mean_dice))
        best_epoch = int(checkpoint.get('best_epoch', best_epoch))
        best_dice_scores = tuple(float(value) for value in checkpoint.get('best_dice_scores', best_dice_scores))
        early_stopping_counter = int(checkpoint.get('early_stopping_counter', 0))
        training_time_seconds = float(checkpoint.get('training_time_seconds', 0.0))
        restore_rng_state(checkpoint, device)
        print(f'Resumed checkpoint: {resume_path}')

    run_config = build_run_config(args, model, device, runtime_config, total_parameters, trainable_parameters)
    atomic_json_save(run_config, run_config_path)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_scalar('Model/Total_Parameters', total_parameters, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_parameters, 0)
    writer.add_text('Model/Configuration', json.dumps(run_config, indent=2, sort_keys=True), 0)

    if best_epoch:
        update_cross_validation_summary(cross_validation_path, {
            'fold': args.fold_number,
            'best_epoch': best_epoch,
            'wt_dice': best_dice_scores[0],
            'tc_dice': best_dice_scores[1],
            'et_dice': best_dice_scores[2],
            'mean_dice': best_mean_dice,
        })

    print('---------------------------------------------------')
    print(f'Current fold: {args.fold_number} / {args.n_folds - 1} ({args.fold_source} source)')
    print(f'TensorBoard directory: {tensorboard_dir}')
    print(f'Model checkpoint directory: {model_dir}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(validation_dataset)}')
    if args.max_train_samples is not None or args.max_val_samples is not None:
        print('DEBUG SAMPLE LIMIT ACTIVE (not suitable for a full experiment).')
    print(f'Device: {device}')
    print(f'AMP={runtime_config["amp_enabled"]} ({runtime_config["amp_dtype"]}), '
          f'TF32={runtime_config["tf32_enabled"]}, '
          f'cuDNN benchmark={runtime_config["cudnn_benchmark_enabled"]}')
    print(f'U-Net base channels: {model.base_channels}; channel widths: {model.channel_widths}')
    print(f'Parameters: total={total_parameters:,}, trainable={trainable_parameters:,}')
    print(f'Starting epoch: {start_epoch}; best previous Dice: {best_mean_dice:.5f}')
    print('---------------------------------------------------')

    current_epoch, next_batch = start_epoch, start_batch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            next_batch = start_batch if epoch == start_epoch else 0
            train_sampler.set_epoch(epoch)
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate * (args.decay_rate ** (epoch - 1))

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start_time = time.perf_counter()
            train_start_time = time.perf_counter()
            model.train()
            epoch_losses: list[float] = []
            epoch_mse_losses: list[float] = []
            epoch_bce_losses: list[float] = []
            samples_seen = 0

            for batch_index, (_, images, segmentation) in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                image = torch.cat([modality.to(device, non_blocking=True) for modality in images], dim=1)
                target = targets_from_segmentation(segmentation.to(device, non_blocking=True))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=amp_enabled,
                ):
                    output = model(image)
                
                # Existing BCE loss must run in FP32.
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=False,
                ):
                    loss, mse_loss, bce_loss = compute_loss(
                        output.float(),
                        target.float(),
                        mse,
                        bce,
                        args,
                        return_components=True,
                    )
                if runtime_config['scaler_enabled']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                global_step += 1
                samples_seen += image.shape[0]
                epoch_losses.append(float(loss.item()))
                epoch_mse_losses.append(float(mse_loss.item()))
                epoch_bce_losses.append(float(bce_loss.item()))
                next_batch = batch_index + 1
                writer.add_scalar('Train/Gradient_Norm', float(gradient_norm), global_step)

                if args.log_interval and global_step % args.log_interval == 0:
                    writer.add_scalar('Train/Batch_Loss', float(loss.item()), global_step)

                if args.checkpoint_steps and next_batch % args.checkpoint_steps == 0:
                    atomic_torch_save(
                        checkpoint_payload(
                            model, optimizer, scaler, runtime_config, epoch, next_batch,
                            best_validation_loss, best_mean_dice, best_epoch, best_dice_scores,
                            global_step, early_stopping_counter, training_time_seconds, args, device,
                        ),
                        latest_path,
                    )
                    writer.flush()

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            train_time_seconds = time.perf_counter() - train_start_time
            training_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            training_mse_loss = float(np.mean(epoch_mse_losses)) if epoch_mse_losses else float('nan')
            training_bce_loss = float(np.mean(epoch_bce_losses)) if epoch_bce_losses else float('nan')
            learning_rate = float(optimizer.param_groups[0]['lr'])

            validation_loss, dice_scores = float('nan'), (float('nan'),) * 3
            mean_dice = float('nan')
            validation_time_seconds = 0.0
            did_validate = epoch % args.val_interval == 0
            if did_validate:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                validation_start_time = time.perf_counter()
                validation_loss, dice_scores = validate(
                    model, validation_loader, device, mse, bce, args, amp_enabled, autocast_dtype,
                )
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                validation_time_seconds = time.perf_counter() - validation_start_time
                mean_dice = float(np.mean(dice_scores))
                append_metrics(
                    metrics_path,
                    [epoch, training_loss, training_mse_loss, training_bce_loss, validation_loss, mean_dice, *dice_scores],
                )

            peak_memory_gb = (
                torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0
            )
            epoch_time_seconds = time.perf_counter() - epoch_start_time
            training_time_seconds += epoch_time_seconds
            samples_per_second = samples_seen / train_time_seconds if train_time_seconds > 0 else 0.0

            writer.add_scalar('Train/Loss', training_loss, epoch)
            writer.add_scalar('Train/MSE_Loss', training_mse_loss, epoch)
            writer.add_scalar('Train/BCE_Loss', training_bce_loss, epoch)
            writer.add_scalar('Train/Learning_Rate', learning_rate, epoch)
            writer.add_scalar('Run/Epoch', epoch, epoch)
            writer.add_scalar('Performance/Epoch_Time_Seconds', epoch_time_seconds, epoch)
            writer.add_scalar('Performance/Train_Time_Seconds', train_time_seconds, epoch)
            writer.add_scalar('Performance/Validation_Time_Seconds', validation_time_seconds, epoch)
            writer.add_scalar('Performance/Peak_GPU_Memory_GB', peak_memory_gb, epoch)
            writer.add_scalar('Performance/Samples_Per_Second', samples_per_second, epoch)

            should_stop = False
            if did_validate:
                writer.add_scalar('Validation/Loss', validation_loss, epoch)
                writer.add_scalar('Validation/WT_Dice', dice_scores[0], epoch)
                writer.add_scalar('Validation/TC_Dice', dice_scores[1], epoch)
                writer.add_scalar('Validation/ET_Dice', dice_scores[2], epoch)
                writer.add_scalar('Validation/Mean_Dice', mean_dice, epoch)

                is_best_validation_loss = validation_loss < best_validation_loss
                if is_best_validation_loss:
                    best_validation_loss = validation_loss

                is_best_mean_dice = mean_dice > best_mean_dice + args.min_delta
                if is_best_mean_dice:
                    best_mean_dice = mean_dice
                    best_epoch = epoch
                    best_dice_scores = dice_scores
                    early_stopping_counter = 0
                    update_cross_validation_summary(cross_validation_path, {
                        'fold': args.fold_number,
                        'best_epoch': best_epoch,
                        'wt_dice': dice_scores[0],
                        'tc_dice': dice_scores[1],
                        'et_dice': dice_scores[2],
                        'mean_dice': best_mean_dice,
                    })
                    writer.add_scalar('Best/Mean_Dice', best_mean_dice, epoch)
                    writer.add_scalar('Best/WT_Dice', dice_scores[0], epoch)
                    writer.add_scalar('Best/TC_Dice', dice_scores[1], epoch)
                    writer.add_scalar('Best/ET_Dice', dice_scores[2], epoch)
                else:
                    early_stopping_counter += 1
                    print(f'No significant Mean Dice improvement '
                          f'({early_stopping_counter}/{args.early_stopping_patience}).')

                # Save after both comparisons, so every saved best checkpoint
                # carries coherent best-loss, best-Dice, and early-stop state.
                current_payload = checkpoint_payload(
                    model, optimizer, scaler, runtime_config, epoch + 1, 0,
                    best_validation_loss, best_mean_dice, best_epoch, best_dice_scores,
                    global_step, early_stopping_counter, training_time_seconds, args, device,
                )
                if is_best_validation_loss:
                    atomic_torch_save(current_payload, best_loss_path)
                if is_best_mean_dice:
                    atomic_torch_save(current_payload, best_dice_path)

                writer.add_scalar('EarlyStopping/Counter', early_stopping_counter, epoch)
                should_stop = early_stopping_counter >= args.early_stopping_patience
                print(
                    f'Epoch {epoch}: train={training_loss:.5f}, val={validation_loss:.5f}, '
                    f'mean Dice={mean_dice:.5f}, time={epoch_time_seconds:.1f}s, '
                    f'peak GPU={peak_memory_gb:.2f} GB.'
                )
            else:
                print(
                    f'Epoch {epoch}: train={training_loss:.5f}, time={epoch_time_seconds:.1f}s, '
                    f'peak GPU={peak_memory_gb:.2f} GB.'
                )

            next_batch = 0
            completed_checkpoint = checkpoint_payload(
                model, optimizer, scaler, runtime_config, epoch + 1, 0,
                best_validation_loss, best_mean_dice, best_epoch, best_dice_scores,
                global_step, early_stopping_counter, training_time_seconds, args, device,
            )
            atomic_torch_save(completed_checkpoint, latest_path)
            if epoch % args.backup_interval == 0:
                atomic_torch_save(completed_checkpoint, backup_dir / f'epoch_{epoch:04d}.pt')
            writer.flush()
            start_batch = 0

            if should_stop:
                print(f'Early stopping triggered at epoch {epoch}. '
                      f'Best Mean Dice={best_mean_dice:.5f} at epoch {best_epoch}.')
                break
    except (KeyboardInterrupt, Exception):
        # This is always the state following the last completely applied
        # optimizer step, so rerunning the same command resumes safely.
        atomic_torch_save(
            checkpoint_payload(
                model, optimizer, scaler, runtime_config, current_epoch, next_batch,
                best_validation_loss, best_mean_dice, best_epoch, best_dice_scores,
                global_step, early_stopping_counter, training_time_seconds, args, device,
            ),
            latest_path,
        )
        writer.flush()
        print(f'Interrupted. Safe resume checkpoint written to {latest_path}.')
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
