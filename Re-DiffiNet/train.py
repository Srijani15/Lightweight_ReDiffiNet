"""Fold-isolated residual diffusion training. Never starts on import."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, help='Clean export root for this fold')
    p.add_argument('--assignments', required=True)
    p.add_argument('--verification', required=True, help='test.py verify-export output')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fold', type=int, choices=range(5), default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', choices=('off', 'bf16', 'fp16'), default='bf16')
    p.add_argument('--determinism', choices=('strict', 'warn-only', 'off'), default='warn-only',
                   help='Default warn-only permits unsupported CUDA operations with warnings; not fully deterministic')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--width', type=int, default=32, help='Original width=64; lighter width=32')
    p.add_argument('--sampling-steps', type=int, default=20)
    p.add_argument('--patch-size', type=int, nargs=3, default=[96, 96, 96])
    p.add_argument('--roi-size', type=int, nargs=3, default=[96, 96, 96])
    p.add_argument('--overlap', type=float, default=.25)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--residual-weight', type=float, default=.5)
    p.add_argument('--gate-weight', type=float, default=.1)
    p.add_argument('--preservation-weight', type=float, default=.05)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--val-every', type=int, default=2)
    p.add_argument('--patience', type=int, default=6, help='Completed validation checks without min-delta gain')
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--resume', action='store_true', help='Resume this fold latest.pt at its completed epoch')
    p.add_argument('--diagnostic', action='store_true', help='Separate directory; never enters CV summary')
    p.add_argument('--max-train-cases', type=int, default=0)
    p.add_argument('--max-val-cases', type=int, default=0)
    p.add_argument('--diagnostic-report', help='Successful CUDA diagnostic.json required for full training')
    return p


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def source_identity():
    from brats_data_utils_multi_label_5folds import sha256
    root = Path(__file__).resolve().parent
    files = ['train.py', 'test.py', 'testing_utils.py', 'brats_data_utils_multi_label_5folds.py',
             'unet/refinement.py', 'unet/basic_unet.py', 'unet/basic_unet_denose.py',
             'guided_diffusion/gaussian_diffusion.py', 'guided_diffusion/respace.py']
    return {name: sha256(root/name) for name in files}


def configuration(args):
    return dict(model=dict(width=args.width, sampling_steps=args.sampling_steps),
                patch_size=args.patch_size, roi_size=args.roi_size, overlap=args.overlap,
                amp=args.amp, determinism=args.determinism, seed=args.seed,
                batch_size=args.batch_size, workers=args.workers,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                residual_weight=args.residual_weight, gate_weight=args.gate_weight,
                preservation_weight=args.preservation_weight, grad_clip=args.grad_clip,
                val_every=args.val_every, patience=args.patience, min_delta=args.min_delta,
                diagnostic=args.diagnostic, max_train_cases=args.max_train_cases,
                max_val_cases=args.max_val_cases)


def validate_diagnostic_configuration(diagnostic_config, config):
    for key in ('patch_size', 'roi_size', 'amp', 'batch_size', 'overlap', 'determinism'):
        if diagnostic_config.get(key) != config[key]:
            raise ValueError(f'Diagnostic did not validate requested {key}')
    if diagnostic_config['model']['width'] != config['model']['width']:
        raise ValueError('Diagnostic used a different model width')


def check_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.val_every < 1 or args.patience < 1:
        raise ValueError('Invalid epochs/batch/workers/validation/patience')
    if args.width < 2 or not 2 <= args.sampling_steps <= 1000:
        raise ValueError('Invalid width or DDIM steps')
    if any(s < 32 or s % 16 for s in args.patch_size + args.roi_size) or not 0 <= args.overlap < 1:
        raise ValueError('Patch/ROI axes must be >=32, multiples of 16; overlap in [0,1)')
    if min(args.max_train_cases, args.max_val_cases, args.min_delta, args.weight_decay,
           args.residual_weight, args.gate_weight, args.preservation_weight) < 0:
        raise ValueError('Limits/weights must be nonnegative')
    if args.learning_rate <= 0 or args.grad_clip <= 0:
        raise ValueError('Learning rate and gradient clip must be positive')
    if not args.diagnostic and (args.max_train_cases or args.max_val_cases):
        raise ValueError('Subset limits require --diagnostic')
    if args.diagnostic and not (args.max_train_cases and args.max_val_cases):
        raise ValueError('Diagnostic requires explicit nonzero case limits')


def main():
    args = parser().parse_args()
    check_args(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import importlib.metadata
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from brats_data_utils_multi_label_5folds import FoldExport, ExportDataset, REGIONS
    from testing_utils import (FORMAT_VERSION, METRIC_VERSION, fold_paths, require_verification,
                               validate_checkpoint, runtime, evaluate, atomic_write, save_json,
                               save_csv, paired_row, update_summary)
    from testing_utils import configure_determinism
    from test import safe_output
    from unet.refinement import DiffUNet

    export = FoldExport(args.data_dir, args.assignments, args.fold)
    verification = require_verification(args.verification, export)
    safe_output(args.output_dir, export)
    config, sources, identity = configuration(args), source_identity(), export.identity()
    device, amp_dtype = runtime(args.device, args.amp)
    if device.type != 'cuda' and not args.diagnostic:
        raise ValueError('Full training is server CUDA only; CPU allowed only for explicit diagnostics')
    if not args.diagnostic:
        if not args.diagnostic_report:
            raise ValueError('Run the CUDA diagnostic first; pass its --diagnostic-report')
        diagnostic = json.loads(Path(args.diagnostic_report).expanduser().read_text(encoding='utf-8'))
        if (not diagnostic.get('passed') or not diagnostic.get('cuda_validated')
                or diagnostic.get('identity') != identity or diagnostic.get('sources') != sources):
            raise ValueError('Missing, failed or stale CUDA diagnostic for this fold/export/code')
        validate_diagnostic_configuration(diagnostic['config'], config)

    configure_determinism(args.seed, args.determinism)
    paths = fold_paths(args.output_dir, args.fold, args.diagnostic)
    latest = paths['checkpoints']/'latest.pt'
    best_path = paths['checkpoints']/'best_mean_dice.pt'
    if not args.resume and paths['fold'].exists() and any(paths['fold'].iterdir()):
        raise ValueError('Output fold already exists; use --resume or a new output root')
    if args.resume and not latest.is_file():
        raise ValueError('No latest.pt for this fold')

    model = DiffUNet(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp == 'fp16'))
    train_ds = ExportDataset(export, 'training', args.patch_size, args.max_train_cases)
    val_ds = ExportDataset(export, 'validation', limit=args.max_val_cases)
    loader_generator = torch.Generator()
    loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                         pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                         generator=loader_generator)
    if args.workers:
        # Recreate workers each epoch so crop streams resume at an epoch boundary exactly.
        loader_kwargs.update(prefetch_factor=1, persistent_workers=False)
    loader = DataLoader(train_ds, **loader_kwargs)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'monai', 'numpy', 'tensorboard')}
    if not args.diagnostic and diagnostic.get('versions') != versions:
        raise ValueError('Dependencies changed since CUDA diagnostic; rerun it')
    metadata = dict(versions=versions, device=str(device), cuda_build=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == 'cuda' else None,
                    training_predictions='in-sample, NOT out-of-fold',
                    validation_selection='same fold used for U-Net and refiner selection; not an independent test set',
                    verification=verification, sources=sources)
    completed, best_epoch, best_mean, best_metrics = 0, 0, None, None
    stop_reference, stalled, duration = None, 0, 0.
    history = []
    if args.resume:
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        validate_checkpoint(checkpoint, args.fold, identity, config)
        if checkpoint['sources'] != sources or checkpoint['versions'] != versions:
            raise ValueError('Code/dependencies changed since checkpoint; exact resume rejected')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        completed, best_epoch = checkpoint['epoch'], checkpoint['best_epoch']
        best_mean, best_metrics = checkpoint['best_mean_dice'], checkpoint['best_metrics']
        stop_reference, stalled = checkpoint['stop_reference'], checkpoint['stalled']
        duration, history = checkpoint['training_duration_seconds'], checkpoint['history']
        random.setstate(checkpoint['python_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device)
    paths['checkpoints'].mkdir(parents=True, exist_ok=True)
    save_json(paths['fold']/'run_config.json', dict(config=config, requested_epochs=args.epochs,
                                                   identity=identity, **metadata))

    def payload():
        return dict(format_version=FORMAT_VERSION, metric_version=METRIC_VERSION, fold=args.fold,
                    config=config, identity=identity, sources=sources, versions=versions,
                    diagnostic=args.diagnostic, epoch=completed, best_epoch=best_epoch,
                    requested_epochs=args.epochs,
                    best_mean_dice=best_mean, best_metrics=best_metrics, stop_reference=stop_reference,
                    stalled=stalled, training_duration_seconds=duration, history=history,
                    model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), grad_scaler_state_dict=scaler.state_dict(),
                    python_rng_state=random.getstate(), numpy_rng_state=np.random.get_state(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)

    def save_checkpoint(path):
        atomic_write(path, lambda tmp: torch.save(payload(), tmp))

    if not args.resume:
        save_checkpoint(latest)  # interruption in epoch 1 can safely replay it
    writer = SummaryWriter(str(paths['tensorboard']), purge_step=completed+1 if args.resume else None)
    interrupted = False
    try:
        for epoch in range(completed+1, args.epochs+1):
            if stalled >= args.patience:
                break
            start = time.monotonic()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            loader_generator.manual_seed(args.seed + epoch)
            np.random.seed((args.seed + epoch) % 2**32)  # workers=0 crop stream
            model.train()
            totals, seen = {}, 0
            for batch in loader:
                image = batch['image'].to(device, non_blocking=True)
                target = batch['label'].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    losses = model.training_losses(image, target, args.residual_weight,
                                                   args.gate_weight, args.preservation_weight)
                if not torch.isfinite(losses['Loss']):
                    raise FloatingPointError('Non-finite training loss')
                scaler.scale(losses['Loss']).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                if norm <= 0:
                    raise FloatingPointError('Zero training gradient')
                scaler.step(optimizer)
                scaler.update()
                size = image.shape[0]
                seen += size
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach())*size
            losses_mean = {key: value/seen for key, value in totals.items()}
            for key, value in losses_mean.items():
                writer.add_scalar('Train/'+key, value, epoch)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            validation = None
            is_best = False
            if epoch % args.val_every == 0 or epoch == args.epochs:
                validation, rows = evaluate(model, val_ds, device, amp_dtype, args.roi_size, args.overlap, args.seed)
                if not args.diagnostic and any(abs(validation['baseline'][r]-verification['baseline'][r]) > 1e-12
                                               for r in (*REGIONS, 'Mean')):
                    raise ValueError('Validation baseline differs from the verified export')
                for name in (*REGIONS, 'Mean'):
                    writer.add_scalar('Validation/'+name+'_Dice', validation['refined'][name], epoch)
                    writer.add_scalar('Baseline/'+name+'_Dice', validation['baseline'][name], epoch)
                    writer.add_scalar('Delta/'+name+'_Dice',
                                      validation['refined'][name]-validation['baseline'][name], epoch)
                mean = validation['refined']['Mean']
                # Always retain the true maximum, independently of early-stop min_delta.
                is_best = best_mean is None or mean > best_mean
                if is_best:
                    best_mean, best_epoch, best_metrics = mean, epoch, validation
                    save_csv(paths['evaluation']/'best_per_case.csv', rows)
                    save_json(paths['evaluation']/'best_metrics.json', validation)
                if stop_reference is None or mean > stop_reference + args.min_delta:
                    stop_reference, stalled = mean, 0
                else:
                    stalled += 1
                scheduler.step(mean)
                writer.add_scalar('EarlyStopping/Counter', stalled, epoch)
            completed = epoch
            elapsed = time.monotonic()-start
            duration += elapsed
            writer.add_scalar('Performance/Epoch_Time_Seconds', elapsed, epoch)
            if device.type == 'cuda':
                writer.add_scalar('Performance/Peak_GPU_Memory_GB', torch.cuda.max_memory_allocated(device)/2**30, epoch)
                writer.add_scalar('Performance/Peak_GPU_Reserved_GB', torch.cuda.max_memory_reserved(device)/2**30, epoch)
            history.append(dict(epoch=epoch, training=losses_mean, validation=validation))
            if is_best:
                save_checkpoint(best_path)
            save_checkpoint(latest)
            writer.flush()
            print(f'Fold {args.fold} epoch {epoch}: loss={losses_mean["Loss"]:.6f}, '
                  f'best={best_mean}, seconds={elapsed:.1f}', flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted. latest.pt contains the last completed epoch; --resume replays the interrupted epoch.')
    finally:
        writer.flush()
        writer.close()
    if interrupted:
        return
    if best_metrics is None:
        raise RuntimeError('No completed sampled validation')
    if args.diagnostic:
        save_json(paths['fold']/'diagnostic.json',
                  dict(passed=True, cuda_validated=device.type == 'cuda', config=config,
                       identity=identity, sources=sources, versions=versions, completed_epochs=completed,
                       sampled_cases=len(val_ds), best_metrics=best_metrics,
                       note='Engineering diagnostic only; no research result or speed guarantee'))
    else:
        row = paired_row('', best_metrics['baseline'], best_metrics['refined'])
        row.pop('subject_id')
        row = dict(fold=args.fold, best_epoch=best_epoch, **row,
                   best_checkpoint=str(best_path), training_duration_seconds=duration)
        print(json.dumps(update_summary(paths['root'], row), indent=2))


if __name__ == '__main__':
    main()
