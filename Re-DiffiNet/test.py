"""Read-only export verification, paired checkpoint evaluation, and CV reporting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify-export', help='Read exported cases, never run U-Net or alter the export')
    for tool in (v, sub.add_parser('evaluate', help='Sample the saved best refinement model')):
        tool.add_argument('--data-dir', required=True)
        tool.add_argument('--assignments', required=True)
        tool.add_argument('--fold', type=int, choices=range(5), default=0)
        tool.add_argument('--out-dir', required=True, help='Separate audit/evaluation output directory')
    v.add_argument('--unet-checkpoint', required=True, help='Trusted generating best_mean_dice.pt')
    v.add_argument('--accept-difference', default='', metavar='REASON',
                   help='Record a reviewed numerical discrepancy; never bypass structural checks')
    e = sub.choices['evaluate']
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--verification', required=True)
    e.add_argument('--device', default='cuda:0')
    s = sub.add_parser('summarize')
    s.add_argument('--output-dir', required=True)
    sub.add_parser('environment', help='Print dependencies/devices; no training or downloads')
    return p


def safe_output(destination, export):
    out = Path(destination).expanduser().resolve()
    protected = [export.root, export.assignments_path.parent.parent,
                 Path(__file__).resolve().parent.parent/'brats2023_updated']
    if any(out == root or root in out.parents for root in protected):
        raise ValueError('Output must be outside the export and protected U-Net directories')
    return out


def check_reference(checkpoint, export):
    """Verify supplied membership against saved U-Net split algorithm; create no folds."""
    import numpy as np
    if checkpoint.get('fold_number') != export.fold or checkpoint.get('n_folds') != 5:
        raise ValueError('Generating U-Net checkpoint has a different fold/n_folds')
    args = checkpoint.get('args', {})
    if 'seed' not in args or args.get('fold_source') not in ('all', 'train'):
        raise ValueError('Missing authoritative U-Net seed/fold_source metadata')
    for key in ('max_train_samples', 'max_val_samples'):
        if args.get(key):
            raise ValueError('Diagnostic U-Net checkpoint is not a research baseline')
    source_order = ('train', 'validation') if args['fold_source'] == 'all' else ('train',)
    ordered = [(source, sid) for source in source_order
               for sid in sorted(sid for sid, (src, _) in export.members.items() if src == source)]
    if len(ordered) != len(export.members):
        raise ValueError('Assignments contain unexpected source directories')
    shuffled = np.arange(len(ordered))
    np.random.default_rng(int(args['seed'])).shuffle(shuffled)
    validation_indices = set(np.array_split(shuffled, 5)[export.fold].tolist())
    for index, (source, sid) in enumerate(ordered):
        split = 'validation' if index in validation_indices else 'training'
        if export.members[sid] != (source, split):
            raise ValueError('Assignment CSV conflicts with checkpoint seed/fold membership')
    scores = np.asarray(checkpoint.get('best_dice_scores'), dtype=np.float64)
    if scores.shape != (3,) or not np.isfinite(scores).all():
        raise ValueError('Checkpoint missing WT/TC/ET best_dice_scores')
    if abs(float(scores.mean()) - float(checkpoint['best_mean_dice'])) > 1e-12:
        raise ValueError('Checkpoint has inconsistent best Mean Dice')
    if checkpoint.get('epoch') != checkpoint.get('best_epoch'):
        raise ValueError('Use the best_mean_dice checkpoint, not a later latest checkpoint')
    shape = tuple(args.get('crop_size') or (128, 192, 128))
    return scores, shape


def verify(args):
    import numpy as np
    import torch
    from brats_data_utils_multi_label_5folds import FoldExport, read_case, sha256, REGIONS, CHANNELS
    from testing_utils import DiceAccumulator, METRIC_VERSION, HISTORICAL_FOLD0, save_json, save_csv
    export = FoldExport(args.data_dir, args.assignments, args.fold)
    out = safe_output(args.out_dir, export)
    checkpoint_path = Path(args.unet_checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    reference, expected_shape = check_reference(checkpoint, export)
    identity = export.identity()
    total, rows, hashes = DiceAccumulator(), [], {}
    # Validate every training case's shape/channel format as well; accumulate
    # metrics ONLY for validation. One case resident at a time, no GPU work.
    for split in ('training', 'validation'):
        for index, sid in enumerate(export.ids[split], 1):
            image, target = read_case(export.paths[sid])
            if image.shape[1:] != expected_shape:
                raise ValueError(f'{sid}: shape {image.shape[1:]} != U-Net crop {expected_shape}')
            if split == 'validation':
                scores = total.add(image[4:], target)
                rows.append(dict(subject_id=sid, **scores))
                hashes[sid] = sha256(export.paths[sid])
            if index % 50 == 0:
                print(f'Checked {split}: {index}/{len(export.ids[split])}', flush=True)
    if export.identity() != identity:
        raise ValueError('Export changed during verification')
    scores = total.scores()
    delta = {r: scores[r]-float(reference[i]) for i, r in enumerate(REGIONS)}
    delta['Mean'] = scores['Mean'] - float(reference.mean())
    matched = all(abs(v) <= 1e-8 for v in delta.values())
    report = dict(metric_version=METRIC_VERSION, identity=identity, baseline=scores,
                  pooled_counts=total.counts.tolist(),
                  count_rows=['intersection', 'prediction_volume', 'target_volume'], count_columns=REGIONS,
                  reference=dict(zip((*REGIONS, 'Mean'), [*reference.tolist(), float(reference.mean())])),
                  difference=delta, matched_within_1e_8=matched,
                  accepted_difference_reason=args.accept_difference.strip(),
                  approved_for_diagnostic=matched or bool(args.accept_difference.strip()),
                  evaluated_cases=len(rows), checked_training_cases=len(export.ids['training']),
                  validation_sha256=hashes, channels=CHANNELS, crop_shape=expected_shape,
                  unet_checkpoint=str(checkpoint_path), unet_checkpoint_sha256=sha256(checkpoint_path),
                  unet_best_epoch=checkpoint['best_epoch'],
                  provenance_basis='User-confirmed regenerated export; supplied provenance file hashed, not independently attested',
                  training_predictions='in-sample, NOT out-of-fold',
                  validation_predictions='same fold checkpoint; validation patients excluded from U-Net optimization',
                  metric_aggregation='pooled counts across validation patients; regional arithmetic mean')
    if args.fold == 0:
        report['difference_from_user_reported_epoch20'] = {
            **{r: scores[r]-float(HISTORICAL_FOLD0[i]) for i, r in enumerate(REGIONS)},
            'Mean': scores['Mean']-0.9174095750860682}
    save_csv(out/'baseline_per_case.csv', rows)
    save_json(out/'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'validation_sha256'}, indent=2))
    print('Hard-label export preserves thresholded masks. A difference is not inherently caused by storing hard labels.')
    print('Check BF16 vs FP32 inference near thresholds, checkpoint identity, crop, and complete patient membership.')
    if not report['approved_for_diagnostic']:
        raise SystemExit('Numerical difference needs review. Keep the export; do not regenerate automatically.')


def evaluate_checkpoint(args):
    import importlib.metadata
    import torch
    from brats_data_utils_multi_label_5folds import FoldExport, ExportDataset
    from testing_utils import (require_verification, validate_checkpoint, runtime, evaluate,
                               save_json, save_csv, configure_determinism)
    from train import source_identity
    from unet.refinement import DiffUNet
    export = FoldExport(args.data_dir, args.assignments, args.fold)
    verification = require_verification(args.verification, export)
    out = safe_output(args.out_dir, export)
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, args.fold, export.identity())
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'monai', 'numpy', 'tensorboard')}
    if checkpoint['sources'] != source_identity() or checkpoint['versions'] != versions:
        raise ValueError('Code/dependencies differ from checkpoint; matched evaluation rejected')
    config = checkpoint['config']
    configure_determinism(config['seed'], config['determinism'])
    device, amp_dtype = runtime(args.device, config['amp'])
    model = DiffUNet(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    result, rows = evaluate(model, ExportDataset(export, 'validation'), device, amp_dtype,
                            config['roi_size'], config['overlap'], config['seed'])
    if any(abs(result['baseline'][r]-verification['baseline'][r]) > 1e-12 for r in ('WT', 'TC', 'ET', 'Mean')):
        raise ValueError('Evaluated baseline differs from the verified export')
    result.update(fold=args.fold, epoch=checkpoint['epoch'], checkpoint=str(args.checkpoint),
                  diagnostic=checkpoint['diagnostic'], training_predictions='in-sample')
    save_csv(out/'paired_per_case.csv', rows)
    save_json(out/'metrics.json', result)
    print(json.dumps(result, indent=2))


def main():
    args = parser().parse_args()
    if args.command == 'verify-export':
        verify(args)
    elif args.command == 'evaluate':
        evaluate_checkpoint(args)
    elif args.command == 'summarize':
        from testing_utils import summarize
        with (Path(args.output_dir).expanduser()/'cross_validation_summary.csv').open(newline='') as handle:
            print(json.dumps(summarize(list(csv.DictReader(handle))), indent=2))
    else:
        import importlib.metadata
        import torch
        for package in ('torch', 'monai', 'numpy', 'tensorboard'):
            print(package, importlib.metadata.version(package))
        print('CUDA build:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(i, props.name, 'VRAM GiB:', round(props.total_memory/2**30, 2))
        print('No training, inference or downloads performed.')


if __name__ == '__main__':
    main()
