"""Optional read-only checkpoint inspection; no training or checkpoint writes.

Keep instrumentation outside the hashed training/evaluation sources so existing
diagnostic reports and checkpoints retain their exact compatibility checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from testing_utils import postprocess, DiceAccumulator
from brats_data_utils_multi_label_5folds import REGIONS


def change_counts(baseline, refined, truth):
    changed = baseline != refined
    count = int(changed.sum())
    corrected = int((changed & (refined == truth)).sum())
    worsened = int((changed & (baseline == truth)).sum())
    return dict(total=int(changed.size), changed=count, changed_fraction=count/changed.size,
                corrected=corrected, worsened=worsened,
                corrected_percent_of_changes=100*corrected/count if count else None,
                worsened_percent_of_changes=100*worsened/count if count else None)


def refinement_statistics(baseline, final, residual, gate, target):
    """Statistics over unique exported voxels AFTER Gaussian window stitching."""
    arrays = [np.asarray(v) for v in (baseline, final, residual, gate, target)]
    if any(v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays):
        raise ValueError('Mismatched or nonfinite diagnostic tensors')
    baseline, final, residual, gate, target = arrays
    b, p, y = postprocess(baseline), postprocess(final), target.astype(bool)
    changed = (b != p).any(axis=0)
    spatial_count = int(changed.sum())
    corrected = int((changed & (p == y).all(axis=0)).sum())
    worsened = int((changed & (b == y).all(axis=0)).sum())
    return dict(gate_mean=float(gate.mean(dtype=np.float64)), gate_max=float(gate.max()),
                abs_residual_mean=float(np.abs(residual).mean(dtype=np.float64)),
                abs_residual_max=float(np.abs(residual).max()),
                mean_abs_prediction_change=float(np.abs(final-baseline).mean(dtype=np.float64)),
                region_decisions=change_counts(b, p, y),
                per_region={r: change_counts(b[i], p[i], y[i]) for i, r in enumerate(REGIONS)},
                spatial_voxels=dict(total=int(changed.size), changed=spatial_count,
                                    changed_fraction=spatial_count/changed.size,
                                    corrected_to_exact_target=corrected,
                                    worsened_from_exact_target=worsened,
                                    wrong_to_different_wrong=spatial_count-corrected-worsened,
                                    corrected_percent_of_changes=100*corrected/spatial_count if spatial_count else None,
                                    worsened_percent_of_changes=100*worsened/spatial_count if spatial_count else None))


def inspect_case(model, item, device, amp_dtype, roi, overlap, seed):
    import torch
    from monai.inferers import sliding_window_inference
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    case_seed = (seed + int(hashlib.sha256(item['subject_id'].encode()).hexdigest()[:8], 16)) % (2**31)
    # Same sampler, seed, windows and output blending as the existing evaluator.
    # Returning additional tensors does not rerun sampling or consume extra RNG.
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(case_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(case_seed)
        def predict(window):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                final, residual, gate = model.sample(window, return_parts=True)
                return final.float(), residual.float(), gate.float()
        outputs = sliding_window_inference(torch.from_numpy(item['image']).unsqueeze(0), roi, 1,
                                           predict, overlap=overlap, mode='gaussian',
                                           sw_device=device, device=torch.device('cpu'))
        final, residual, gate = [v[0].numpy() for v in outputs]
        final = np.clip(final, 0, 1)
        gate = np.clip(gate, 0, 1)
        residual = np.clip(residual, -1, 1)
    baseline = item['image'][4:]
    before, after = DiceAccumulator(), DiceAccumulator()
    before.add(postprocess(baseline), item['label'])
    after.add(postprocess(final), item['label'])
    return dict(subject_id=item['subject_id'], baseline=before.scores(), refined=after.scores(),
                baseline_counts=before.counts.tolist(), refined_counts=after.counts.tolist(),
                **refinement_statistics(baseline, final, residual, gate, item['label']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir', 'assignments', 'verification', 'checkpoint'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-cases', type=int, default=2, help='First sorted validation cases; 0 means all')
    args = parser.parse_args()
    if args.max_cases < 0:
        parser.error('--max-cases must be nonnegative')
    import importlib.metadata
    import torch
    from train import source_identity
    from unet.refinement import DiffUNet
    from brats_data_utils_multi_label_5folds import FoldExport, ExportDataset, sha256
    from testing_utils import require_verification, validate_checkpoint, configure_determinism, runtime
    export = FoldExport(args.data_dir, args.assignments, args.fold)
    verification = require_verification(args.verification, export)
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, args.fold, export.identity())
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'tensorboard')}
    if checkpoint['sources'] != source_identity() or checkpoint['versions'] != versions:
        raise ValueError('Code/dependencies differ from checkpoint; inspection rejected')
    config = checkpoint['config']
    configure_determinism(config['seed'], config['determinism'])
    device, dtype = runtime(args.device, config['amp'])
    model = DiffUNet(**config['model']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    rows = [inspect_case(model, item, device, dtype, config['roi_size'], config['overlap'], config['seed'])
            for item in ExportDataset(export, 'validation', limit=args.max_cases)]
    before, after = DiceAccumulator(), DiceAccumulator()
    for row in rows:
        before.counts += np.asarray(row['baseline_counts'])
        after.counts += np.asarray(row['refined_counts'])
    if len(rows) == len(export.ids['validation']):
        if any(abs(before.scores()[r]-verification['baseline'][r]) > 1e-12 for r in (*REGIONS, 'Mean')):
            raise ValueError('Inspected baseline differs from verified full export')
    print(json.dumps(dict(checkpoint=str(args.checkpoint), epoch=checkpoint['epoch'],
                          inspector_sha256=sha256(__file__), fold=args.fold, cases=rows,
                          pooled_baseline=before.scores(), pooled_refined=after.scores(),
                          note='Read-only inference. Unique cropped voxels; gate/residual are separately Gaussian-stitched. '
                               'Region decisions count WT/TC/ET separately; spatial voxels count any region change. '
                               'Null percentages mean no changes. Subset metrics are diagnostic only.'), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
