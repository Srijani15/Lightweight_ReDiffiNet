"""Standalone fold-0 figures. Reads research artifacts; writes only a fresh output directory.

No training, threshold tuning, or changes to checkpoint compatibility. Missing saved
per-case evaluation triggers one full validation pass using testing_utils.evaluate.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile

import numpy as np
from brats_data_utils_multi_label_5folds import REGIONS, FoldExport, ExportDataset, sha256
from testing_utils import (DiceAccumulator, postprocess, paired_row, evaluate,
                           require_verification, validate_checkpoint, runtime,
                           configure_determinism, save_csv, save_json, METRIC_VERSION)

METRICS = (*REGIONS, 'Mean')


def select_cases(rows):
    """Rank by per-case regional mean, with subject-ID tie breaking; never pooled mean."""
    ordered = sorted(rows, key=lambda r: r['subject_id'])
    improved = [r for r in ordered if r['delta_Mean'] > 0]
    worsened = [r for r in ordered if r['delta_Mean'] < 0]
    selected = []
    if improved:
        selected.append(('best_improved', max(improved, key=lambda r: r['delta_Mean'])))
    # Use a distinct case where possible; exact equality is not invented.
    used = {r['subject_id'] for _, r in selected}
    candidates = [r for r in ordered if r['subject_id'] not in used] or ordered
    near = min(candidates, key=lambda r: abs(r['delta_Mean']))
    selected.append(('unchanged' if near['delta_Mean'] == 0 else 'near_unchanged', near))
    if worsened:
        worst = min(worsened, key=lambda r: r['delta_Mean'])
        selected.append(('most_worsened', worst))
    return selected, {'improved_cases': len(improved), 'worsened_cases': len(worsened),
                      'exactly_unchanged_cases': sum(r['delta_Mean'] == 0 for r in rows)}


def disjoint(regions):
    out = np.zeros(regions.shape[1:], dtype=np.uint8)
    out[regions[0].astype(bool)] = 2
    out[regions[1].astype(bool)] = 1
    out[regions[2].astype(bool)] = 3
    return out


def corrections(baseline, refined, truth):
    """0 unchanged, 1 repaired, 2 worsened, 3 wrong-to-different-wrong."""
    changed = baseline != refined
    out = np.zeros(baseline.shape, dtype=np.uint8)
    out[changed] = 3
    out[changed & (refined == truth)] = 1
    out[changed & (baseline == truth)] = 2
    return out


def representative_slice(truth, baseline, refined):
    area = (truth > 0).sum(axis=(0, 1))
    if area.max() > 0:
        return int(area.argmax()), 'largest GT lesion area; lowest index breaks ties'
    area = ((baseline > 0) | (refined > 0)).sum(axis=(0, 1))
    if area.max() > 0:
        return int(area.argmax()), 'no GT lesion: largest predicted lesion union'
    return truth.shape[2] // 2, 'no lesion: central exported slice'


def check_rows(rows, ids):
    if len(rows) != len(ids) or {r['subject_id'] for r in rows} != set(ids):
        raise ValueError('Per-case CSV must cover exactly the validation subjects once')
    for row in rows:
        for prefix in ('baseline', 'refined'):
            for region in METRICS:
                value = float(row[f'{prefix}_{region}'])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid per-case Dice')
                row[f'{prefix}_{region}'] = value
            if abs(row[f'{prefix}_Mean'] - np.mean([row[f'{prefix}_{r}'] for r in REGIONS])) > 1e-12:
                raise ValueError('Per-case regional mean mismatch')
        for region in METRICS:
            row[f'delta_{region}'] = row[f'refined_{region}'] - row[f'baseline_{region}']
    return rows


def sample_volume(model, item, device, dtype, config):
    """Minimal volume-returning equivalent of evaluate (which returns metrics only).

    Same subject seed, fork_rng, model.sample, autocast, Gaussian stitching and
    clamp. No extra tensors or random draws are introduced.
    """
    import torch
    from monai.inferers import sliding_window_inference
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    seed = (config['seed'] + int(hashlib.sha256(item['subject_id'].encode()).hexdigest()[:8], 16)) % (2**31)
    model.eval()
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        def predict(window):
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                return model.sample(window).float()
        output = sliding_window_inference(
            torch.from_numpy(item['image']).unsqueeze(0), config['roi_size'], 1,
            predict, overlap=config['overlap'], mode='gaussian',
            sw_device=device, device=torch.device('cpu'))
        return output.clamp(0, 1)[0].numpy()


def plot_summary(plt, metrics, path, dpi):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(4)
    for offset, name, label, color in ((-.19, 'baseline', 'Frozen U-Net', '#4477AA'),
                                      (.19, 'refined', 'ReDiffiNet', '#EE7733')):
        bars = ax.bar(x + offset, [metrics[name][r] for r in METRICS], .36,
                      label=label, color=color)
        ax.bar_label(bars, fmt='%.4f', fontsize=10, padding=4)
    ax.set(xticks=x, xticklabels=METRICS, ylim=(0, 1.08), ylabel='Dice (pooled validation counts)',
           title='Fold 0 validation (251 cases) | best checkpoint: epoch 12')
    ax.legend(loc='lower right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.2)
    delta = '   |   '.join(f'{r}: {metrics["refined"][r]-metrics["baseline"][r]:+.6f}' for r in METRICS)
    fig.text(.5, .035, 'Delta Dice (ReDiffiNet - U-Net)\n' + delta,
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def plot_case(plt, item, refined, reason, row, path, dpi):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    b = disjoint(postprocess(item['image'][4:]))
    p = disjoint(postprocess(refined))
    y = disjoint(item['label'])
    z, rule = representative_slice(y, b, p)
    cm = corrections(b, p, y)
    flair = item['image'][2, :, :, z].T
    foreground = flair[flair != 0]
    lo, hi = np.percentile(foreground, [1, 99]) if foreground.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    seg_colors = ['#000000', '#CC79A7', '#56B4E9', '#E69F00']
    corr_colors = ['#000000', '#009E73', '#D55E00', '#CC79A7']
    norm = BoundaryNorm(np.arange(-.5, 4.5), 4)
    fig, axes = plt.subplots(1, 5, figsize=(19, 5.6))
    for ax, title in zip(axes, ('T2-FLAIR', 'Ground truth', 'Frozen U-Net', 'ReDiffiNet', 'Correction map')):
        ax.imshow(flair, cmap='gray', vmin=lo, vmax=hi, origin='lower', interpolation='nearest')
        ax.set_title(title, fontsize=13)
        ax.axis('off')
    for ax, labels in zip(axes[1:4], (y, b, p)):
        overlay = labels[:, :, z].T
        ax.imshow(np.ma.masked_equal(overlay, 0), cmap=ListedColormap(seg_colors), norm=norm,
                  alpha=.65, origin='lower', interpolation='nearest')
    overlay = cm[:, :, z].T
    axes[4].imshow(np.ma.masked_equal(overlay, 0), cmap=ListedColormap(corr_colors), norm=norm,
                   alpha=.85, origin='lower', interpolation='nearest')
    title = (f'{item["subject_id"]} | exported axial slice {z} (0-based)\n'
             f'Selected example: {reason.replace("_", " ")} | case Mean Dice '
             f'{row["baseline_Mean"]:.6f} -> {row["refined_Mean"]:.6f} '
             f'(delta {row["delta_Mean"]:+.6f})')
    fig.suptitle(title, fontsize=14, y=.98)
    handles = [Patch(color=seg_colors[i], label=label) for i, label in (
        (1, 'Non-enhancing/necrotic core'), (2, 'Edema'), (3, 'Enhancing tumor'))]
    handles += [Patch(color=corr_colors[i], label=label) for i, label in (
        (1, 'Corrected to GT'), (2, 'Worsened from GT'), (3, 'Wrong to different wrong'))]
    handles.append(Patch(color='#888888', label='Unchanged: grayscale (correct or wrong)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(.5, .04))
    fig.text(.5, .015, f'Slice rule: {rule}. Display uses exported X/Y/Z axes; anatomical orientation metadata unavailable.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.01, right=.99, top=.78, bottom=.23, wspace=.035)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return {'slice_index': z, 'slice_rule': rule,
            'slice_correction_counts': {name: int((cm[:, :, z] == i).sum()) for i, name in
                enumerate(('unchanged', 'corrected', 'worsened', 'wrong_to_different_wrong'))}}


def main():
    root = Path.home() / 'Major_Project'
    fold = root / 'ReDiffiNet_Output_20Steps_ancflJ/fold_0'
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ('data-dir', root / 'ReDiffiNet_Data_Fold0_Clean'),
        ('assignments', root / 'UNet_Output_Light_BF16/fold_0/fold_assignments.csv'),
        ('verification', root / 'ReDiffiNet_Output/audit/fold_0/verification.json'),
        ('checkpoint', fold / 'checkpoints/best_mean_dice.pt'),
        ('evaluation-dir', fold / 'evaluation/best_checkpoint'),
        ('output-dir', fold / 'visualizations')):
        parser.add_argument('--' + name, type=lambda s: Path(s).expanduser().resolve(), default=default)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error('--dpi must be at least 150')
    out = args.output_dir.resolve()
    protected = [args.data_dir, args.assignments.parent.parent, args.checkpoint.parent,
                 args.checkpoint.parent.parent / 'evaluation', args.evaluation_dir,
                 args.checkpoint.parent.parent / 'tensorboard', args.verification.parent,
                 Path(__file__).resolve().parent]
    if any(out == p.resolve() or p.resolve() in out.parents for p in protected):
        raise ValueError('Output overlaps protected research inputs/source directory')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output must be absent or empty; existing artifacts will not be overwritten')

    import torch
    from train import source_identity
    from unet.refinement import DiffUNet
    export = FoldExport(args.data_dir, args.assignments, 0)
    if len(export.ids['training']) != 1000 or len(export.ids['validation']) != 251:
        raise ValueError('Expected the confirmed 1,000 training / 251 validation export')
    verification = require_verification(args.verification, export)
    checkpoint_hash = sha256(args.checkpoint)
    c = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(c, 0, export.identity())
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'tensorboard')}
    if c['sources'] != source_identity() or c['versions'] != versions:
        raise ValueError('Code/dependencies differ from checkpoint; visualization rejected')
    if c['diagnostic'] or c['epoch'] != c['best_epoch'] or c['epoch'] != 12:
        raise ValueError('Expected the confirmed epoch-12 research best checkpoint')
    metrics = c['best_metrics']
    expected = {'baseline': [.934224401740881, .9204570445150903, .8973947253367587, .9173587238642433],
                'refined': [.9362251245101753, .9202756939487812, .8973496842473618, .9179501675687728]}
    for kind in expected:
        if any(abs(metrics[kind][r] - v) > 1e-12 for r, v in zip(METRICS, expected[kind])):
            raise ValueError('Checkpoint does not match the confirmed result')
    if any(abs(metrics['baseline'][r] - verification['baseline'][r]) > 1e-12 for r in METRICS):
        raise ValueError('Checkpoint baseline differs from accepted verification')

    # Keep Matplotlib's cache outside the project; remove it on process completion.
    import os
    with tempfile.TemporaryDirectory(prefix='rediffinet-figures-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        configure_determinism(c['config']['seed'], c['config']['determinism'])
        device, dtype = runtime(args.device, c['config']['amp'])
        model = DiffUNet(**c['config']['model']).to(device)
        model.load_state_dict(c['model_state_dict'])
        ds = ExportDataset(export, 'validation')
        csv_path = args.evaluation_dir / 'paired_per_case.csv'
        json_path = args.evaluation_dir / 'metrics.json'
        if csv_path.exists() and json_path.exists():
            saved = json.loads(json_path.read_text(encoding='utf-8'))
            if (saved.get('epoch') != c['epoch'] or saved.get('fold') != 0
                    or saved.get('metric_version') != METRIC_VERSION or saved.get('cases') != len(ds)
                    or Path(saved.get('checkpoint', '')).expanduser().resolve() != args.checkpoint.resolve()):
                raise ValueError('Saved evaluation provenance mismatch')
            for kind in ('baseline', 'refined'):
                if any(abs(saved[kind][r] - metrics[kind][r]) > 1e-12 for r in METRICS):
                    raise ValueError('Saved evaluation scores differ from checkpoint')
            with csv_path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            ranking_source = str(csv_path)
        else:
            print('Saved evaluation pair missing: evaluating all validation cases read-only.', flush=True)
            measured, rows = evaluate(model, ds, device, dtype, c['config']['roi_size'],
                                      c['config']['overlap'], c['config']['seed'])
            for kind in ('baseline', 'refined'):
                if any(abs(measured[kind][r] - metrics[kind][r]) > 1e-8 for r in METRICS):
                    raise ValueError('Fresh full-fold evaluation differs from saved result')
            ranking_source = 'fresh testing_utils.evaluate; fixed checkpoint settings'
        rows = check_rows(rows, export.ids['validation'])
        selected, counts = select_cases(rows)
        print(json.dumps(counts), flush=True)
        if not counts['worsened_cases']:
            print('No worsened validation case exists in these per-case metrics.')
        if not counts['exactly_unchanged_cases']:
            print('No exactly unchanged case; closest available delta is labeled near-unchanged.')
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'summary.json', {k: {r: metrics[k][r] for r in METRICS}
                  for k in ('baseline', 'refined', 'delta')})
        save_csv(out / 'case_ranking.csv', sorted(rows, key=lambda r: (-r['delta_Mean'], r['subject_id'])))
        plot_summary(plt, metrics, out / 'dice_comparison.png', args.dpi)
        selections = []
        for reason, row in selected:
            sid = row['subject_id']
            print(f'Read-only sampling: {sid} ({reason})', flush=True)
            item = ds[ds.ids.index(sid)]
            refined = sample_volume(model, item, device, dtype, c['config'])
            before, after = DiceAccumulator(), DiceAccumulator()
            before.add(postprocess(item['image'][4:]), item['label'])
            after.add(postprocess(refined), item['label'])
            actual = paired_row(sid, before.scores(), after.scores())
            if any(abs(actual[f'{k}_{r}'] - row[f'{k}_{r}']) > 1e-8
                   for k in ('baseline', 'refined') for r in METRICS):
                raise ValueError(f'{sid}: sampled Dice differs from ranking; refusing misleading figure')
            filename = f'{reason}_{sid}.png'
            details = plot_case(plt, item, refined, reason, actual, out / filename, args.dpi)
            selections.append(dict(subject_id=sid, selection_reason=reason, filename=filename,
                                   **details, **{k: v for k, v in actual.items() if k != 'subject_id'}))
        if sha256(args.checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during visualization')
        save_json(out / 'selection_manifest.json', dict(
            completed=True, checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
            script_sha256=sha256(__file__), sources=c['sources'], versions=versions,
            identity=export.identity(), config=c['config'], ranking_source=ranking_source,
            ranking_csv_sha256=sha256(csv_path) if csv_path.exists() else None,
            counts=counts, selections=selections,
            notes=['Selected examples, not representative estimates of performance.',
                   'Per-case regional Mean ranks examples; summary uses pooled fold Dice.',
                   'Unchanged selection means unchanged per-case Mean Dice, not necessarily identical masks.',
                   'Axes are exported X/Y/Z; spacing and anatomical affine are not in the dataset.',
                   'Unchanged correction-map voxels may be correct or incorrect.',
                   'Validation used for checkpoint selection; not an independent test set.']))
        print(f'Completed figures and tables: {out}', flush=True)


if __name__ == '__main__':
    main()
