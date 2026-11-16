"""Read-only Fold-0 result figures from the accepted evaluation and best checkpoint.

Writes a new, empty output directory only.  It never trains, changes metrics,
or writes into an evaluation, data, checkpoint, or source directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np

from brats_data_utils_multi_label_5folds import REGIONS, FoldExport, ExportDataset, sha256
from testing_utils import (DiceAccumulator, METRIC_VERSION, paired_row, postprocess,
                           require_verification, runtime, configure_determinism,
                           validate_checkpoint, save_json)

METRICS = (*REGIONS, 'Mean')
MODALITIES = (
    ('T1ce', 0, 't1c'),
    ('T1', 1, 't1n'),
    ('FLAIR', 2, 't2f'),
    ('T2', 3, 't2w'),
)
REGION_COLORS = {'WT': '#F4A261', 'TC': '#C77DFF', 'ET': '#00B4D8'}


def checked_rows(path, validation_ids):
    """Parse the saved paired evaluation table without averaging per-case Dice."""
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(validation_ids) or {r.get('subject_id') for r in rows} != set(validation_ids):
        raise ValueError('paired_per_case.csv must contain each validation subject exactly once')
    parsed = []
    for raw in rows:
        row = {'subject_id': raw['subject_id']}
        for prefix in ('baseline', 'refined'):
            for region in METRICS:
                try:
                    value = float(raw[f'{prefix}_{region}'])
                except (KeyError, ValueError) as exc:
                    raise ValueError(f'{row["subject_id"]}: invalid {prefix}_{region}') from exc
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{row["subject_id"]}: Dice must be finite in [0, 1]')
                row[f'{prefix}_{region}'] = value
            expected_mean = np.mean([row[f'{prefix}_{r}'] for r in REGIONS])
            if abs(row[f'{prefix}_Mean'] - expected_mean) > 1e-12:
                raise ValueError(f'{row["subject_id"]}: per-case Mean is inconsistent')
        for region in METRICS:
            row[f'delta_{region}'] = row[f'refined_{region}'] - row[f'baseline_{region}']
        parsed.append(row)
    return parsed


def choose_case(rows, subject_id, selection, overall_delta):
    ordered = sorted(rows, key=lambda r: r['subject_id'])
    if subject_id:
        selected = next((r for r in ordered if r['subject_id'] == subject_id), None)
        if selected is None:
            raise ValueError(f'--subject-id is not a validation subject: {subject_id}')
        return selected, 'explicit subject_id supplied on the command line'
    if selection == 'best-improved':
        return max(ordered, key=lambda r: r['delta_Mean']), \
            'automatic: largest per-case Mean Dice increase; subject_id breaks ties alphabetically'
    # A transparent representative rule: closest individual change to the full-fold
    # pooled change. This is a selection for illustration, not an estimate.
    return min(ordered, key=lambda r: (abs(r['delta_Mean'] - overall_delta), r['subject_id'])), \
        'automatic: per-case Mean Dice change closest to the full-fold pooled Mean Dice change; subject_id breaks ties alphabetically'


def choose_slice(target, baseline, refined):
    """Axial Z slice maximizing GT WT extent; then prediction disagreement; then low Z."""
    gt_wt = target[0].astype(bool).sum(axis=(0, 1))
    disagreement = (postprocess(baseline) != postprocess(refined)).sum(axis=(0, 1, 2))
    candidates = np.flatnonzero(gt_wt == gt_wt.max())
    if gt_wt.max() == 0:
        candidates = np.arange(target.shape[3])
    best_disagreement = disagreement[candidates].max()
    return int(candidates[np.flatnonzero(disagreement[candidates] == best_disagreement)[0]]), \
        'largest ground-truth WT area; ties use largest U-Net/ReDiffiNet region disagreement, then lowest slice index'


def exact_case_metrics(item, refined):
    before, after = DiceAccumulator(), DiceAccumulator()
    before.add(postprocess(item['image'][4:]), item['label'])
    after.add(postprocess(refined), item['label'])
    return paired_row(item['subject_id'], before.scores(), after.scores())


def same_metrics(left, right, tolerance=1e-8):
    return all(abs(left[f'{prefix}_{region}'] - right[f'{prefix}_{region}']) <= tolerance
               for prefix in ('baseline', 'refined') for region in METRICS)


def draw_region(ax, base, region_mask, color, alpha=.58):
    from matplotlib.colors import ListedColormap
    ax.imshow(base, cmap='gray', origin='lower', interpolation='nearest')
    ax.imshow(np.ma.masked_where(~region_mask, np.ones(region_mask.shape, dtype=np.uint8)),
              cmap=ListedColormap([color]), vmin=0, vmax=1,
              alpha=alpha, origin='lower', interpolation='nearest')
    # A contour remains visible on bright MR regions and avoids opaque masks.
    if region_mask.any():
        ax.contour(region_mask.astype(np.uint8), levels=[.5], colors=[color], linewidths=.9,
                   origin='lower')


def draw_overlay(ax, base, masks):
    ax.imshow(base, cmap='gray', origin='lower', interpolation='nearest')
    for region in REGIONS:
        mask = masks[REGIONS.index(region)]
        if mask.any():
            ax.contour(mask.astype(np.uint8), levels=[.5], colors=[REGION_COLORS[region]], linewidths=1.1,
                       origin='lower')


def modality_limits(array):
    foreground = array[np.isfinite(array) & (array != 0)]
    if not foreground.size:
        return 0., 1.
    low, high = np.percentile(foreground, (1, 99))
    return (float(low), float(high if high > low else low + 1.))


def plot_overall(plt, metrics, path, dpi):
    fig = plt.figure(figsize=(13.2, 6.4), facecolor='white')
    grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.42), wspace=.27)
    table_ax = fig.add_subplot(grid[0, 0])
    chart_ax = fig.add_subplot(grid[0, 1])
    table_ax.axis('off')
    cell = [[f'{metrics[name][region]:.6f}' for region in METRICS]
            for name in ('baseline', 'refined', 'delta')]
    table = table_ax.table(cellText=cell, colLabels=METRICS,
                           rowLabels=('Frozen U-Net', 'ReDiffiNet', 'Delta'),
                           cellLoc='center', colLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.2, 2.15)
    for (row, col), box in table.get_celld().items():
        box.set_edgecolor('#CBD5E1')
        if row == 0:
            box.set_facecolor('#E2E8F0')
            box.set_text_props(weight='bold')
        elif row == 2:
            box.set_facecolor('#FFF7ED')
        elif row == 3:
            box.set_facecolor('#ECFDF5')
    table_ax.set_title('Pooled validation Dice', weight='bold', pad=15)

    x = np.arange(len(METRICS))
    width = .35
    for offset, key, label, color in ((-width / 2, 'baseline', 'Frozen U-Net', '#457B9D'),
                                      (width / 2, 'refined', 'ReDiffiNet', '#E76F51')):
        bars = chart_ax.bar(x + offset, [metrics[key][r] for r in METRICS], width,
                            label=label, color=color, edgecolor='white', linewidth=.7)
        chart_ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9, rotation=90)
    ymin = min(v for key in ('baseline', 'refined') for v in metrics[key].values()) - .012
    chart_ax.set(xticks=x, xticklabels=METRICS, ylim=(max(0, ymin), 0.945),
                 ylabel='Dice', title='Model comparison')
    chart_ax.grid(axis='y', color='#CBD5E1', alpha=.55)
    chart_ax.set_axisbelow(True)
    chart_ax.spines[['top', 'right']].set_visible(False)
    chart_ax.legend(frameon=False, loc='lower right')
    fig.suptitle('Fold-0 validation performance (251 cases) | best ReDiffiNet checkpoint, epoch 12',
                 fontsize=15, weight='bold', y=.98)
    fig.text(.5, .015, 'Metric: pooled counts across validation cases; Mean = arithmetic mean of WT, TC, ET. '
             'Validation was used for checkpoint selection.', ha='center', fontsize=9.2)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def plot_case(plt, item, refined, row, reason, path, dpi):
    from matplotlib.patches import Patch
    baseline = item['image'][4:]
    truth = item['label'].astype(bool)
    base_regions = postprocess(baseline)
    refined_regions = postprocess(refined)
    z, slice_rule = choose_slice(truth, baseline, refined)
    fig, axes = plt.subplots(7, 4, figsize=(14.6, 21.0), facecolor='white')
    column_titles = ('WT', 'TC', 'ET', 'Overlay')
    row_titles = [name for name, _, _ in MODALITIES] + ['Frozen U-Net prediction', 'ReDiffiNet prediction', 'Ground truth']
    all_rows = []
    for _, channel, _ in MODALITIES:
        all_rows.append(('modality', item['image'][channel, :, :, z].T, truth[:, :, :, z].transpose(0, 2, 1)))
    flair = item['image'][2, :, :, z].T
    all_rows += [('segmentation', flair, base_regions[:, :, :, z].transpose(0, 2, 1)),
                 ('segmentation', flair, refined_regions[:, :, :, z].transpose(0, 2, 1)),
                 ('segmentation', flair, truth[:, :, :, z].transpose(0, 2, 1))]
    for i, (kind, image, masks) in enumerate(all_rows):
        low, high = modality_limits(image)
        for j, ax in enumerate(axes[i]):
            if j < 3:
                draw_region(ax, image, masks[j], REGION_COLORS[REGIONS[j]])
            else:
                draw_overlay(ax, image, masks)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(column_titles[j], fontsize=13, weight='bold', pad=8)
            if j == 0:
                ax.set_ylabel(row_titles[i], fontsize=11, weight='bold', rotation=0,
                              labelpad=68, va='center')
            # Reset grayscale scaling after initial image call, without reordering overlays.
            ax.images[0].set_clim(low, high)
    fig.text(.5, .958, f'Fold-0 validation case: {item["subject_id"]} | axial exported slice {z} (0-based)',
             ha='center', fontsize=15, weight='bold')
    fig.text(.5, .941, f'Selection: {reason}', ha='center', fontsize=9.5)
    metric_text = '     '.join(
        f'{label}: WT {row[f"{prefix}_WT"]:.4f} | TC {row[f"{prefix}_TC"]:.4f} | ET {row[f"{prefix}_ET"]:.4f} | Mean {row[f"{prefix}_Mean"]:.4f}'
        for label, prefix in (('U-Net', 'baseline'), ('ReDiffiNet', 'refined'), ('Delta', 'delta')))
    fig.text(.5, .923, metric_text, ha='center', fontsize=8.7,
             color='#1F2937')
    handles = [Patch(facecolor='none', edgecolor=REGION_COLORS[r], linewidth=2, label=r) for r in REGIONS]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, .02), title='Contour colors')
    fig.text(.5, .002, f'Slice rule: {slice_rule}. Top modality rows use GT contours for anatomical context; '
             'prediction/GT rows use FLAIR as the grayscale background.', ha='center', fontsize=8.6)
    fig.subplots_adjust(left=.15, right=.99, top=.90, bottom=.065, hspace=.10, wspace=.025)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return z, slice_rule


def main():
    root = Path.home() / 'Major_Project'
    fold_root = root / 'ReDiffiNet_Output_20Steps_ancflJ/fold_0'
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = (
        ('data-dir', root / 'ReDiffiNet_Data_Fold0_Clean'),
        ('assignments', root / 'UNet_Output_Light_BF16/fold_0/fold_assignments.csv'),
        ('verification', root / 'ReDiffiNet_Output/audit/fold_0/verification.json'),
        ('checkpoint', fold_root / 'checkpoints/best_mean_dice.pt'),
        ('evaluation-dir', fold_root / 'evaluation/best_checkpoint'),
        ('output-dir', fold_root / 'result_visuals'),
    )
    for name, default in defaults:
        parser.add_argument('--' + name, type=lambda p: Path(p).expanduser().resolve(), default=default)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--subject-id', help='Explicit validation subject_id for the case figure')
    group.add_argument('--selection', choices=('best-improved', 'representative'), default='best-improved',
                       help='Automatic selection when --subject-id is absent')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error('--dpi must be at least 150')
    out = args.output_dir.resolve()
    protected = (args.data_dir, args.assignments.parent.parent, args.checkpoint.parent,
                 args.evaluation_dir, args.verification.parent, Path(__file__).resolve().parent)
    if any(out == p.resolve() or p.resolve() in out.parents for p in protected):
        raise ValueError('Output must be outside data, checkpoints, evaluation, audit, and source directories')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output must be absent or empty; refusing to overwrite visual results')

    metrics_path = args.evaluation_dir / 'metrics.json'
    cases_path = args.evaluation_dir / 'paired_per_case.csv'
    if not metrics_path.is_file() or not cases_path.is_file():
        raise FileNotFoundError('Expected metrics.json and paired_per_case.csv in --evaluation-dir')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('fold') != 0 or metrics.get('metric_version') != METRIC_VERSION:
        raise ValueError('metrics.json is not the expected Fold-0 paired evaluation')
    for prefix in ('baseline', 'refined', 'delta'):
        if set(metrics.get(prefix, {})) != set(METRICS):
            raise ValueError(f'metrics.json missing {prefix} WT/TC/ET/Mean')
        for region in METRICS:
            metrics[prefix][region] = float(metrics[prefix][region])
    if any(abs(metrics['delta'][r] - (metrics['refined'][r] - metrics['baseline'][r])) > 1e-12 for r in METRICS):
        raise ValueError('metrics.json has inconsistent deltas')

    import torch
    from train import source_identity
    from unet.refinement import DiffUNet
    from visualize_comparison import sample_volume
    export = FoldExport(args.data_dir, args.assignments, 0)
    if (len(export.ids['training']), len(export.ids['validation'])) != (1000, 251):
        raise ValueError('Expected the confirmed 1,000 training / 251 validation export')
    verification = require_verification(args.verification, export)
    if any(abs(metrics['baseline'][r] - verification['baseline'][r]) > 1e-12 for r in METRICS):
        raise ValueError('metrics.json baseline differs from accepted verification')
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint, 0, export.identity())
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'monai', 'numpy', 'tensorboard')}
    if checkpoint['sources'] != source_identity() or checkpoint['versions'] != versions:
        raise ValueError('Code/dependencies differ from checkpoint; visualization rejected')
    if checkpoint.get('diagnostic') or checkpoint['epoch'] != checkpoint['best_epoch']:
        raise ValueError('Checkpoint must be the research best checkpoint')
    for prefix in ('baseline', 'refined', 'delta'):
        if any(abs(metrics[prefix][r] - checkpoint['best_metrics'][prefix][r]) > 1e-12 for r in METRICS):
            raise ValueError('metrics.json does not match best checkpoint metrics')
    rows = checked_rows(cases_path, export.ids['validation'])
    selected, selection_reason = choose_case(rows, args.subject_id, args.selection, metrics['delta']['Mean'])
    print(f'Selected {selected["subject_id"]}: {selection_reason}', flush=True)

    # Only the chosen case is sampled. Full-fold numbers are read from accepted outputs.
    with tempfile.TemporaryDirectory(prefix='rediffinet-result-figures-') as mpl_cache:
        os.environ['MPLCONFIGDIR'] = mpl_cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        configure_determinism(checkpoint['config']['seed'], checkpoint['config']['determinism'])
        device, dtype = runtime(args.device, checkpoint['config']['amp'])
        model = DiffUNet(**checkpoint['config']['model']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ds = ExportDataset(export, 'validation')
        item = ds[ds.ids.index(selected['subject_id'])]
        refined = sample_volume(model, item, device, dtype, checkpoint['config'])
        exact = exact_case_metrics(item, refined)
        if not same_metrics(selected, exact):
            raise ValueError('Selected-case inference does not reproduce paired_per_case.csv')
        out.mkdir(parents=True, exist_ok=True)
        plot_overall(plt, metrics, out / 'overall_results_summary.png', args.dpi)
        z, slice_rule = plot_case(plt, item, refined, exact, selection_reason,
                                  out / f'case_comparison_{item["subject_id"]}.png', args.dpi)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError('Checkpoint changed during visualization; outputs are not accepted')
    summary = dict(fold=0, cases=len(export.ids['validation']), epoch=checkpoint['epoch'],
                   checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                   evaluation_metrics=str(metrics_path), evaluation_metrics_sha256=sha256(metrics_path),
                   paired_per_case=str(cases_path), paired_per_case_sha256=sha256(cases_path),
                   metric_version=METRIC_VERSION, **{p: metrics[p] for p in ('baseline', 'refined', 'delta')})
    save_json(out / 'overall_results_summary.json', summary)
    case_info = dict(subject_id=item['subject_id'], selection_reason=selection_reason,
                     slice_index=z, slice_rule=slice_rule, **{key: exact[key] for key in exact if key != 'subject_id'},
                     checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                     evaluation_metrics=str(metrics_path), paired_per_case=str(cases_path),
                     note='Actual selected-case checkpoint inference; verified against paired_per_case.csv. '
                          'This selected case illustrates output structure and is not a performance estimate.')
    save_json(out / f'case_comparison_{item["subject_id"]}.json', case_info)
    print(f'Completed: {out}', flush=True)


if __name__ == '__main__':
    main()
