"""Exact U-Net metrics, shared evaluation, verification guards and atomic output."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from brats_data_utils_multi_label_5folds import REGIONS, labels_to_regions

METRIC_VERSION = 'unet-pooled-float64-threshold-045-040-045-v1'
FORMAT_VERSION = 1
HISTORICAL_FOLD0 = np.array([0.9342993500350412, 0.9205175596223508, 0.8974118156008125])


def postprocess(probabilities):
    """Exact overlapping_probs_to_preds then disjoint_to_overlapping semantics."""
    p = np.asarray(probabilities)
    if p.shape[0] != 3 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite WT/TC/ET probabilities in [0,1]')
    wt, tc, et = p[0] > .45, p[1] > .40, p[2] > .45
    labels = wt.astype(np.uint8)
    labels[wt & ~tc] = 2
    labels[wt & et] = 3
    return labels_to_regions(labels)


class DiceAccumulator:
    def __init__(self):
        self.counts = np.zeros((3, 3), dtype=np.float64)

    def add(self, predicted, target):
        p, y = np.asarray(predicted, dtype=bool), np.asarray(target, dtype=bool)
        if p.shape != y.shape or p.shape[0] != 3:
            raise ValueError('Dice shape mismatch')
        axes = tuple(range(1, p.ndim))
        counts = np.stack(((p & y).sum(axis=axes, dtype=np.float64),
                           p.sum(axis=axes, dtype=np.float64), y.sum(axis=axes, dtype=np.float64)))
        self.counts += counts
        return self.from_counts(counts)

    @staticmethod
    def from_counts(counts):
        scores = (2 * counts[0] + 1e-8) / (counts[1] + counts[2] + 1e-8)
        return dict(zip((*REGIONS, 'Mean'), [*map(float, scores), float(scores.mean())]))

    def scores(self):
        return self.from_counts(self.counts)


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        write(Path(temporary))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8'))


def save_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write empty metrics')
    def write(tmp):
        with tmp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(path, write)


def fold_paths(output, fold, diagnostic=False):
    if fold not in range(5):
        raise ValueError('Invalid fold')
    root = Path(output).expanduser().resolve()
    if diagnostic:
        root /= 'diagnostic'
    base = root / f'fold_{fold}'
    return dict(root=root, fold=base, checkpoints=base/'checkpoints',
                tensorboard=base/'tensorboard', evaluation=base/'evaluation')


def require_verification(path, export):
    report = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
    if report.get('metric_version') != METRIC_VERSION or report.get('identity') != export.identity():
        raise ValueError('Verification belongs to different/changed data, assignments, or fold')
    if not report.get('approved_for_diagnostic'):
        raise ValueError('Export verification unresolved; inspect differences before training')
    if report.get('evaluated_cases') != len(export.ids['validation']):
        raise ValueError('Verification did not cover the complete validation fold')
    return report


def validate_checkpoint(checkpoint, fold, identity, config=None):
    if checkpoint.get('format_version') != FORMAT_VERSION or checkpoint.get('fold') != fold:
        raise ValueError('Checkpoint format/fold mismatch')
    if checkpoint.get('metric_version') != METRIC_VERSION or checkpoint.get('identity') != identity:
        raise ValueError('Checkpoint data/metric mismatch')
    if config is not None and checkpoint.get('config') != config:
        raise ValueError('Resume configuration changed; use saved configuration or a new output root')


def summarize(rows):
    if not rows:
        raise ValueError('No completed research folds')
    keys = [k for k in rows[0] if k.startswith(('baseline_', 'refined_', 'delta_'))]
    return {'completed_folds': [int(r['fold']) for r in rows],
            'complete_five_fold_cv': sorted(int(r['fold']) for r in rows) == list(range(5)),
            'aggregation': 'unweighted mean of pooled fold Dice; population std (ddof=0)',
            'metrics': {k: {'mean': float(np.mean([float(r[k]) for r in rows])),
                            'std': float(np.std([float(r[k]) for r in rows], ddof=0))} for k in keys}}


def update_summary(output, row):
    path = Path(output)/'cross_validation_summary.csv'
    previous = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as handle:
            previous = list(csv.DictReader(handle))
    rows = [r for r in previous if int(r['fold']) != int(row['fold'])] + [row]
    rows.sort(key=lambda r: int(r['fold']))
    save_csv(path, rows)
    return summarize(rows)


def paired_row(sid, baseline, refined):
    row = {'subject_id': sid}
    for region in (*REGIONS, 'Mean'):
        row[f'baseline_{region}'] = baseline[region]
        row[f'refined_{region}'] = refined[region]
        row[f'delta_{region}'] = refined[region] - baseline[region]
    return row


def evaluate(model, dataset, device, amp_dtype, roi, overlap, seed):
    """Full exported volumes, CPU stitched output, no label-conditioned sampling."""
    import torch
    from monai.inferers import sliding_window_inference
    baseline_total, refined_total = DiceAccumulator(), DiceAccumulator()
    rows = []
    model.eval()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        for item in dataset:
            sid = item['subject_id']
            case_seed = (seed + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % (2**31)
            torch.manual_seed(case_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(case_seed)
            image = torch.from_numpy(item['image']).unsqueeze(0)
            def predict(window):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    return model.sample(window).float()
            output = sliding_window_inference(image, roi, 1, predict, overlap=overlap,
                                              mode='gaussian', sw_device=device, device=torch.device('cpu'))
            # Floating-point blending can exceed [0,1] by a few ulps.
            output = output.clamp(0, 1)
            baseline = baseline_total.add(postprocess(item['image'][4:]), item['label'])
            refined = refined_total.add(postprocess(output[0].numpy()), item['label'])
            rows.append(paired_row(sid, baseline, refined))
    baseline, refined = baseline_total.scores(), refined_total.scores()
    return dict(baseline=baseline, refined=refined,
                delta={r: refined[r]-baseline[r] for r in (*REGIONS, 'Mean')},
                baseline_counts=baseline_total.counts.tolist(), refined_counts=refined_total.counts.tolist(),
                count_rows=['intersection', 'prediction_volume', 'target_volume'], count_columns=REGIONS,
                cases=len(rows), metric_version=METRIC_VERSION), rows


def runtime(device_name, amp):
    import torch
    device = torch.device(device_name)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Supported devices: cpu and cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
    if amp == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 unavailable; explicitly select --amp off or fp16 on CUDA')
    if amp == 'fp16' and device.type != 'cuda':
        raise ValueError('FP16 supported only on CUDA')
    return device, {'off': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[amp]


def configure_determinism(seed, policy='warn-only'):
    """Keep seeded streams; warn-only permits nondeterministic CUDA operators."""
    if policy not in ('strict', 'warn-only', 'off'):
        raise ValueError(f'Unknown determinism policy: {policy}')
    import random
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(policy != 'off', warn_only=policy == 'warn-only')

