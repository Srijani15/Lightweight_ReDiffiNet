"""Existing six-array exports; no new folds, re-normalization, or label inputs."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

CHANNELS = ('t1c', 't1n', 't2f', 't2w', 'WT', 'TC', 'ET')
REGIONS = ('WT', 'TC', 'ET')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def labels_to_regions(labels):
    return np.stack((labels > 0, (labels == 1) | (labels == 3), labels == 3))


def read_case(path):
    """Only trusted locally generated pickle files. Never silently skip corruption."""
    path = Path(path)
    with (gzip.open if path.name.endswith('.gz') else open)(path, 'rb') as handle:
        values = pickle.load(handle)
    if not isinstance(values, (tuple, list)) or len(values) != 6:
        raise ValueError(f'{path}: expected six arrays')
    values = tuple(np.asarray(value) for value in values)
    shape = values[0].shape
    if len(shape) != 3 or min(shape) < 1 or any(v.shape != shape for v in values):
        raise ValueError(f'{path}: inconsistent 3D shapes')
    for value in values[:4]:
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'{path}: MRI must be finite float32')
    for value in values[4:]:
        if value.dtype != np.uint8 or not np.isin(value, (0, 1, 2, 3)).all():
            raise ValueError(f'{path}: expected uint8 labels 0/1/2/3')
    image = np.concatenate((np.stack(values[:4]), labels_to_regions(values[5]).astype(np.float32)))
    return image, labels_to_regions(values[4]).astype(np.float32)


class FoldExport:
    """Authoritative membership, manifest agreement, and export identity."""
    def __init__(self, data_dir, assignments, fold):
        if fold not in range(5):
            raise ValueError('fold must be 0..4')
        self.root = Path(data_dir).expanduser().resolve()
        self.assignments_path = Path(assignments).expanduser().resolve()
        self.fold = fold
        self.manifest_path = self.root / 'export_manifest.csv'
        self.provenance_path = self.root / 'export_provenance.txt'
        if not self.provenance_path.is_file() or not self.provenance_path.read_text(encoding='utf-8-sig').strip():
            raise ValueError('Missing nonempty export_provenance.txt; do not invent provenance')
        with self.assignments_path.open(newline='', encoding='utf-8-sig') as handle:
            self.rows = list(csv.DictReader(handle))
        self.members = {}
        for row in self.rows:
            sid, split, source = row.get('subject_id', ''), row.get('fold_split'), row.get('source_directory')
            if (not sid or '/' in sid or '\\' in sid or sid in ('.', '..')
                    or split not in ('training', 'validation') or source not in ('train', 'validation')):
                raise ValueError(f'Invalid assignment: {row}')
            if sid in self.members:
                raise ValueError(f'Duplicate patient: {sid}')
            self.members[sid] = (source, split)
        self.paths, self.ids = {}, {}
        for split, directory in (('training', 'train'), ('validation', 'validation')):
            expected = {sid for sid, (_, part) in self.members.items() if part == split}
            if not expected:
                raise ValueError(f'Empty {split} fold')
            found = {}
            for path in sorted((self.root / directory).glob('*')):
                suffix = '.pkl.gz' if path.name.endswith('.pkl.gz') else '.pkl'
                if not path.name.endswith(suffix):
                    continue
                sid = path.name[:-len(suffix)]
                if sid in found or not path.is_file():
                    raise ValueError(f'Duplicate/invalid export: {path}')
                found[sid] = path
            if set(found) != expected:
                raise ValueError(f'{split}: missing={sorted(expected-set(found))[:5]}, '
                                 f'unexpected={sorted(set(found)-expected)[:5]}')
            self.paths.update(found)
            self.ids[split] = sorted(expected)
        with self.manifest_path.open(newline='', encoding='utf-8-sig') as handle:
            manifest = list(csv.DictReader(handle))
        seen = set()
        for row in manifest:
            sid = row.get('subject_id')
            if sid in seen or sid not in self.members:
                raise ValueError('Manifest has duplicate or unknown patients')
            if (row.get('source_directory'), row.get('fold_split')) != self.members[sid]:
                raise ValueError(f'Manifest split/source disagreement: {sid}')
            tail = row.get('output_path', '').replace('\\', '/').split('/')[-2:]
            if tail != [self.paths[sid].parent.name, self.paths[sid].name]:
                raise ValueError(f'Manifest path disagreement: {sid}')
            seen.add(sid)
        if seen != set(self.members):
            raise ValueError('Manifest does not cover assignments')

    def identity(self):
        # Verification also records full validation hashes. This inventory detects
        # ordinary later changes without rereading 13 GB on every resume.
        inventory = [(sid, self.paths[sid].stat().st_size, self.paths[sid].stat().st_mtime_ns)
                     for sid in sorted(self.paths)]
        return dict(fold=self.fold, assignments_sha256=sha256(self.assignments_path),
                    manifest_sha256=sha256(self.manifest_path), provenance_sha256=sha256(self.provenance_path),
                    inventory_sha256=hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
                    training_cases=len(self.ids['training']), validation_cases=len(self.ids['validation']))


class ExportDataset:
    def __init__(self, export, split, patch_size=None, limit=0):
        self.export = export
        self.ids = export.ids[split][:limit or None]
        self.patch_size = tuple(patch_size) if patch_size else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        image, target = read_case(self.export.paths[sid])
        if self.patch_size:
            shape = image.shape[1:]
            if any(p > s for p, s in zip(self.patch_size, shape)):
                raise ValueError(f'Patch {self.patch_size} exceeds {sid}: {shape}')
            # Uniform label-independent crops, seeded per worker/epoch.
            starts = [np.random.randint(s-p+1) for p, s in zip(self.patch_size, shape)]
            slices = (slice(None),) + tuple(slice(a, a+p) for a, p in zip(starts, self.patch_size))
            image, target = image[slices], target[slices]
        return dict(subject_id=sid, image=np.ascontiguousarray(image), label=np.ascontiguousarray(target))

