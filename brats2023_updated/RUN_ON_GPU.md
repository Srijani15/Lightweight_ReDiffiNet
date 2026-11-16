# BraTS 2023 single-GPU baseline U-Net

This portable guide covers the baseline only; Re-DiffiNet is not changed. Run
commands from the directory above `brats2023_updated/`. The original first-level
width is **64**, giving encoder widths **64, 96, 128, 192, 256, 384, 512**.
Use `--base-channels 64` for the exact original model or `32` for half width.

## Environment check

Install the server's CUDA-enabled PyTorch build first, then install the package
requirements. This avoids replacing a working CUDA build with a CPU wheel.

```bash
conda activate rediffinet
cd ~/Major_Project
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
pip install -r brats2023_updated/requirements.txt
```

The intended environment is Python 3.11, PyTorch 2.5.1+cu121, CUDA 12.1,
cuDNN 9.1, MONAI 1.4, and NumPy 1.26.x. After `CUDA_VISIBLE_DEVICES=0`, use
logical `cuda:0` only.

## Smoke test

```bash
python -m brats2023_updated.model_routines.train_baseline \
  --train-dir /path/to/BraTS2023_split/train \
  --val-dir /path/to/BraTS2023_split/validation \
  --out-dir /path/to/UNet_Smoke_Test \
  --n-folds 5 --fold-number 0 --fold-source all \
  --device cuda:0 --epochs 1 --batch-size 1 --workers 2 --val-interval 1 \
  --max-train-samples 10 --max-val-samples 5 --amp
```

Sample limits are deterministic, written to `run_config.json`, and disabled by
default. Do not use them in a full run.

## Optimized Fold 0

For an RTX A6000, start with four workers; reduce them when NIfTI storage or CPU
is saturated. TF32 and cuDNN benchmark default to enabled on CUDA.

```bash
python -m brats2023_updated.model_routines.train_baseline \
  --train-dir /path/to/BraTS2023_split/train \
  --val-dir /path/to/BraTS2023_split/validation \
  --out-dir /path/to/UNet_Output_Original \
  --n-folds 5 --fold-number 0 --fold-source all \
  --device cuda:0 --epochs 200 --batch-size 1 --workers 4 \
  --val-interval 5 --early-stopping-patience 4 --min-delta 0.001 \
  --base-channels 64 --amp
```

Half-width experiment:

```bash
python -m brats2023_updated.model_routines.train_baseline \
  --train-dir /path/to/BraTS2023_split/train \
  --val-dir /path/to/BraTS2023_split/validation \
  --out-dir /path/to/UNet_Output_Light \
  --n-folds 5 --fold-number 0 --fold-source all \
  --device cuda:0 --epochs 200 --batch-size 1 --workers 4 \
  --val-interval 5 --early-stopping-patience 4 --min-delta 0.001 \
  --base-channels 32 --amp
```

Validation runs at 5, 10, 15, … and early stopping counts only validation
checks. Use a new output directory for each width/crop experiment. `fold_0/`
contains `fold_assignments.csv`, `run_config.json`, `model/latest.pt`, best
Mean-Dice/loss checkpoints, and backup checkpoints; TensorBoard is under
`OUT_DIR/tensorboard/fold_0/`.

## TensorBoard, resume, and later folds

```bash
tensorboard --logdir /path/to/UNet_Output_Light/tensorboard --port 6006 --bind_all
```

Rerunning the identical command resumes automatically from
`OUT_DIR/fold_X/model/latest.pt`. Or add:

```bash
  --resume /path/to/UNet_Output_Light/fold_0/model/latest.pt
```

The checkpoint stores model/optimizer/AMP-GradScaler state, next epoch/batch,
global step, early-stopping and best metrics, architecture/crop metadata, and
Python/NumPy/CPU/CUDA RNG state. It rejects conflicting fold, seed, crop, AMP,
or base-channel settings. Change only `--fold-number 0` to 1, 2, 3, or 4 for
later folds; this trainer never starts them automatically.

Inspect model size without data:

```bash
python -m brats2023_updated.model_routines.inspect_model --base-channels 64
python -m brats2023_updated.model_routines.inspect_model --base-channels 32
```

## Optional reduced crop

Omit `--crop-size` to preserve the historical fixed `(128, 192, 128)` crop
exactly. An explicit `--crop-size X Y Z` is deterministic and image-centred,
never tumour-label guided. Dimensions must be multiples of 64 and at least 128;
for example, `--crop-size 128 128 128`. It is a separate experiment and needs
matching crop settings for downstream prediction generation.
