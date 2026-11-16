import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Loading helpers
# =========================

def load_export_case(case_path):
    case_path = Path(case_path)
    with gzip.open(case_path, "rb") as f:
        obj = pickle.load(f)

    # Supports dict-style export
    if isinstance(obj, dict):
        t1c = obj["t1c"]
        t1n = obj["t1n"]
        t2f = obj["t2f"]
        t2w = obj["t2w"]
        gt = obj["ground_truth"]
        baseline = obj["baseline_prediction"]
        return t1c, t1n, t2f, t2w, gt, baseline

    # Supports tuple/list export
    if isinstance(obj, (list, tuple)) and len(obj) == 6:
        return obj[0], obj[1], obj[2], obj[3], obj[4], obj[5]

    raise ValueError("Unsupported export format in pkl.gz")


def load_prediction(pred_path):
    pred_path = Path(pred_path)

    if pred_path.suffix == ".npy":
        arr = np.load(pred_path)
        return arr

    if pred_path.suffix == ".npz":
        data = np.load(pred_path)
        # tries common names
        for key in ["prediction", "pred", "mask", "masks", "arr_0"]:
            if key in data:
                return data[key]
        # fallback first key
        first_key = list(data.keys())[0]
        return data[first_key]

    raise ValueError("Prediction file must be .npy or .npz")


# =========================
# Mask conversion helpers
# =========================

def to_three_region_masks(arr):
    """
    Convert prediction/GT to 3 binary masks: WT, TC, ET
    Output shape: (3, H, W, D)

    Supports:
    1) shape (3, H, W, D) directly
    2) shape (H, W, D) label map with BraTS labels:
       WT = labels {1,2,4}
       TC = labels {1,4}
       ET = label {4}
    """
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[0] == 3:
        return arr.astype(bool)

    if arr.ndim == 3:
        wt = np.isin(arr, [1, 2, 4])
        tc = np.isin(arr, [1, 4])
        et = (arr == 4)
        return np.stack([wt, tc, et], axis=0)

    raise ValueError(f"Unsupported mask shape: {arr.shape}")


# =========================
# Dice helpers
# =========================

def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    return (2.0 * inter) / (pred_sum + target_sum + 1e-8)


def compute_region_scores(pred_3ch, gt_3ch):
    wt = dice_score(pred_3ch[0], gt_3ch[0])
    tc = dice_score(pred_3ch[1], gt_3ch[1])
    et = dice_score(pred_3ch[2], gt_3ch[2])
    mean = (wt + tc + et) / 3.0
    return {"WT": wt, "TC": tc, "ET": et, "Mean": mean}


# =========================
# Visualization helpers
# =========================

def normalize_image(x):
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    return x


def choose_slice(gt_3ch, baseline_3ch=None, refined_3ch=None):
    """
    Best slice selection:
    1) largest GT WT area
    2) if tie, largest disagreement between baseline and refined
    """
    wt_gt = gt_3ch[0]
    gt_area = wt_gt.sum(axis=(0, 1))  # over H, W -> per slice

    if baseline_3ch is not None and refined_3ch is not None:
        disagreement = (baseline_3ch[0] != refined_3ch[0]).sum(axis=(0, 1))
    else:
        disagreement = np.zeros_like(gt_area)

    candidates = np.where(gt_area == gt_area.max())[0]
    if len(candidates) == 1:
        return int(candidates[0])

    best = candidates[np.argmax(disagreement[candidates])]
    return int(best)


def draw_mask_contour(ax, img, mask, color, title=""):
    ax.imshow(img, cmap="gray")
    if np.any(mask):
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def draw_overlay(ax, img, wt, tc, et, title=""):
    ax.imshow(img, cmap="gray")
    if np.any(wt):
        ax.contour(wt.astype(np.uint8), levels=[0.5], colors=["orange"], linewidths=1.5)
    if np.any(tc):
        ax.contour(tc.astype(np.uint8), levels=[0.5], colors=["violet"], linewidths=1.5)
    if np.any(et):
        ax.contour(et.astype(np.uint8), levels=[0.5], colors=["deepskyblue"], linewidths=1.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


# =========================
# Main figure function
# =========================

def make_figure(
    subject_id,
    t1ce, t1, flair, t2,
    gt_3ch,
    baseline_3ch,
    refined_3ch,
    baseline_scores,
    refined_scores,
    output_path,
    slice_idx=None
):
    if slice_idx is None:
        slice_idx = choose_slice(gt_3ch, baseline_3ch, refined_3ch)

    # modality slices
    mods = {
        "T1ce": normalize_image(t1ce[:, :, slice_idx]),
        "T1": normalize_image(t1[:, :, slice_idx]),
        "FLAIR": normalize_image(flair[:, :, slice_idx]),
        "T2": normalize_image(t2[:, :, slice_idx]),
    }

    gt_s = gt_3ch[:, :, :, slice_idx]
    base_s = baseline_3ch[:, :, :, slice_idx]
    ref_s = refined_3ch[:, :, :, slice_idx]

    fig, axes = plt.subplots(7, 4, figsize=(15, 22))
    fig.patch.set_facecolor("white")

    main_title = f"Validation case: {subject_id} | axial slice {slice_idx}"
    score_title = (
        f"U-Net: WT {baseline_scores['WT']:.4f} | TC {baseline_scores['TC']:.4f} | "
        f"ET {baseline_scores['ET']:.4f} | Mean {baseline_scores['Mean']:.4f}     "
        f"ReDiffiNet: WT {refined_scores['WT']:.4f} | TC {refined_scores['TC']:.4f} | "
        f"ET {refined_scores['ET']:.4f} | Mean {refined_scores['Mean']:.4f}"
    )
    fig.suptitle(main_title + "\n" + score_title, fontsize=14, fontweight="bold", y=0.995)

    col_titles = ["WT", "TC", "ET", "Overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold")

    # rows 0-3 -> modalities with GT contours
    modality_names = ["T1ce", "T1", "FLAIR", "T2"]
    for i, mod_name in enumerate(modality_names):
        img = mods[mod_name]

        draw_mask_contour(axes[i, 0], img, gt_s[0], "orange")
        draw_mask_contour(axes[i, 1], img, gt_s[1], "violet")
        draw_mask_contour(axes[i, 2], img, gt_s[2], "deepskyblue")
        draw_overlay(axes[i, 3], img, gt_s[0], gt_s[1], gt_s[2])

        axes[i, 0].set_ylabel(
            mod_name,
            fontsize=11,
            fontweight="bold",
            rotation=0,
            labelpad=45,
            va="center"
        )

    # row 4 -> U-Net prediction
    flair_img = mods["FLAIR"]
    row = 4
    draw_mask_contour(axes[row, 0], flair_img, base_s[0], "orange")
    draw_mask_contour(axes[row, 1], flair_img, base_s[1], "violet")
    draw_mask_contour(axes[row, 2], flair_img, base_s[2], "deepskyblue")
    draw_overlay(axes[row, 3], flair_img, base_s[0], base_s[1], base_s[2])

    axes[row, 0].set_ylabel(
        "Frozen U-Net\nprediction",
        fontsize=11,
        fontweight="bold",
        rotation=0,
        labelpad=65,
        va="center"
    )

    # row 5 -> ReDiffiNet prediction
    row = 5
    draw_mask_contour(axes[row, 0], flair_img, ref_s[0], "orange")
    draw_mask_contour(axes[row, 1], flair_img, ref_s[1], "violet")
    draw_mask_contour(axes[row, 2], flair_img, ref_s[2], "deepskyblue")
    draw_overlay(axes[row, 3], flair_img, ref_s[0], ref_s[1], ref_s[2])

    axes[row, 0].set_ylabel(
        "ReDiffiNet\nprediction",
        fontsize=11,
        fontweight="bold",
        rotation=0,
        labelpad=65,
        va="center"
    )

    # row 6 -> Ground Truth
    row = 6
    draw_mask_contour(axes[row, 0], flair_img, gt_s[0], "orange")
    draw_mask_contour(axes[row, 1], flair_img, gt_s[1], "violet")
    draw_mask_contour(axes[row, 2], flair_img, gt_s[2], "deepskyblue")
    draw_overlay(axes[row, 3], flair_img, gt_s[0], gt_s[1], gt_s[2])

    axes[row, 0].set_ylabel(
        "Ground\nTruth",
        fontsize=11,
        fontweight="bold",
        rotation=0,
        labelpad=65,
        va="center"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.figtext(
        0.5, 0.01,
        "Contour colors: WT = orange, TC = violet, ET = blue",
        ha="center",
        fontsize=11
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Create a multimodal comparison figure for one BraTS case.")
    parser.add_argument("--case-pkl", required=True, help="Path to exported case .pkl.gz")
    parser.add_argument("--refined-pred", required=True, help="Path to ReDiffiNet prediction (.npy or .npz)")
    parser.add_argument("--subject-id", required=True, help="Case subject ID for title")
    parser.add_argument("--output", default="case_multimodal_comparison.png", help="Output PNG path")
    parser.add_argument("--slice-idx", type=int, default=None, help="Optional axial slice index")
    args = parser.parse_args()

    t1ce, t1, flair, t2, gt_raw, baseline_raw = load_export_case(args.case_pkl)
    refined_raw = load_prediction(args.refined_pred)

    gt_3ch = to_three_region_masks(gt_raw)
    baseline_3ch = to_three_region_masks(baseline_raw)
    refined_3ch = to_three_region_masks(refined_raw)

    baseline_scores = compute_region_scores(baseline_3ch, gt_3ch)
    refined_scores = compute_region_scores(refined_3ch, gt_3ch)

    make_figure(
        subject_id=args.subject_id,
        t1ce=t1ce,
        t1=t1,
        flair=flair,
        t2=t2,
        gt_3ch=gt_3ch,
        baseline_3ch=baseline_3ch,
        refined_3ch=refined_3ch,
        baseline_scores=baseline_scores,
        refined_scores=refined_scores,
        output_path=args.output,
        slice_idx=args.slice_idx
    )


if __name__ == "__main__":
    main()