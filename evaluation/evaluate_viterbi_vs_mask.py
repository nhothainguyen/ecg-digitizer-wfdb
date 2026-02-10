"""
evaluate_viterbi_vs_mask.py

Quantitative evaluation of Viterbi centerline masks against
ground-truth signal masks.

Metrics:
- IoU
- Dice (F1)
- Coverage (%)
- MAE (pixel-wise)
- Hausdorff Distance
- ASSD

IMPORTANT:
This script is a DIRECT conversion from the original notebook code.
NO algorithmic changes have been made.
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import jaccard_score, f1_score, mean_absolute_error
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt


# =================================================
# CONFIG (EDIT PATHS IF NEEDED)
# =================================================
VITERBI_DIR = Path("outputs/step04_digitized_viterbi/0607458/black_white")
MASK_DIR = Path("outputs/step03_signal_cleanup_and_mask/0607458")
OUTPUT_CSV = Path("evaluation/tables/table7.csv")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

assert VITERBI_DIR.exists(), f"❌ Centerline dir not found: {VITERBI_DIR}"
assert MASK_DIR.exists(), f"❌ Mask dir not found: {MASK_DIR}"


# =================================================
# Utility functions (UNCHANGED)
# =================================================
def load_mask_image(path: str) -> np.ndarray | None:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read image: {path}")
        return None
    return (mask > 127).astype(np.uint8)


def compute_segmentation_metrics(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return {"IoU": 1.0, "Dice": 1.0}
    elif np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
        return {"IoU": 0.0, "Dice": 0.0}

    iou = jaccard_score(gt_flat, pred_flat)
    dice = f1_score(gt_flat, pred_flat)
    return {"IoU": iou, "Dice": dice}


def coverage_percent(mask):
    return 100.0 * np.sum(mask) / mask.size


def mae_metric(pred_mask, gt_mask):
    return mean_absolute_error(gt_mask.flatten(), pred_mask.flatten())


def hausdorff_distance(mask1, mask2):
    points1 = np.column_stack(np.where(mask1))
    points2 = np.column_stack(np.where(mask2))

    if len(points1) == 0 or len(points2) == 0:
        return np.nan

    d1 = directed_hausdorff(points1, points2)[0]
    d2 = directed_hausdorff(points2, points1)[0]
    return max(d1, d2)


def assd(mask1, mask2):
    points1 = np.column_stack(np.where(mask1))
    points2 = np.column_stack(np.where(mask2))

    if len(points1) == 0 or len(points2) == 0:
        return np.nan

    dist_transform_2 = distance_transform_edt(1 - mask2)
    ssd_1 = dist_transform_2[points1[:, 0], points1[:, 1]]

    dist_transform_1 = distance_transform_edt(1 - mask1)
    ssd_2 = dist_transform_1[points2[:, 0], points2[:, 1]]

    return (ssd_1.sum() + ssd_2.sum()) / (len(ssd_1) + len(ssd_2))


def classify_background(gt_mask):
    mean_val = np.mean(gt_mask)
    if mean_val > 200:
        return "white"
    elif mean_val > 100:
        return "gray"
    else:
        return "dark"


# =================================================
# Main evaluation loop
# =================================================
def main():
    results = []
    centerline_paths = sorted(VITERBI_DIR.glob("*.png"))

    if not centerline_paths:
        print("⚠️ No centerline images found.")
        return

    for path in centerline_paths:
        file_name = path.name
        file_stem = path.stem.replace("_centerline", "")

        print(f"Evaluating: {file_name}")

        pred_mask = load_mask_image(str(path))
        if pred_mask is None:
            continue

        # ---- find corresponding GT mask ----
        gt_path = None
        for ext in [".png", ".PNG", ".jpg", ".jpeg"]:
            candidate = MASK_DIR / f"{file_stem}{ext}"
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path is None:
            print(f"  ⚠️ Missing GT mask for {file_name}")
            seg_metrics = {"IoU": None, "Dice": None}
            coverage = mae_val = hd = assd_val = background = None
        else:
            gt_mask = load_mask_image(str(gt_path))
            if gt_mask is None:
                seg_metrics = {"IoU": None, "Dice": None}
                coverage = mae_val = hd = assd_val = background = None
            else:
                if pred_mask.shape != gt_mask.shape:
                    gt_mask = cv2.resize(
                        gt_mask,
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                seg_metrics = compute_segmentation_metrics(pred_mask, gt_mask)
                coverage = coverage_percent(pred_mask)
                mae_val = mae_metric(pred_mask, gt_mask)
                hd = hausdorff_distance(pred_mask, gt_mask)
                assd_val = assd(pred_mask, gt_mask)
                background = classify_background(gt_mask)

        results.append({
            "File": file_name,
            "Background": background,
            "Coverage (%)": coverage,
            "MAE (px)": mae_val,
            "Hausdorff Distance": hd,
            "ASSD": assd_val,
            **seg_metrics
        })

    # ---- Save results ----
    
    df = pd.DataFrame(results)

    # Round numeric columns to 6 decimal places (for paper)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(6)

    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Evaluated {len(results)} images.")
    print(f"✅ Results saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()