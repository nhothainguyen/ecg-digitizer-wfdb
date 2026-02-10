import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import skeletonize

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

from sklearn.metrics import jaccard_score, f1_score, mean_absolute_error
from medpy.metric.binary import hd, assd


# ======================================================
# CONFIG
# ======================================================
IMAGE_FOLDER = "outputs/step02_lead_auto_crop/segments_test/0607458"
MASK_DIR = "outputs/step03_signal_cleanup_and_mask/0607458"
CENTERLINE_DIR = "outputs/step04_digitized_centerline_proposed/0607458/black_white"

# 🎯 file phục vụ TABLE 8
CSV_OUTPUT_PATH = "evaluation/tables/table8.csv"

NUM_LEADS = 12

os.makedirs(CENTERLINE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)


# ======================================================
# ---------- PHÂN LOẠI GRID ----------
# ======================================================
def classify_background_type(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    pink_mask = (h > 140) & (h < 180) & (s > 50) & (v > 150)
    pink_ratio = np.sum(pink_mask) / (image_rgb.shape[0] * image_rgb.shape[1])

    if pink_ratio > 0.01:
        return 'pink_line'

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    small_dots = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 20:
            small_dots += 1

    if small_dots > 500:
        return 'dot_grid'

    return 'pink_line'


# ======================================================
# ---------- PINK GRID ----------
# ======================================================
def process_pink_grid(image_rgb):
    grayscale = color.rgb2gray(image_rgb)
    blurred_image = gaussian(grayscale, sigma=1.3)
    global_thresh = threshold_otsu(blurred_image)
    binary_global = blurred_image < global_thresh
    binary_uint8 = (binary_global * 255).astype(np.uint8)
    return binary_uint8


# ======================================================
# ---------- DOT GRID ----------
# ======================================================
def extract_signal_dot_grid(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 190, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(gray, dtype=np.uint8)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 7:
            mask[labels == i] = 255

    skeleton = skeletonize(mask > 127).astype(np.uint8) * 255
    skeleton_float = skeleton.astype(np.float32) / 255.0
    smoothed = gaussian_filter(skeleton_float, sigma=1)
    smoothed = (smoothed > 0.1).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thick_line = cv2.dilate(smoothed, kernel, iterations=1)

    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(thick_line, connectivity=8)
    final_mask = np.zeros_like(thick_line, dtype=np.uint8)

    for i in range(1, num_labels_final):
        if stats_final[i, cv2.CC_STAT_AREA] >= 30:
            final_mask[labels_final == i] = 255

    return final_mask


# ======================================================
# ---------- SMOOTH & INTERPOLATE ----------
# ======================================================
def smooth_and_interpolate_signal(binary_mask):
    rows, cols = binary_mask.shape

    smooth = cv2.medianBlur(binary_mask, 3)
    _, binary = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)

    ys = []
    for x in range(cols):
        y_indices = np.where(binary[:, x] > 0)[0]
        ys.append(np.mean(y_indices) if len(y_indices) > 0 else np.nan)

    xs = np.arange(cols)
    ys = np.array(ys, dtype=np.float64)

    if np.sum(~np.isnan(ys)) < 2:
        return binary_mask

    f_interp = interp1d(xs[~np.isnan(ys)], ys[~np.isnan(ys)], kind='linear', fill_value='extrapolate')
    ys_interp = f_interp(xs)
    ys_final = np.clip(ys_interp, 0, rows - 1).astype(int)

    result = np.zeros_like(binary_mask)
    prev_point = None

    for x, y in zip(xs, ys_final):
        pt = (x, y)
        if prev_point is not None:
            cv2.line(result, prev_point, pt, 255, 1)
        prev_point = pt

    result = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    return result


# ======================================================
# CENTERLINE METRICS
# ======================================================
def compute_centerline_metrics(mask, centerline):

    mask_bin = (mask > 127).astype(np.uint8)
    centerline_bin = (centerline > 127).astype(np.uint8)

    overlap = (centerline_bin & mask_bin)
    coverage = np.sum(overlap) / np.sum(centerline_bin) if np.sum(centerline_bin) > 0 else 0

    cols = mask.shape[1]
    mask_ys, center_ys = [], []

    for x in range(cols):
        mask_col_y = np.where(mask_bin[:, x])[0]
        center_col_y = np.where(centerline_bin[:, x])[0]

        if len(mask_col_y) > 0 and len(center_col_y) > 0:
            mask_center_y = np.mean(mask_col_y)
            center_y = center_col_y[0]
            mask_ys.append(mask_center_y)
            center_ys.append(center_y)

    mae = mean_absolute_error(mask_ys, center_ys) if len(mask_ys) > 0 else np.nan

    try:
        hd_val = hd(centerline_bin, mask_bin)
        assd_val = assd(centerline_bin, mask_bin)
    except:
        hd_val, assd_val = np.nan, np.nan

    return {
        'Coverage (%)': round(coverage * 100, 2),
        'MAE (px)': round(mae, 2) if not np.isnan(mae) else 'NaN',
        'Hausdorff Distance': round(hd_val, 2) if not np.isnan(hd_val) else 'NaN',
        'ASSD': round(assd_val, 2) if not np.isnan(assd_val) else 'NaN'
    }


# ======================================================
# SEGMENTATION METRICS
# ======================================================
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


# ======================================================
# LOAD GT
# ======================================================
def load_ground_truth_mask(file_name):
    file_stem = Path(file_name).stem

    for ext in ['.png', '.PNG', '.jpg', '.jpeg']:
        mask_path = os.path.join(MASK_DIR, file_stem + ext)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
            return (mask > 127).astype(np.uint8)

    return None


# ======================================================
# MAIN
# ======================================================
def main():

    lead_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if os.path.isfile(os.path.join(IMAGE_FOLDER, f))
    ]

    Leads = []
    for file_name in lead_files:
        image_path = os.path.join(IMAGE_FOLDER, file_name)
        image = imread(image_path)
        Leads.append((file_name, image))

    results = []

    for x, (file_name, lead_image) in enumerate(Leads[:NUM_LEADS]):

        if lead_image.max() <= 1.0:
            lead_image = (lead_image * 255).astype(np.uint8)
        else:
            lead_image = lead_image.astype(np.uint8)

        if lead_image.ndim == 2:
            image_bgr = cv2.cvtColor(lead_image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = cv2.cvtColor(lead_image, cv2.COLOR_RGB2BGR)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        bg_type = classify_background_type(image_rgb)

        if bg_type == 'pink_line':
            signal_mask = process_pink_grid(image_rgb)
            processed = signal_mask
        else:
            signal_mask = extract_signal_dot_grid(image_rgb)
            processed = smooth_and_interpolate_signal(signal_mask)

        processed_resized = cv2.resize(
            processed,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        processed_binary = (processed_resized > 127).astype(np.uint8)

        gt_mask = load_ground_truth_mask(file_name)

        if gt_mask is not None:
            gt_mask_resized = cv2.resize(
                gt_mask,
                (processed_binary.shape[1], processed_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            seg_metrics = compute_segmentation_metrics(processed_binary, gt_mask_resized)
        else:
            seg_metrics = {"IoU": None, "Dice": None}

        metrics = compute_centerline_metrics(signal_mask, processed_resized)

        save_name = Path(file_name).stem + "_centerline.png"
        save_path = os.path.join(CENTERLINE_DIR, save_name)
        cv2.imwrite(save_path, processed_resized)

        results.append({
            'File': file_name,
            'Background': bg_type,
            **metrics,
            **seg_metrics
        })

        print(f"✅ Done: {file_name}")

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(CSV_OUTPUT_PATH, index=False)

    print("\n================ CSV PREVIEW ================")
    print(df_metrics.head())

    print("\n📄 File saved at:")
    print(CSV_OUTPUT_PATH)

    print(f"\n✅ Saved {len(results)} centerlines to: {CENTERLINE_DIR}")
    print("✅ Table 8 CSV exported successfully.")


if __name__ == "__main__":
    main()