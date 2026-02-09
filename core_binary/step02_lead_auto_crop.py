# ============================================================
# File: core_binary/step02_lead_auto_crop.py
# ============================================================
"""
[CORE_BINARY – PLAIN PYTHON]
Step 02 – Lead auto-cropping from YOLO detections

This step performs automatic cropping of ECG leads based on YOLO
detection results and saves cropped segments, labels, and
visualization images.

This step does NOT contain proprietary algorithms.
DO NOT modify this file during review.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import os


# ==========================
# HARD-CODED PARAMETERS
# ==========================
# NOTE:
# Auto-cropping uses the SAME YOLO model as lead-region detection (Step 02)
MODEL_PATH = "outputs/step02_lead_detection_region_yolo/weights/best.pt"

SKIP_START = 79
TARGET_LENGTH = 590

CROP_SUBDIR_NAME = "segments"
LABEL_SUBDIR_NAME = "labels"
DETECT_SUBDIR_NAME = "detections"


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def crop_lead(image, box, skip_start=0, target_length=None):
    x_min, y_min, x_max, y_max = map(int, box)

    x_min_new = x_min + skip_start

    if target_length:
        x_max_new = x_min_new + target_length
        if x_max_new > x_max:
            x_max_new = min(x_min_new + target_length, image.shape[1])
    else:
        x_max_new = x_max

    x_min_new = max(0, x_min_new)
    x_max_new = min(image.shape[1], x_max_new)
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)

    cropped = image[y_min:y_max, x_min_new:x_max_new]

    if target_length and cropped.shape[1] != target_length:
        cropped = cv2.resize(
            cropped,
            (target_length, cropped.shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

    return cropped


def core_run(input_dir, output_dir):
    """
    Core entry point for Step 02 – Lead auto-cropping.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    crop_root = output_dir / CROP_SUBDIR_NAME
    label_root = output_dir / LABEL_SUBDIR_NAME
    detect_root = output_dir / DETECT_SUBDIR_NAME

    crop_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    detect_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    for image_name in os.listdir(input_dir):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = input_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width, _ = image.shape
        base_name = image_path.stem

        crop_subdir = crop_root / base_name
        crop_subdir.mkdir(parents=True, exist_ok=True)

        results = model.predict(source=str(image_path), save=False)
        result = results[0]

        detect_img = result.plot()
        cv2.imwrite(
            str(detect_root / f"{base_name}_detect.jpg"),
            detect_img
        )

        label_path = label_root / f"{base_name}.txt"
        with open(label_path, "w") as f:
            selected_boxes = {}

            for box in result.boxes.data.tolist():
                x_min, y_min, x_max, y_max, conf, cls_id = map(float, box[:6])
                cls_id = int(cls_id)

                if cls_id not in selected_boxes:
                    selected_boxes[cls_id] = [x_min, y_min, x_max, y_max, conf]
                else:
                    iou = calculate_iou(
                        [x_min, y_min, x_max, y_max],
                        selected_boxes[cls_id][:4]
                    )
                    if iou < 0.5 or conf > selected_boxes[cls_id][4]:
                        selected_boxes[cls_id] = [x_min, y_min, x_max, y_max, conf]

            for cls_id, (x_min, y_min, x_max, y_max, conf) in selected_boxes.items():
                x_center = ((x_min + x_max) / 2) / width
                y_center = ((y_min + y_max) / 2) / height
                box_width = (x_max - x_min) / width
                box_height = (y_max - y_min) / height

                f.write(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n"
                )

                crop_img = crop_lead(
                    image,
                    [x_min, y_min, x_max, y_max],
                    skip_start=SKIP_START,
                    target_length=TARGET_LENGTH
                )

                crop_path = crop_subdir / f"{model.names[cls_id]}.jpg"
                cv2.imwrite(str(crop_path), crop_img)