"""
[CORE_BINARY]

Step 01 – Grid detection & DPI normalization.
This file is part of the protected core_binary pipeline.

DO NOT MODIFY this file during review or reproduction.
Any modification invalidates reproducibility guarantees.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

from core_binary._step01_core import (
    preprocess_for_grid,
    detect_grid_spacing_hough,
    detect_grid_spacing_fft,
    detect_grid_spacing_hough_circles,
)

# ==========================
# 🔒 HARD-CODED PARAMETERS
# ==========================
CONFIG = {
    "major_grid_mm": 5,
    "dpi": 300,
    "output_dpi": (300, 300),
    "min_spacing_px": 20,
    "max_spacing_px": 80,
    "max_scale_factor": 4,
}

MM_TO_INCH = 25.4
PX_PER_MM = CONFIG["dpi"] / MM_TO_INCH
TARGET_GRID_PX = CONFIG["major_grid_mm"] * PX_PER_MM


def _set_seed():
    random.seed(0)
    np.random.seed(0)


def _scale_and_save(image_bgr, scale_factor, output_path):
    new_w = int(image_bgr.shape[1] * scale_factor)
    new_h = int(image_bgr.shape[0] * scale_factor)

    resized = cv2.resize(
        image_bgr, (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(output_path, dpi=CONFIG["output_dpi"])


def core_run(input_dir, output_dir):
    """
    Core entry point for Step 01.
    """
    _set_seed()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*"))

    for f in files:
        if f.suffix.lower() not in [".pdf", ".jpg", ".jpeg", ".png"]:
            continue

        if f.suffix.lower() == ".pdf":
            images = convert_from_path(
                str(f),
                dpi=CONFIG["dpi"],
                first_page=1,
                last_page=1
            )
            if not images:
                continue
            image = cv2.cvtColor(
                np.array(images[0]),
                cv2.COLOR_RGB2BGR
            )
        else:
            image = cv2.imread(str(f))
            if image is None:
                continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        proc = preprocess_for_grid(gray)

        sx, sy = detect_grid_spacing_hough(
            proc,
            CONFIG["min_spacing_px"],
            CONFIG["max_spacing_px"]
        )

        spacing = sy or sx

        if spacing is None or not (CONFIG["min_spacing_px"] < spacing < CONFIG["max_spacing_px"]):
            spacing = detect_grid_spacing_fft(
                proc,
                CONFIG["min_spacing_px"],
                CONFIG["max_spacing_px"]
            )

        if spacing is None or not (CONFIG["min_spacing_px"] < spacing < CONFIG["max_spacing_px"]):
            spacing = detect_grid_spacing_hough_circles(
                proc,
                CONFIG["min_spacing_px"],
                CONFIG["max_spacing_px"]
            )

        if spacing is None:
            spacing = TARGET_GRID_PX

        scale_factor = TARGET_GRID_PX / spacing
        if scale_factor > CONFIG["max_scale_factor"]:
            scale_factor = CONFIG["max_scale_factor"]

        out_path = output_dir / f"{f.stem}.jpg"
        _scale_and_save(image, scale_factor, str(out_path))