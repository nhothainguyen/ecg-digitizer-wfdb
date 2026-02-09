# ============================================================
# File: core_binary/step03_signal_cleanup_and_mask.py
# ============================================================
"""
[CORE_BINARY]
Step 03 – Signal cleanup and binary mask generation

This file wraps compiled core algorithms.
DO NOT modify this file during review.
"""

import random
import numpy as np
from pathlib import Path
import cv2
from skimage.io import imread

# reproducibility
random.seed(0)
np.random.seed(0)

# import CORE (Cython)
from core_binary._step03_core import (
    classify_background_type,
    process_pink_grid,
    extract_signal_dot_grid,
    smooth_and_interpolate_signal,
)


def core_run(input_dir, output_dir):
    input_dir = Path(input_dir)

    pipe_root = Path(output_dir) / "step03_signal_cleanup_and_mask"
    eval_root = Path(output_dir) / "step03_centerline_reference_mask"

    pipe_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    for record_dir in sorted(input_dir.iterdir()):
        if not record_dir.is_dir():
            continue

        record_id = record_dir.name
        pipe_dir = pipe_root / record_id
        eval_dir = eval_root / record_id
        pipe_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        for lead_path in sorted(f for f in record_dir.iterdir() if f.is_file()):
            img = imread(str(lead_path))
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)

            image_rgb = (
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.ndim == 3
                else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            )

            bg = classify_background_type(image_rgb)

            if bg == "pink_line":
                processed = process_pink_grid(image_rgb)
                eval_mask = processed.copy()
            else:
                raw = extract_signal_dot_grid(image_rgb)
                processed = smooth_and_interpolate_signal(raw)
                eval_mask = raw

            cv2.imwrite(str(pipe_dir / lead_path.name), processed)
            cv2.imwrite(str(eval_dir / lead_path.name), eval_mask)