# ============================================================
# File: core_binary/step04_digitized_viterbi.py
# ============================================================
"""
[CORE_BINARY – PLAIN PYTHON]
Step 04 – Viterbi-based signal extraction (baseline ecgdigitize)

This step applies the baseline Viterbi algorithm from ecgdigitize
to cropped lead images and saves overlays, signal points, and
black-white masks.

This step does NOT contain proprietary algorithms.
DO NOT modify this file during review.
"""

# =================================================
# Setup import path for baselines/ecgdigitize
# =================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = PROJECT_ROOT / "baselines"
sys.path.insert(0, str(BASELINES_DIR))

# =================================================
# Imports (baseline-compatible, UNCHANGED)
# =================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from ecgdigitize.image import openImage
from ecgdigitize.signal import detection as signal_detection
from ecgdigitize.signal.extraction import viterbi


# ==========================
# HARD-CODED PARAMETERS
# ==========================
SEGMENTS_SUBDIR = "segments"
OVERLAY_SUBDIR = "overlay"
POINTS_SUBDIR = "points"
BW_SUBDIR = "black_white"
FIGS_SUBDIR = "figs"
FIG_NAME = "fig8_viterbi.png"


# =================================================
# Utils (UNCHANGED baseline logic)
# =================================================
def draw_viterbi_on_image(image_rgb, signal):
    image_bgr = (
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if image_rgb.ndim == 3
        else cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    )
    overlay = image_bgr.copy()

    points = np.array(signal)
    if points.ndim == 1:
        points = np.stack((np.arange(len(points)), points), axis=1)

    points = points.astype(np.int32)

    cv2.polylines(
        overlay,
        [points],
        isClosed=False,
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return overlay


def signal_to_points(signal):
    signal = np.asarray(signal)
    if signal.ndim == 1:
        return np.stack([np.arange(len(signal)), signal], axis=1).astype(np.int32)
    return signal.astype(np.int32)


def draw_signal_white_on_black(signal, shape):
    img = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    points = np.array(signal)
    if points.ndim == 1:
        points = np.stack((np.arange(len(points)), points), axis=1)

    points = points.astype(np.int32)

    cv2.polylines(
        img,
        [points],
        isClosed=False,
        color=255,
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return img


def save_viterbi_paper_figure(overlays, titles, out_path):
    fig, axes = plt.subplots(4, 3)
    fig.set_size_inches(20, 20)

    idx = 0
    for r in range(4):
        for c in range(3):
            ax = axes[r, c]
            ax.axis("off")
            if idx < len(overlays):
                ax.imshow(cv2.cvtColor(overlays[idx], cv2.COLOR_BGR2RGB))
                ax.set_title(titles[idx], fontsize=10)
            idx += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# =================================================
# Core entry point
# =================================================
def core_run(input_dir, output_dir):
    """
    Core entry point for Step 04 – baseline Viterbi digitization.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for record_dir in sorted(input_dir.iterdir()):
        if not record_dir.is_dir():
            continue

        record_id = record_dir.name

        out_overlay_dir = output_dir / record_id / OVERLAY_SUBDIR
        out_points_dir = output_dir / record_id / POINTS_SUBDIR
        out_bw_dir = output_dir / record_id / BW_SUBDIR

        out_overlay_dir.mkdir(parents=True, exist_ok=True)
        out_points_dir.mkdir(parents=True, exist_ok=True)
        out_bw_dir.mkdir(parents=True, exist_ok=True)

        paper_overlays = []
        paper_titles = []

        lead_files = sorted([f for f in record_dir.iterdir() if f.is_file()])

        for lead_path in lead_files:
            lead_image = imread(lead_path)
            if lead_image.max() <= 1.0:
                lead_image = (lead_image * 255).astype(np.uint8)
            else:
                lead_image = lead_image.astype(np.uint8)

            if lead_image.ndim == 2:
                image_rgb = cv2.cvtColor(lead_image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = lead_image

            # ===== EXACT baseline logic =====
            ecg_img = openImage(lead_path)
            binary = signal_detection.adaptive(ecg_img)
            signal = viterbi.extractSignal(binary)

            if signal is None:
                continue

            overlay = draw_viterbi_on_image(image_rgb, signal)
            cv2.imwrite(str(out_overlay_dir / lead_path.name), overlay)

            points = signal_to_points(signal)
            np.save(out_points_dir / f"{lead_path.stem}.npy", points)

            h, w = lead_image.shape[:2]
            bw_mask = draw_signal_white_on_black(signal, (h, w))
            cv2.imwrite(str(out_bw_dir / f"{lead_path.stem}.png"), bw_mask)

            if len(paper_overlays) < 12:
                overlay_resized = cv2.resize(
                    overlay, (450, 300), interpolation=cv2.INTER_AREA
                )
                paper_overlays.append(overlay_resized)
                paper_titles.append(f"VITERBI – {lead_path.name}")

        if paper_overlays:
            fig_dir = output_dir / FIGS_SUBDIR
            fig_dir.mkdir(parents=True, exist_ok=True)
            save_viterbi_paper_figure(
                paper_overlays,
                paper_titles,
                fig_dir / FIG_NAME
            )