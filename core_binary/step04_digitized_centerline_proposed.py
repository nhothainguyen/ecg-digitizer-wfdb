"""
[CORE_BINARY]
Step 4 (Proposed): Centerline-based signal path extraction.

⚠️ This file belongs to core_binary.
⚠️ DO NOT MODIFY during review or reproduction.
⚠️ All algorithmic cores are compiled (Cython) for IP protection.

Paper reproducibility version.
"""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Reproducibility (HARD)
# =========================
random.seed(0)
np.random.seed(0)

# =========================
# Import CORE (Cython)
# =========================
from ._step04_core import (
    extract_centerline_from_mask,
    centerline_to_points
)

# =========================
# Visualization (NON-CORE)
# =========================
def overlay_centerline_on_lead(lead_img, points):
    if lead_img.ndim == 2:
        lead_bgr = cv2.cvtColor(lead_img, cv2.COLOR_GRAY2BGR)
    else:
        lead_bgr = cv2.cvtColor(lead_img, cv2.COLOR_RGB2BGR)

    overlay = lead_bgr.copy()
    if len(points) > 1:
        cv2.polylines(
            overlay,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    return overlay


def save_centerline_paper_figure(overlays, titles, out_path):
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


# =========================
# PUBLIC ENTRY POINT
# =========================
def core_run(input_mask_dir: Path, input_lead_dir: Path, output_dir: Path):
    """
    ONLY public API for Step 04.
    """

    for record_dir in sorted(input_mask_dir.iterdir()):
        if not record_dir.is_dir():
            continue

        record_id = record_dir.name
        lead_dir = input_lead_dir / record_id
        if not lead_dir.exists():
            continue

        out_center_dir = output_dir / record_id / "centerline"
        out_pts_dir = output_dir / record_id / "points"
        out_overlay_dir = output_dir / record_id / "overlay"

        out_center_dir.mkdir(parents=True, exist_ok=True)
        out_pts_dir.mkdir(parents=True, exist_ok=True)
        out_overlay_dir.mkdir(parents=True, exist_ok=True)

        paper_overlays = []
        paper_titles = []

        for mask_path in sorted(record_dir.iterdir()):
            if not mask_path.is_file():
                continue

            lead_path = lead_dir / mask_path.name
            if not lead_path.exists():
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            lead_img = cv2.imread(str(lead_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or lead_img is None:
                continue

            centerline = extract_centerline_from_mask(mask)
            points = centerline_to_points(centerline)

            cv2.imwrite(str(out_center_dir / mask_path.name), centerline)
            np.save(out_pts_dir / f"{mask_path.stem}.npy", points)

            overlay = overlay_centerline_on_lead(lead_img, points)
            cv2.imwrite(str(out_overlay_dir / mask_path.name), overlay)

            if len(paper_overlays) < 12:
                paper_overlays.append(
                    cv2.resize(overlay, (450, 300), interpolation=cv2.INTER_AREA)
                )
                paper_titles.append(f"PROPOSED – {mask_path.name}")

        if paper_overlays:
            fig_dir = output_dir / "figs"
            fig_dir.mkdir(parents=True, exist_ok=True)
            save_centerline_paper_figure(
                paper_overlays,
                paper_titles,
                fig_dir / "fig9.png"
            )