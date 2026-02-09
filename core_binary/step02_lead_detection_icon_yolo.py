# ============================================================
# File: core_binary/step02_lead_detection_icon_yolo.py
# ============================================================
"""
[CORE_BINARY – PLAIN PYTHON]
Step 02 – Lead icon / label detection (YOLO)

This step performs lead icon/label detection using a pretrained YOLO model.
All raw YOLO predictions are copied into the step output directory to ensure
a consistent and reviewer-friendly pipeline structure.

This step does NOT contain proprietary algorithms.
DO NOT modify this file during review.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil


# ==========================
# HARD-CODED PARAMETERS
# ==========================
MODEL_PATH = "outputs/step02_lead_detection_icon_yolo/weights/best.pt"
YOLO_CONF = 0.05

YOLO_RUNS_PREDICT = Path("runs/detect/predict")


def core_run(input_dir, output_dir):
    """
    Core entry point for Step 02 (lead icon detection).

    Parameters
    ----------
    input_dir : str or Path
        Directory produced by the previous pipeline step.
    output_dir : str or Path
        Directory where this step's results will be written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Final, pipeline-consistent location for YOLO outputs
    yolo_output_dir = output_dir / "predictions_raw"
    yolo_output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Run YOLO prediction (YOLO writes to runs/detect/predict internally)
    model(source=str(input_dir), conf=YOLO_CONF, save=True)

    # Copy raw YOLO prediction results into pipeline output directory
    if YOLO_RUNS_PREDICT.exists():
        shutil.copytree(
            YOLO_RUNS_PREDICT,
            yolo_output_dir,
            dirs_exist_ok=True
        )

    # Clean up temporary YOLO directories
    if Path("runs").exists():
        shutil.rmtree("runs")