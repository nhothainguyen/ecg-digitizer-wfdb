# ============================================================
# File: core_binary/step05_waveform_extraction.py
# ============================================================
"""
[CORE_BINARY – PLAIN PYTHON]
Step 05 – Waveform extraction and resampling

This step converts digitized ECG centerline paths into quantitative
ECG waveforms (mV) and resamples them to a fixed sampling rate.

This step does NOT contain proprietary algorithms.
DO NOT modify this file during review.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample


# =================================================
# HARD-CODED MEDICAL CALIBRATION CONSTANTS
# =================================================
PIXELS_PER_LARGE_BOX = 59        # 1 large box = 59 px
MV_PER_LARGE_BOX = 0.5           # 1 large box = 0.5 mV
MV_PER_PIXEL = MV_PER_LARGE_BOX / PIXELS_PER_LARGE_BOX

DURATION_S = 0.2                 # 200 ms
FS_TARGET = 1000                 # 1000 Hz
NUM_SAMPLES = int(DURATION_S * FS_TARGET)

LEAD_ORDER = [
    "I", "II", "III",
    "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


# =================================================
# Utils (UNCHANGED LOGIC)
# =================================================
def points_to_signal_y(points: np.ndarray, width: int) -> np.ndarray:
    y_signal = np.full(width, np.nan, dtype=np.float64)

    for x, y in points:
        if 0 <= x < width:
            y_signal[int(x)] = y

    y_signal = (
        pd.Series(y_signal)
        .interpolate(limit_direction="both")
        .to_numpy()
    )
    return y_signal


def convert_pixels_to_mv(y_signal: np.ndarray, image_height: int) -> np.ndarray:
    baseline_y = image_height // 2
    return (y_signal - baseline_y) * MV_PER_PIXEL


# =================================================
# Core entry point
# =================================================
def core_run(input_dir, output_dir):
    """
    Core entry point for Step 05 – waveform extraction.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for record_dir in sorted(input_dir.iterdir()):
        if not record_dir.is_dir():
            continue

        record_id = record_dir.name
        print(f"\n[STEP 05] Processing record {record_id}")

        points_dir = record_dir / "points"
        if not points_dir.exists():
            print(f"  ⚠️ Missing points directory for {record_id}")
            continue

        lead_signals = {}

        for points_path in sorted(points_dir.glob("*.npy")):
            lead_name = points_path.stem.upper()

            points = np.load(points_path)
            if points.ndim != 2 or points.shape[1] != 2:
                continue

            width = int(points[:, 0].max()) + 1
            height = int(points[:, 1].max()) + 1

            y_signal = points_to_signal_y(points, width)
            signal_mv = convert_pixels_to_mv(y_signal, height)
            signal_mv_resampled = resample(signal_mv, NUM_SAMPLES)

            lead_signals[lead_name] = signal_mv_resampled

        ordered_leads = {
            lead: lead_signals[lead]
            for lead in LEAD_ORDER
            if lead in lead_signals
        }

        if not ordered_leads:
            print(f"  ⚠️ No valid leads for {record_id}")
            continue

        df = pd.DataFrame(ordered_leads)
        output_csv = output_dir / f"{record_id}.csv"
        df.to_csv(output_csv, index=False)

        print(f"  ✅ Saved {output_csv}")

    print("\nStep 05 completed: waveform CSV files generated.")