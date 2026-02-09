# ============================================================
# File: core_binary/step06_wfdb_export.py
# ============================================================
"""
[CORE_BINARY – PLAIN PYTHON]
Step 06 – WFDB export from extracted ECG waveforms

This step converts extracted ECG waveforms (CSV, Step 05) into
standard WFDB records (.hea, .dat, optional .atr).

This step does NOT contain proprietary algorithms.
DO NOT modify this file during review.
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
import re


# ==========================
# HARD-CODED PARAMETERS
# ==========================
FS = 100                     # Sampling frequency (Hz)
GAIN = 1000                  # mV → µV
BASELINE = 0
FORMAT = "16"
UNITS = "mV"
TARGET_NUM_SAMPLES = 1000    # Final WFDB length

# Optional metadata (record_id → (gender, age, diagnosis))
METADATA = {
    "0600182": ("M", 54, "Atrial Fibrillation"),
    "0602239": ("F", 67, "Normal"),
    "0605032": ("M", 45, "Bradycardia"),
    "0607458": ("F", 72, "Tachycardia"),
}


def core_run(input_dir, output_dir):
    """
    Core entry point for Step 06 – WFDB export.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found.")
        return

    for csv_file in csv_files:
        try:
            print(f"\n🔄 Processing {csv_file.name}")

            df = pd.read_csv(csv_file)

            # Keep numeric signals only
            df_numeric = df.select_dtypes(include=[np.number])

            # Clean column names (remove .jpg if exists)
            df_numeric.columns = [
                re.sub(r"\.jpg$", "", col, flags=re.IGNORECASE)
                for col in df_numeric.columns
            ]

            record_name = re.sub(r"[^a-zA-Z0-9\-]", "-", csv_file.stem)
            sig_names = df_numeric.columns.tolist()

            # mV → µV (int16)
            signals_mv = df_numeric.to_numpy(dtype=np.float32)
            signals_uv = (signals_mv * GAIN).astype(np.int16)

            num_original = signals_uv.shape[0]

            # Repeat signal to reach TARGET_NUM_SAMPLES
            full_repeats = TARGET_NUM_SAMPLES // num_original
            remainder = TARGET_NUM_SAMPLES % num_original

            signals_uv_repeated = np.tile(signals_uv, (full_repeats, 1))
            if remainder > 0:
                signals_uv_repeated = np.vstack(
                    [signals_uv_repeated, signals_uv[:remainder, :]]
                )

            # Write WFDB record
            wfdb.wrsamp(
                record_name=record_name,
                write_dir=str(output_dir),
                fs=FS,
                units=[UNITS] * len(sig_names),
                sig_name=sig_names,
                p_signal=None,
                d_signal=signals_uv_repeated,
                fmt=[FORMAT] * len(sig_names),
                adc_gain=[float(GAIN)] * len(sig_names),
                baseline=[BASELINE] * len(sig_names),
            )

            # Optional annotation
            if record_name in METADATA:
                gender, age, diagnosis = METADATA[record_name]
                symbol = f"{gender[0].upper()}{diagnosis[0].upper()}"
                middle_sample = np.array([TARGET_NUM_SAMPLES // 2])

                wfdb.wrann(
                    record_name=record_name,
                    extension="atr",
                    sample=middle_sample,
                    symbol=[symbol],
                    subtype=np.array([0]),
                    chan=np.array([0]),
                    num=np.array([0]),
                    write_dir=str(output_dir),
                )

                print(f"📝 Annotation written: {symbol}")
            else:
                print("⚠️ No metadata → annotation skipped.")

            print(f"✅ WFDB saved: {record_name}.hea / .dat / .atr")

        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")

    print("\n🎯 Step 06 completed: all WFDB records generated.")