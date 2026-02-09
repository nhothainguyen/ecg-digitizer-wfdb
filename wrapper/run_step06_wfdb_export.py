# ============================================================
# File: wrapper/run_step06.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 06 – OFFICIAL WRAPPER SCRIPT (WFDB EXPORT)
============================================================

This wrapper script is the ONLY official entry point for this step.

HOW TO RUN:
    python wrapper/run_step06_wfdb_export.py --input <input_dir> --output <output_dir>
EX:
    python wrapper/run_step06_wfdb_export.py --input outputs/step05_waveform_extraction --output results/step06_wfdb_export
INPUT:
    Directory containing CSV waveform files from Step 05.

OUTPUT:
    Directory where WFDB records (.hea, .dat, optional .atr) are written.

IMPORTANT NOTES FOR REVIEWERS:
- Do NOT run any files under core_binary/ directly.
- Do NOT modify any code during review.
- All parameters are hard-coded for reproducibility.

============================================================
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime


# ============================================================
# ADD PROJECT ROOT TO PYTHONPATH
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT CORE (AND ONLY CORE)
# ============================================================
from core_binary.step06_wfdb_export import core_run


def _count_valid_inputs(input_dir: Path):
    return len(list(input_dir.glob("*.csv")))


def main():
    parser = argparse.ArgumentParser(
        description="Run Step 06: WFDB export"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    start_dt = datetime.utcnow()

    print(f"🕒 Start time (UTC): {start_dt.isoformat()}Z")
    print(f"📂 Input dir : {input_dir}")
    print(f"📁 Output dir: {output_dir}")

    n_inputs = _count_valid_inputs(input_dir)
    print(f"📊 Valid input CSV files: {n_inputs}")

    if n_inputs == 0:
        print("⚠️ No valid CSV files found. Exiting.")
        sys.exit(0)

    core_run(str(input_dir), str(output_dir))

    end_time = time.time()
    end_dt = datetime.utcnow()

    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {end_time - start_time:.2f} seconds")
    print(f"📄 Processed CSV files: {n_inputs}")


if __name__ == "__main__":
    main()