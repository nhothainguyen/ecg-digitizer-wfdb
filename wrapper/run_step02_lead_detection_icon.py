# ============================================================
# File: wrapper/run_step02_icon.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 02 (ICON) – OFFICIAL WRAPPER SCRIPT
============================================================

This wrapper script is the ONLY official entry point for
Step 02 – Lead icon / label detection.

HOW TO RUN:
    python wrapper/run_step02_lead_detection_icon.py --input <input_dir> --output <output_dir>
EX:
    python wrapper/run_step02_lead_detection_icon.py --input outputs/step01_grid_detection_and_dpi  --output results/step02_icon
ARGUMENTS:
    --input   Directory produced by the previous pipeline step
    --output  Directory where this step's results will be written

IMPORTANT NOTES FOR REVIEWERS:
- Do NOT run any files under core_binary/ directly.
- Do NOT modify any code during review.
- All algorithmic parameters are hard-coded for reproducibility.
- This wrapper provides a unified execution interface consistent
  with all pipeline steps.

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
from core_binary.step02_lead_detection_icon_yolo import core_run


def _count_valid_inputs(input_dir: Path):
    valid_ext = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Step 02: Lead icon / label detection (YOLO)"
    )
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    start_dt = datetime.utcnow()

    print("=" * 60)
    print("🚀 STEP 02 – LEAD ICON / LABEL DETECTION (YOLO)")
    print("=" * 60)
    print(f"🕒 Start time (UTC): {start_dt.isoformat()}Z")
    print(f"📂 Input dir : {input_dir}")
    print(f"📁 Output dir: {output_dir}")

    n_inputs = _count_valid_inputs(input_dir)
    print(f"📊 Valid input files: {n_inputs}")

    if n_inputs == 0:
        print("⚠️  No valid input files found. Exiting.")
        sys.exit(0)

    print("\n▶ Running core...")
    core_run(str(input_dir), str(output_dir))

    end_time = time.time()
    end_dt = datetime.utcnow()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print("✅ STEP 02 (ICON) COMPLETED")
    print("=" * 60)
    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {elapsed:.2f} seconds")
    print(f"📄 Processed files: {n_inputs}")
    print("=" * 60)


if __name__ == "__main__":
    main()