# ============================================================
# File: wrapper/run_step02_auto_crop.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 02 – OFFICIAL WRAPPER SCRIPT (AUTO CROP)
============================================================

This wrapper script is the ONLY official entry point for this step.

HOW TO RUN:
    python wrapper/run_step02_auto_crop.py --input <input_dir> --output <output_dir>
EX:
    python wrapper/run_step02_auto_crop.py --input outputs/step01_grid_detection_and_dpi --output results/step02_auto_crop
INPUT:
    Directory containing images from the previous pipeline step.

OUTPUT:
    Directory where cropped leads, labels, and detection images
    will be written.

IMPORTANT:
- Do NOT run any files under core_binary/ directly.
- Do NOT modify any code during review.
============================================================
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core_binary.step02_lead_auto_crop import core_run


def _count_valid_inputs(input_dir: Path):
    valid_ext = {".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Step 02: Lead auto-cropping"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    start_dt = datetime.utcnow()

    print(f"🕒 Start time (UTC): {start_dt.isoformat()}Z")
    print(f"📂 Input dir : {input_dir}")
    print(f"📁 Output dir: {output_dir}")

    n_inputs = _count_valid_inputs(input_dir)
    print(f"📊 Valid input files: {n_inputs}")

    if n_inputs == 0:
        sys.exit(0)

    core_run(str(input_dir), str(output_dir))

    end_time = time.time()
    end_dt = datetime.utcnow()

    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {end_time - start_time:.2f} seconds")
    print(f"📄 Processed files: {n_inputs}")


if __name__ == "__main__":
    main()