# ============================================================
# File: wrapper/run_step04.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 04 – OFFICIAL WRAPPER SCRIPT (VITERBI BASELINE)
============================================================

This wrapper script is the ONLY official entry point for this step.

HOW TO RUN:
    python wrapper/run_step04_viterbi.py --input <input_dir> --output <output_dir>
EX:
    python wrapper/run_step04_viterbi.py --input outputs\step02_lead_auto_crop\segments_test --output results/step04_digitized_viterbi
INPUT:
    Directory produced by Step 02 (auto-cropped lead segments).

OUTPUT:
    Directory where Viterbi digitization results will be written.

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


# ============================================================
# ADD PROJECT ROOT TO PYTHONPATH
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT CORE (AND ONLY CORE)
# ============================================================
from core_binary.step04_digitized_viterbi import core_run


def _count_valid_inputs(input_dir: Path):
    return sum(1 for d in input_dir.iterdir() if d.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description="Run Step 04: Baseline Viterbi digitization"
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
    print(f"📊 Valid input records: {n_inputs}")

    if n_inputs == 0:
        sys.exit(0)

    core_run(str(input_dir), str(output_dir))

    end_time = time.time()
    end_dt = datetime.utcnow()

    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {end_time - start_time:.2f} seconds")
    print(f"📄 Processed records: {n_inputs}")


if __name__ == "__main__":
    main()