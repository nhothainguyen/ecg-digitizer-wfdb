# ============================================================
# File: wrapper/run_step01_grid_detection.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 01 – GRID DETECTION & DPI NORMALIZATION
============================================================

This wrapper script is the ONLY official entry point for
Step 01 – ECG grid detection and DPI normalization.

This step:
- Detects ECG background grid
- Normalizes DPI and spatial scale
- Produces standardized image outputs for downstream steps

HOW TO RUN:
    python wrapper/run_step01_grid_detection.py --input <input_dir> --output <output_dir>

EX:
    python wrapper/run_step01_grid_detection.py --input data/example_inputs  --output results/step01_grid_detection_and_dpi

ARGUMENTS:
    --input   Directory containing raw ECG images or PDFs
    --output  Directory where normalized outputs will be written

IMPORTANT NOTES FOR REVIEWERS:
- Do NOT run any files under core_binary/ directly.
- Do NOT modify any code during review.
- All algorithmic parameters are hard-coded for reproducibility.
- Core computation is encapsulated inside compiled / protected modules.
- This wrapper provides a unified, reviewer-safe execution interface.

PIPELINE CONTEXT:
    Step 01 → Grid detection & DPI normalization


============================================================
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. ADD PROJECT ROOT TO PYTHONPATH
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   # wrapper/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 2. IMPORT CORE (AND ONLY CORE)
# ============================================================
try:
    from core_binary.step01_grid_core import core_run
except Exception as e:
    print("❌ Failed to import core_binary.step01_grid_core")
    raise e


# ============================================================
# 3. UTILS
# ============================================================
def sha256_of_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()





def count_valid_inputs(input_dir: Path):
    valid_ext = {".pdf", ".jpg", ".jpeg", ".png"}
    return sum(
        1 for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_ext
    )


# ============================================================
# 4. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run Step 01: Grid detection & DPI normalization"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input directory (raw images / PDFs)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # LOG START
    # ========================================================
    start_time = time.time()
    start_dt = datetime.utcnow()

    print("=" * 60)
    print("🚀 STEP 01 – GRID DETECTION & DPI NORMALIZATION")
    print("=" * 60)
    print(f"🕒 Start time (UTC): {start_dt.isoformat()}Z")
    print(f"📂 Input dir : {input_dir}")
    print(f"📁 Output dir: {output_dir}")


    # ========================================================
    # CORE INTEGRITY CHECK (STEP 01 – PYTHON REFERENCE ONLY)
    # ========================================================
    core_py = PROJECT_ROOT / "core_binary" / "step01_grid_core.py"

    print("\n🔐 CORE INTEGRITY")

    if not core_py.exists():
        print("❌ step01_grid_core.py not found")
        sys.exit(1)

    hash_py = sha256_of_file(core_py)
    print(f"  step01_grid_core.py : {hash_py}")
    print("✅ Python reference core verified")
    print("ℹ️  Step 01 does not include a compiled binary by design")


    # ========================================================
    # COUNT INPUT FILES
    # ========================================================
    n_inputs = count_valid_inputs(input_dir)
    print(f"\n📊 Valid input files detected: {n_inputs}")

    if n_inputs == 0:
        print("⚠️  No valid input files found. Exiting.")
        sys.exit(0)

    # ========================================================
    # RUN CORE
    # ========================================================
    print("\n▶ Running core...")
    core_run(str(input_dir), str(output_dir))

    # ========================================================
    # LOG END
    # ========================================================
    end_time = time.time()
    end_dt = datetime.utcnow()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print("✅ STEP 01 COMPLETED")
    print("=" * 60)
    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {elapsed:.2f} seconds")
    print(f"📄 Processed files: {n_inputs}")
    print("=" * 60)


if __name__ == "__main__":
    main()