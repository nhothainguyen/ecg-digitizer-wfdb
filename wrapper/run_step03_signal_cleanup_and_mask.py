# ============================================================
# File: wrapper/run_step03.py
# ============================================================
#!/usr/bin/env python3
"""
============================================================
STEP 03 – OFFICIAL WRAPPER SCRIPT
Signal cleanup and binary mask generation
============================================================

This wrapper script is the ONLY official entry point for this step.

HOW TO RUN (from project root):
    python wrapper/run_step03_signal_cleanup_and_mask.py --input <input_dir> --output <output_dir>
EX: 
    python wrapper/run_step03_signal_cleanup_and_mask.py --input outputs/step02_lead_auto_crop/segments_test --output results/step03
INPUT:
    Directory containing record folders with cropped lead images
    (output of Step 02).

OUTPUT:
    Root directory where Step 03 outputs will be created:
      - step03_signal_cleanup_and_mask/
      - step03_centerline_reference_mask/

IMPORTANT:
- Do NOT run any files under core_binary/ directly.
- Do NOT modify any code during review.
- No algorithmic parameters are exposed.
============================================================
"""

import argparse
import sys
import time
import hashlib
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
from core_binary.step03_signal_cleanup_and_mask import core_run


# ============================================================
# UTILS
# ============================================================
def _sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_valid_inputs(input_dir: Path):
    """
    Count valid input records (directories containing at least one image).
    """
    valid = 0
    for d in input_dir.iterdir():
        if not d.is_dir():
            continue
        imgs = [
            f for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if imgs:
            valid += 1
    return valid


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run Step 03: Signal cleanup and mask generation"
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

    # --------------------------------------------------------
    # HASH CHECKS (REPRODUCIBILITY GUARD)
    # --------------------------------------------------------
    core_py = PROJECT_ROOT / "core_binary" / "step03_signal_cleanup_and_mask.py"

    # detect compiled core (.pyd on Windows, .so on Unix)
    compiled_candidates = list(
        (PROJECT_ROOT / "core_binary").glob("_step03_core.*")
    )
    compiled_core = compiled_candidates[0] if compiled_candidates else None

    core_py_hash = _sha256(core_py)
    compiled_hash = _sha256(compiled_core) if compiled_core else None

    print("============================================================")
    print("🔐 CORE FILE HASHES")
    print("------------------------------------------------------------")
    print(f"Python core : {core_py.name}")
    print(f"  SHA256 = {core_py_hash}")
    if compiled_core:
        print(f"Binary core : {compiled_core.name}")
        print(f"  SHA256 = {compiled_hash}")
    else:
        print("⚠️  Compiled core (_step03_core.*) NOT FOUND")
    print("============================================================")

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    start_time = time.time()
    start_dt = datetime.utcnow()

    print(f"🕒 Start time (UTC): {start_dt.isoformat()}Z")
    print(f"📂 Input dir : {input_dir}")
    print(f"📁 Output dir: {output_dir}")

    n_inputs = _count_valid_inputs(input_dir)
    print(f"📊 Valid input records: {n_inputs}")

    if n_inputs == 0:
        print("⚠️ No valid input records found. Exiting.")
        sys.exit(0)

    core_run(str(input_dir), str(output_dir))

    end_time = time.time()
    end_dt = datetime.utcnow()

    print("============================================================")
    print("✅ STEP 03 COMPLETED")
    print("============================================================")
    print(f"🕒 End time (UTC): {end_dt.isoformat()}Z")
    print(f"⏱  Total runtime : {end_time - start_time:.2f} seconds")
    print(f"📄 Processed records: {n_inputs}")


if __name__ == "__main__":
    main()