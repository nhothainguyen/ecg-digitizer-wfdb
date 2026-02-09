#!/usr/bin/env python3
"""
Wrapper runner for Step 04 — Centerline (Proposed)

⚠️ PURE WRAPPER — NO algorithmic parameters exposed
⚠️ MUST preserve original pipeline structure
⚠️ MUST only call core_binary.step04_digitized_centerline_proposed.core_run

USAGE (RECOMMENDED):
    python wrapper/run_step04_centerline_proposed.py \
        --input outputs \
        --output results/step04
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. Ensure project root is in PYTHONPATH (FIRST)
# ============================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 2. Import ONLY from core_binary
# ============================================================
from core_binary.step04_digitized_centerline_proposed import core_run

# ============================================================
# 3. Hash utilities
# ============================================================
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_compiled_core(core_dir: Path) -> Path | None:
    for ext in (".so", ".pyd"):
        matches = list(core_dir.glob(f"_step04_core*{ext}"))
        if matches:
            return matches[0]
    return None


# ============================================================
# 4. CLI (NO algorithmic args)
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 04 — Centerline (Proposed) [Wrapper]"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Pipeline output root (MUST be: outputs/)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory for Step 04 results"
    )
    return parser.parse_args()


# ============================================================
# 5. Resolve pipeline root robustly
# ============================================================
def resolve_pipeline_root(user_input: Path) -> Path:
    """
    Accepts:
      - outputs/
      - outputs/step03_signal_cleanup_and_mask
      - outputs/step02_lead_auto_crop
      - outputs/step02_lead_auto_crop/segments_test

    Returns:
      - outputs/
    """
    p = user_input.resolve()

    # Case 1: user already passed outputs/
    if (p / "step02_lead_auto_crop").exists() and \
       (p / "step03_signal_cleanup_and_mask").exists():
        return p

    # Case 2: .../outputs/step02_lead_auto_crop/segments_test
    if p.name == "segments_test":
        return p.parents[1]   # ✅ outputs/

    # Case 3: .../outputs/step02_lead_auto_crop
    if p.name == "step02_lead_auto_crop":
        return p.parent       # ✅ outputs/

    # Case 4: .../outputs/step03_signal_cleanup_and_mask
    if p.name == "step03_signal_cleanup_and_mask":
        return p.parent       # ✅ outputs/

    raise FileNotFoundError(
        f"Cannot resolve pipeline root from input: {user_input}\n"
        f"Expected outputs/ or a direct step subdirectory."
    )

# ============================================================
# 6. Main
# ============================================================
def main():
    args = parse_args()

    user_input = Path(args.input)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 04 — CENTERLINE (PROPOSED)")
    print("Wrapper execution started")
    print("=" * 72)

    # --------------------------------------------------------
    # Resolve & validate pipeline structure
    # --------------------------------------------------------
    pipeline_root = resolve_pipeline_root(user_input)

    mask_dir = pipeline_root / "step03_signal_cleanup_and_mask"
    lead_dir = pipeline_root / "step02_lead_auto_crop" / "segments_test"

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask directory: {mask_dir}")

    if not lead_dir.exists():
        raise FileNotFoundError(f"Missing lead directory: {lead_dir}")

    print("\n[PIPELINE]")
    print(f"  Pipeline root : {pipeline_root}")
    print(f"  Mask dir      : {mask_dir}")
    print(f"  Lead dir      : {lead_dir}")

    # --------------------------------------------------------
    # Hash audit
    # --------------------------------------------------------
    core_dir = PROJECT_ROOT / "core_binary"
    py_core = core_dir / "step04_digitized_centerline_proposed.py"
    bin_core = find_compiled_core(core_dir)

    print("\n[HASH CHECK]")

    py_hash = sha256_of_file(py_core)
    print(f"  Python core : {py_core.name}")
    print(f"  SHA256      : {py_hash}")

    if bin_core:
        bin_hash = sha256_of_file(bin_core)
        print(f"  Binary core : {bin_core.name}")
        print(f"  SHA256      : {bin_hash}")
    else:
        bin_hash = None
        print("  ⚠️ Compiled core binary NOT FOUND")

    # --------------------------------------------------------
    # Input stats
    # --------------------------------------------------------
    num_masks = sum(1 for _ in mask_dir.rglob("*") if _.is_file())

    print("\n[INPUT SUMMARY]")
    print(f"  Total mask files : {num_masks}")

    # --------------------------------------------------------
    # Run core
    # --------------------------------------------------------
    print("\n[RUN]")
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Start time : {start_dt}")

    core_run(
        input_mask_dir=mask_dir,
        input_lead_dir=lead_dir,
        output_dir=output_dir
    )

    end_time = time.time()
    elapsed = end_time - start_time
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n[SUMMARY]")
    print(f"  End time      : {end_dt}")
    print(f"  Runtime       : {elapsed:.2f} seconds")
    print(f"  Output dir    : {output_dir}")

    print("\n[REPRODUCIBILITY]")
    print(f"  Core py hash  : {py_hash}")
    if bin_hash:
        print(f"  Core bin hash : {bin_hash}")
    else:
        print("  Core bin hash : N/A")

    print("\nDONE.")
    print("=" * 72)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()