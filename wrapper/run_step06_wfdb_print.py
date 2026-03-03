import argparse
from pathlib import Path
import sys

# Ensure project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import core module
import core_binary.step06_wfdb_print as core_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")

    # 🔥 Override hard-coded paths safely
    core_module.INPUT_DIR = input_dir
    core_module.OUTPUT_DIR = output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 06 — WFDB PRINT (OVERRIDDEN I/O)")
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print("=" * 60)

    core_module.core_run(input_dir=input_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
