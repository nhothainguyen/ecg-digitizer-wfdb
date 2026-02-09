import subprocess
import sys
from pathlib import Path

def main():
    script = Path("pipeline/step06_wfdb_print.py")

    cmd = [
        sys.executable,
        str(script),
        "--calibration_pulse", "0.8",
        "--standard_grid_color", "5",
        "--add_qr_code",
    ]

    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()