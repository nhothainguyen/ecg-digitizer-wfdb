"""
[CORE_BINARY]

STEP 06 — WFDB ECG IMAGE GENERATION (PRINT)

⚠️ This file belongs to core_binary
⚠️ DO NOT MODIFY during review
⚠️ No algorithmic logic is implemented here
⚠️ This file ONLY orchestrates a frozen WFDB renderer

Purpose:
- Public reproducibility
- Parameter freezing
- IP protection (external renderer)
"""

import os
import sys
import random
import warnings
from pathlib import Path

# =================================================
# Reproducibility (HARD)
# =================================================
random.seed(0)

# =================================================
# Resolve repo root & PYTHONPATH
# =================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =================================================
# Fixed I/O (HARD-CODED)
# =================================================
INPUT_DIR = REPO_ROOT / "outputs" / "step06_wfdb_export"
OUTPUT_DIR = REPO_ROOT / "outputs" / "step06_wfdb_print"

# =================================================
# Environment
# =================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# =================================================
# External (ALGORITHMIC) Imports
# =================================================
from public.wrapper.print_ecg_wfdb.helper_functions import find_records
from public.wrapper.print_ecg_wfdb.gen_ecg_image_from_data import run_single_file

# =================================================
# Frozen argument object
# =================================================
class _FrozenArgs:
    class _FrozenArgs:
        def __init__(self):
            # ---- core ----
            self.seed = 0
            self.config_file = "config.yaml"
            self.num_leads = "twelve"
            self.max_num_images = -1

            # ---- layout ----
            self.resolution = 200
            self.pad_inches = 0
            self.num_columns = -1
            self.full_mode = "II"

            # ---- rendering ----
            self.print_header = True
            self.mask_unplotted_samples = False
            self.add_qr_code = True

            # ---- text / handwriting ----
            self.link = ""
            self.num_words = 5
            self.x_offset = 30
            self.y_offset = 30
            self.handwriting_size_factor = 0.2

            # ---- wrinkles ----
            self.crease_angle = 90
            self.num_creases_vertically = 10
            self.num_creases_horizontally = 10

            # ---- noise / distortions ----
            self.rotate = 0
            self.noise = 50
            self.crop = 0.01
            self.temperature = 40000

            # ---- random flags ----
            self.random_resolution = False
            self.random_padding = False
            self.random_grid_color = False
            self.standard_grid_color = 5
            self.calibration_pulse = 0.8
            self.random_grid_present = 0.7
            self.random_print_header = 0
            self.random_bw = 0

            # ---- lead options ----
            self.remove_lead_names = True
            self.lead_name_bbox = False
            self.lead_bbox = False

            # ---- deterministic ----
            self.deterministic_offset = False
            self.deterministic_num_words = False
            self.deterministic_hw_size = False
            self.deterministic_angle = False
            self.deterministic_vertical = False
            self.deterministic_horizontal = False
            self.deterministic_rot = False
            self.deterministic_noise = False
            self.deterministic_crop = False
            self.deterministic_temp = False

            # ---- misc ----
            self.fully_random = False
            self.hw_text = False
            self.wrinkles = False
            self.augment = False
            self.store_config = 0


# =================================================
# PUBLIC ENTRY POINT
# =================================================
def core_run(input_dir: Path | None = None, output_dir: Path | None = None):
    """
    ONLY public API for Step 06.
    input_dir / output_dir are ignored (fixed by design).
    """
    args = _FrozenArgs()
    assert hasattr(args, "config_file"), "FrozenArgs missing config_file"
    args = _FrozenArgs()

    if not INPUT_DIR.is_dir():
        raise RuntimeError(f"Missing input directory: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header_files, recording_files = find_records(
        str(INPUT_DIR),
        str(OUTPUT_DIR)
    )

    if not header_files:
        raise RuntimeError("No WFDB records found.")

    total_generated = 0

    for header, recording in zip(header_files, recording_files):
        args.input_file = str(INPUT_DIR / recording)
        args.header_file = str(INPUT_DIR / header)
        args.start_index = -1

        subfolders = Path(header).parent
        args.output_directory = str(OUTPUT_DIR / subfolders)
        args.encoding = Path(recording).suffix

        generated = run_single_file(args)
        total_generated += int(generated or 0)

        if args.max_num_images != -1 and total_generated >= args.max_num_images:
            break

    return total_generated