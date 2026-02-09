# ECG Digitization from Scanned Images

This repository provides an end-to-end pipeline for digitizing 12-lead ECG
signals from scanned images or PDF documents. The pipeline converts raster ECG
images into structured digital waveforms and exports them in standard formats
for downstream analysis and interoperability.

The implementation follows the same high-level workflow described in the
accompanying paper, with modular decomposition for clarity, maintainability,
and reproducibility.

---

## Pipeline Overview

The ECG digitization process is organized into six main steps. While the paper
describes these steps conceptually, the codebase decomposes some steps into
multiple sub-modules for practical implementation.

<details>
<summary><b>🔍 Pipeline ↔ Code Mapping (click to expand)</b></summary>

<br>

| Paper Step | Description | Code File(s) |
|-----------|------------|--------------|
| Step 1 | Grid detection and DPI estimation | `wrapper/run_step01_grid_detection.py` |
| Step 2 | Lead detection and automatic cropping (YOLO-based) | `wrapper/run_step02_lead_detection_region.py`, `wrapper/run_step02_lead_detection_icon.py`, `wrapper/run_step02_auto_crop.py` |
| Step 3 | Signal cleanup and binary mask generation | `wrapper/run_step03_signal_cleanup_and_mask.py` |
| Step 4 | Signal path extraction (Viterbi / Centerline) | `wrapper/run_step04_viterbi.py`, `wrapper/run_step04_centerline_proposed.py` |
| Step 5 | Waveform reconstruction | `wrapper/run_step05_waveform_extraction.py` |
| Step 6 | WFDB export and ECG image rendering | `wrapper/run_step06_wfdb_export.py`, `wrapper/run_step06_wfdb_print.py` |
| Evaluation | Quantitative and visual evaluation scripts | `evaluate/evaluate_centerline_proposed_vs_mask.py`, `evaluate/evaluate_lead_detection_icon.py`, `evaluate/evaluate_lead_detection_region.py`, `evaluate/evaluate_viterbi_vs_mask.py` |

</details>

---

## Step-by-step Description

**Step 1 – Grid detection and DPI estimation**  
Detects the background grid to estimate the physical resolution (DPI)
for spatial normalization.

**Step 2 – Lead detection and auto-cropping**  
Localizes ECG leads using YOLO-based detectors and automatically crops
individual lead images.

**Step 3 – Signal cleanup and mask generation**  
Extracts clean binary masks from each cropped lead image using adaptive
preprocessing techniques.

**Step 4 – Signal path extraction**  
Infers the waveform trajectory using either:
- Viterbi-based dynamic programming (baseline)
- Centerline-based geometric tracing (proposed)

**Step 5 – Waveform reconstruction**  
Converts extracted signal paths into numerical waveforms (CSV format).

**Step 6 – WFDB export and ECG image rendering**  
Exports waveforms to WFDB-compatible files and optionally generates printable
ECG images.

**Evaluation**  
Compares outputs to reference data or between methods:
- Lead detection accuracy
- Signal reconstruction accuracy
- Centerline vs Viterbi vs mask comparison

---

## Repository Structure

```
├─ wrapper/
│  ├─ run_step01_grid_detection.py
│  ├─ run_step02_lead_detection_region.py
│  ├─ run_step02_lead_detection_icon.py
│  ├─ run_step02_auto_crop.py
│  ├─ run_step03_signal_cleanup_and_mask.py
│  ├─ run_step04_viterbi.py
│  ├─ run_step04_centerline_proposed.py
│  ├─ run_step05_waveform_extraction.py
│  ├─ run_step06_wfdb_export.py
│  └─ run_step06_wfdb_print.py
├─ evaluate/
│  ├─ evaluate_centerline_proposed_vs_mask.py
│  ├─ evaluate_lead_detection_icon.py
│  ├─ evaluate_lead_detection_region.py
│  └─ evaluate_viterbi_vs_mask.py
├─ data/
│  └─ example_inputs/
├─ results/
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

---

## Quick Start (Wrapper-based)

To run the **full ECG digitization pipeline** with default configuration:

```bash
python wrapper/run_step01_grid_detection.py --input data/example_inputs --output results/step01_grid_detection_and_dpi

python wrapper/run_step02_lead_detection_region.py --input results/step01_grid_detection_and_dpi --output results/step02_region
python wrapper/run_step02_lead_detection_icon.py --input results/step01_grid_detection_and_dpi --output results/step02_icon
python wrapper/run_step02_auto_crop.py --input results/step01_grid_detection_and_dpi --output results/step02_auto_crop

python wrapper/run_step03_signal_cleanup_and_mask.py --input results/step02_auto_crop/segments_test --output results/step03

python wrapper/run_step04_viterbi.py --input results/step02_auto_crop/segments_test --output results/step04_digitized_viterbi
python wrapper/run_step04_centerline_proposed.py --input results/step02_auto_crop/segments_test --output results/step04_centerline_proposed

python wrapper/run_step05_waveform_extraction.py --input results/step04_centerline_proposed --output results/step05_waveform_extraction

python wrapper/run_step06_wfdb_export.py --input results/step05_waveform_extraction --output results/step06_wfdb_export
python wrapper/run_step06_wfdb_print.py --input results/step06_wfdb_export --output results/step06
```

---

## Evaluation

After running the full pipeline, you can run **quantitative and visual evaluation** scripts:

```bash
python evaluate/evaluate_centerline_proposed_vs_mask.py 

python evaluate/evaluate_lead_detection_icon.py 

python evaluate/evaluate_lead_detection_region.py 

python evaluate/evaluate_viterbi_vs_mask.py -
```

**Outputs** include:
- Accuracy tables (CSV/Excel)
- Visual overlays for qualitative inspection
- Quantitative metrics comparing methods

---

## Citation

```bibtex
@article{ecg_digitization_2026,
  title   = {Automated 12-lead ECG digitization and WFDB standardization: insights from Vietnamese clinical data},
  author  = {Nho Thai Nguyen,  Thang Van Doan, Dung Ngoc Nguyen and Van Hai Ho},
  journal = {Journal},
  year    = {2026}
}
```

---

## License

See the LICENSE file for details.

---

## Authorship & Attribution

The implementation and software engineering of this repository were carried out
by Nho Thai Nguyen.

Please cite the associated research paper when using this code for academic or
research purposes.

---

## Notes for Reviewers

Step 01 is implemented as a **Python reference module** for transparency.  
Compiled binary cores are intentionally introduced **only in later stages** to protect intellectual property.

