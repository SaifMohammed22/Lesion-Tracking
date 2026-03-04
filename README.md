## Lesion Tracking Between Two Timepoints

### Overview

This repository provides a lightweight pipeline to **track MS lesions** between a baseline and a follow-up scan.  
Given binary lesion masks at both timepoints (registered or unregistered), it:

- Registers follow-up to baseline (via ANTsPy)
- Labels individual lesions
- Matches lesions across time
- Classifies each lesion as **Present**, **Enlarged**, **Shrinking**, **Absent**, **Merged**, **Split**, or **New**
- Saves labeled NIfTI volumes and CSV/JSON summaries

### Installation

```bash
conda env create --file environment.yml
conda activate lesion_env
```

Make sure **ANTsPy** (`antspyx`) is installed in this environment for registration:

```bash
pip install antspyx
```

### Quick start (high‑level pipeline)

```python
from lesion_tracker import run_tracking

results = run_tracking(
    baseline_flair="MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz",
    baseline_mask="MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz",
    followup_flair="MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz",
    followup_mask="MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz",
    output_dir="./output/P1_T1_T2",      # where reports and NIfTIs are saved
    registration_type="Affine",          # "Rigid", "Affine", or "SyN"
    min_lesion_size=7,                   # voxels
    change_threshold=0.25,               # volume change ratio
)

print(results["summary"])
```

You can also run the example directly from the project root:

```bash
python lesion_tracker/main.py
```

### Using only lesion matching

```python
import numpy as np
from lesion_tracker import track_lesions, label_lesions

# baseline_mask and followup_mask are 3D binary NumPy arrays in the same space

results = track_lesions(
    baseline_mask=baseline_mask,
    followup_mask=followup_mask,
    min_lesion_size=5,
    change_threshold=0.25,
    max_distance_mm=20.0,
    voxel_spacing=(1.0, 1.0, 1.0),
)

print(results["summary"])
lesions = results["lesions"]  # list of per‑lesion records
```

### Project structure (core modules)

```text
lesion_tracker/
├── __init__.py       # Public API (run_tracking, track_lesions, register_to_baseline, etc.)
├── main.py           # High‑level pipeline and example CLI entry point
├── registration.py   # ANTs‑based image registration utilities
├── lesion_ops.py     # Lesion labeling, matching, and classification logic
├── reporting.py      # Printing, CSV/JSON/TXT reports, and NIfTI saving
└── utils.py          # NIfTI I/O and Dice utilities
```

### Data used for testing

The examples and tests in this repository use the **MSLesSeg** multiple‑sclerosis lesion segmentation dataset (MICCAI challenge data) in a local folder layout such as:

```text
MSLesSeg Dataset/
└── train/
    ├── P1/
    │   ├── T1/
    │   │   ├── P1_T1_FLAIR.nii.gz
    │   │   └── P1_T1_MASK.nii.gz
    │   └── T2/
    │       ├── P1_T2_FLAIR.nii.gz
    │       └── P1_T2_MASK.nii.gz
    └── P5/
        └── ...
```

Paths in the examples (e.g. `MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz`) assume you have downloaded the MSLesSeg data and organized it in a similar way. You can adapt the paths to your own dataset as long as you provide:

- A baseline FLAIR image and binary lesion mask
- A follow‑up FLAIR image and binary lesion mask

Dataset is available at https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209?file=53623946

### Requirements (main runtime)

- Python ≥ 3.9
- `numpy`
- `scipy`
- `nibabel`
- `pandas`
- `antspyx` (for registration; optional if you supply pre‑registered masks)

