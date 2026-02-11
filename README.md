# LST Longitudinal Pipeline - Python Implementation

Python implementation of the algorithm from:

**"Automated segmentation of changes in FLAIR-hyperintense white matter lesions in multiple sclerosis on serial magnetic resonance imaging"**

## Overview

This tool detects new, enlarging, shrinking, and disappearing MS lesions using voxel-wise statistical testing within a joint lesion map. It compares FLAIR intensity changes against a null distribution estimated from normal-appearing white matter (NAWM).

## Features

- **Tissue Segmentation**: Automatic WM/GM segmentation from T1 images
- **Lesion Filling**: Fill lesions with NAWM intensities for better registration
- **Registration**: ANTs-based registration of follow-up to baseline
- **Bias Correction**: N4 bias field correction
- **Intensity Normalization**: GM-referenced intensity normalization
- **Statistical Testing**: NAWM-based null distribution for change detection

## Installation

```bash
conda env create --file environment.yml
conda activate lesion_tracking
```

## Quick Start

```python
from lesion_tracker import run_lst_longitudinal

results = run_lst_longitudinal(
    baseline_dir="MSLesSeg Dataset/train/P1/T1",
    followup_dir="MSLesSeg Dataset/train/P1/T2",
    output_dir="./output/P1_T1_T2",
    patient_id="P1",
    baseline_tp="T1",
    followup_tp="T2"
)

print(f"New lesions: {results['statistics']['num_new_lesions']}")
print(f"Disappeared: {results['statistics']['num_disappeared_lesions']}")
```

## Detailed Usage

```python
from lesion_tracker import LSTLongitudinal

pipeline = LSTLongitudinal(
    p_threshold=0.05,                    # Statistical significance threshold
    min_lesion_size=3,                   # Minimum lesion size in voxels
    registration_type='Affine',          # 'Rigid', 'Affine', or 'SyN'
    apply_bias_correction=True,          # Apply N4 bias correction
    tissue_segmentation_method='kmeans'  # 'otsu', 'kmeans', or 'sitk'
)

results = pipeline.run_pipeline(
    t1_baseline_path="path/to/baseline_t1.nii.gz",
    flair_baseline_path="path/to/baseline_flair.nii.gz",
    lesions_baseline_path="path/to/baseline_lesions.nii.gz",
    t1_followup_path="path/to/followup_t1.nii.gz",
    flair_followup_path="path/to/followup_flair.nii.gz",
    lesions_followup_path="path/to/followup_lesions.nii.gz",
    output_dir="./output"
)
```

## Algorithm Workflow

```
Input: T1_baseline, FLAIR_baseline, T1_followup, FLAIR_followup, 
       Lesions_baseline, Lesions_followup

Step 1: Tissue Segmentation
        Extract WM, GM from T1 baseline using k-means/Otsu

Step 2: Lesion Filling
        Fill lesions in T1 images with local NAWM intensities

Step 3: Registration
        Register follow-up to baseline (Affine/Rigid/SyN)

Step 4: Joint Lesion Map
        Joint_lesions = Lesions_baseline OR Lesions_followup_registered

Step 5: NAWM Extraction
        NAWM = WM AND NOT Joint_lesions

Step 6: Preprocessing
        - N4 bias field correction on FLAIR images
        - GM-referenced intensity normalization

Step 7: Intensity Difference
        FLAIR_diff = FLAIR_followup - FLAIR_baseline

Step 8: Statistical Testing
        For each voxel in Joint_lesions:
            z_score = (diff - NAWM_mean) / NAWM_std
            p_value = 2 * (1 - Φ(|z_score|))
            
            if p_value < threshold AND diff > 0:
                → New/expanding lesion
            elif p_value < threshold AND diff < 0:
                → Shrinking/disappearing lesion
            else:
                → Stable lesion

Output: new_lesions.nii.gz, disappeared_lesions.nii.gz, 
        stable_lesions.nii.gz, statistics.json
```

## Output Files

| File | Description |
|------|-------------|
| `new_lesions.nii.gz` | Mask of new/expanding lesions |
| `disappeared_lesions.nii.gz` | Mask of shrinking/disappearing lesions |
| `stable_lesions.nii.gz` | Mask of stable lesions |
| `joint_lesions.nii.gz` | Union of baseline and follow-up lesions |
| `nawm_mask.nii.gz` | Normal-appearing white matter mask |
| `wm_mask.nii.gz` | White matter segmentation |
| `flair_difference.nii.gz` | FLAIR intensity difference map |
| `statistics.json` | Summary statistics |

## Project Structure

```
lesion_tracker/
├── __init__.py           # Package exports
├── change_detection.py   # Main LST longitudinal pipeline
├── segmentation.py       # Tissue segmentation (WM/GM)
├── preprocessing.py      # Lesion filling, bias correction, normalization
├── registration.py       # ANTs-based registration
└── utils.py              # I/O utilities
```

## Requirements

- Python >= 3.8
- nibabel
- numpy
- scipy
- scikit-learn
- SimpleITK
- ANTsPy

## Citation

If you use this tool, please cite the original paper:

```
Schmidt P, et al. "Automated segmentation of changes in FLAIR-hyperintense 
white matter lesions in multiple sclerosis on serial magnetic resonance imaging."
NeuroImage: Clinical. 2019.
```
