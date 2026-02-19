"""
Lesion Tracking - Core Module

Track MS lesions between two timepoints (baseline and follow-up).

- ``registration``: registration utilities (ANTs-based)
- ``lesion_ops``: pure lesion-tracking and matching logic
- ``reporting``: reporting ans saving results
"""

import os
import tempfile
from typing import Dict, Any, Optional

import nibabel as nib
import numpy as np

from lesion_tracker.utils import load_nifti, save_nifti
from lesion_tracker.registration import register_to_baseline, apply_transform
from lesion_tracker.lesion_ops import track_lesions, label_lesions
from lesion_tracker.reporting import print_summary, save_results


# =============================================================================
# Main Pipeline
# =============================================================================


def run_tracking(
    baseline_flair: str,
    baseline_mask: str,
    followup_flair: str,
    followup_mask: str,
    output_dir: Optional[str] = None,
    registration_type: str = "Affine",
    min_lesion_size: int = 7,
    change_threshold: float = 0.25,
    save_visualization: bool = False,
    num_slices: int = 100,
) -> Dict[str, Any]:
    """
    Main entry point - track lesions between two timepoints.

    Args:
        baseline_flair: Path to baseline FLAIR image
        baseline_mask: Path to baseline lesion mask
        followup_flair: Path to follow-up FLAIR image
        followup_mask: Path to follow-up lesion mask
        output_dir: Optional directory to save results
        registration_type: ANTs registration type ('Rigid', 'Affine', 'SyN')
        min_lesion_size: Minimum lesion size in voxels
        change_threshold: Volume change ratio for enlarged/shrunk classification
        save_visualization: Whether to save PNG visualization (default: True)
        num_slices: Number of slices in visualization (1=single best, >1=grid)

    Returns:
        Dict with tracking results and statistics
    """
    print("=" * 60)
    print("Lesion Tracking Pipeline")
    print("=" * 60)

    # 1. Load baseline mask
    print("\n[1/4] Loading data...")
    bl_mask_data, bl_mask_img = load_nifti(baseline_mask)
    bl_mask_data = (bl_mask_data > 0).astype(np.uint8)

    # 2. Register follow-up to baseline
    print(f"[2/4] Registering follow-up to baseline ({registration_type})...")
    reg_result = register_to_baseline(baseline_flair, followup_flair, registration_type)

    # 3. Apply transform to follow-up mask
    print("[3/4] Transforming follow-up mask...")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_mask_path = tmp.name

    try:
        # Save follow-up mask temporarily for ANTs
        fu_mask_data, _ = load_nifti(followup_mask)
        nib.save(
            nib.Nifti1Image(fu_mask_data, nib.load(followup_flair).affine),
            tmp_mask_path,
        )

        # Apply registration transform
        fu_mask_registered = apply_transform(
            tmp_mask_path,
            reg_result["fixed"],
            reg_result["transforms"],
            interpolation="nearestNeighbor",
        )
        fu_mask_registered = (fu_mask_registered > 0.5).astype(np.uint8)
    finally:
        if os.path.exists(tmp_mask_path):
            os.remove(tmp_mask_path)

    # 4. Track lesions
    print("[4/4] Tracking lesions...")
    voxel_spacing = bl_mask_img.header.get_zooms()[:3]
    results = track_lesions(
        bl_mask_data,
        fu_mask_registered,
        min_lesion_size=min_lesion_size,
        change_threshold=change_threshold,
        max_distance_mm=20.0,
        voxel_spacing=voxel_spacing,
    )

    # Add voxel volume for mm3 calculations
    voxel_vol = np.prod(voxel_spacing)
    results["voxel_volume_mm3"] = float(voxel_vol)

    # Print summary
    print_summary(results)

    # Save results if output_dir specified
    if output_dir:
        save_results(
            results,
            bl_mask_img,
            fu_mask_registered,
            output_dir,
            baseline_flair,
            followup_flair,
            save_visualization,
            num_slices,
            baseline_mask,
            followup_mask,
        )

    return results


if __name__ == "__main__":
    # Example usage
    results = run_tracking(
        baseline_flair="MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz",
        baseline_mask="MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz",
        followup_flair="MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz",
        followup_mask="MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz",
        output_dir="./output/P1_T1_T2",
    )