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

from utils import load_nifti, save_nifti
from registration import register_to_baseline, apply_transform
from lesion_ops import track_lesions, label_lesions
from reporting import print_summary, save_results


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
    # ...existing code...
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
        # ...existing code...

    Returns:
        Dict with tracking results and statistics
    """
    print("=" * 60)
    print("Lesion Tracking Pipeline")
    print("=" * 60)

    # 1. Load baseline mask
    print("\n[1/2] Loading data...")
    bl_mask_data, bl_mask_img = load_nifti(baseline_mask)
    bl_mask_data = (bl_mask_data > 0).astype(np.uint8)
    fu_mask_data, fu_mask_img = load_nifti(followup_mask)
    fu_mask_data = (fu_mask_data > 0).astype(np.uint8)

    # 2. Track lesions (skip registration)
    print("[2/2] Tracking lesions...")
    voxel_spacing = bl_mask_img.header.get_zooms()[:3]
    results = track_lesions(
        bl_mask_data,
        fu_mask_data,
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

    # Save minimal results if output_dir specified
    if output_dir:
        from lesion_tracker.reporting import convert_numpy
        import json
        output_data = {
            "summary": results["summary"],
            "lesions": results["lesions"],
            "voxel_volume_mm3": results["voxel_volume_mm3"],
        }
        with open(os.path.join(output_dir, "tracking_results.json"), "w") as f:
            json.dump(convert_numpy(output_data), f, indent=2)
        print("  - tracking_results.json")

        # Save CSV table of lesions
        import pandas as pd
        lesion_table = [
            {
                "id": l["id"],
                "status": l["status"],
                "baseline_volume": l["baseline_volume"],
                "followup_volume": l["followup_volume"],
                "change_ratio": l.get("change_ratio", None),
                "centroid": l["centroid"],
            }
            for l in results["lesions"]
        ]
        df = pd.DataFrame(lesion_table)
        df.to_csv(os.path.join(output_dir, "lesion_table.csv"), index=False)
        print("  - lesion_table.csv")

        # Save labeled baseline and follow-up arrays
        save_nifti(results["baseline_labeled"], bl_mask_img, os.path.join(output_dir, "baseline_labeled.nii.gz"))
        save_nifti(results["followup_labeled"], fu_mask_img, os.path.join(output_dir, "followup_labeled.nii.gz"))
        print("  - baseline_labeled.nii.gz")
        print("  - followup_labeled.nii.gz")

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