"""
Utility functions for NIfTI I/O.
"""

import numpy as np
import nibabel as nib
from typing import Tuple


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI file, return (data, image object)."""
    img = nib.load(path)
    return img.get_fdata(), img


def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, path: str):
    """Save array as NIfTI using reference image for affine/header."""
    nib.save(
        nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header),
        path,
    )


def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice similarity coefficient between two binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        Dice score (0 to 1)
    """
    mask1_binary = (mask1 > 0).astype(np.int32)
    mask2_binary = (mask2 > 0).astype(np.int32)

    intersection = np.sum(mask1_binary & mask2_binary)
    total = np.sum(mask1_binary) + np.sum(mask2_binary)

    if total == 0:
        return 1.0  # Both empty = perfect match

    return (2.0 * intersection) / total


def lesion_dice_score(labeled: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Calculate per-lesion and overall Dice scores.

    Args:
        labeled: Labeled lesion array (output from tracking)
        ground_truth: Binary or labeled ground truth mask

    Returns:
        Dictionary with overall dice and per-lesion dice scores
    """
    gt_binary = (ground_truth > 0).astype(np.int32)
    labeled_binary = (labeled > 0).astype(np.int32)

    overall_dice = dice_score(labeled_binary, gt_binary)

    per_lesion_dice = {}
    unique_labels = np.unique(labeled[labeled > 0])

    for label in unique_labels:
        les_mask = (labeled == label).astype(np.int32)
        per_lesion_dice[int(label)] = dice_score(les_mask, gt_binary)

    return {
        "overall_dice": overall_dice,
        "per_lesion_dice": per_lesion_dice,
        "num_labeled_lesions": len(unique_labels),
        "num_gt_lesions": len(np.unique(ground_truth[ground_truth > 0])),
    }


if __name__ == "__main__":
    bl_path = "MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz"
    fu_path = "MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz"

    f_data_bl, img_bl = load_nifti(bl_path)
    f_data_fu, img_fu = load_nifti(fu_path)

    print(f_data_bl.shape)
    print(img_bl)
