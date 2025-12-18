"""
Utility functions for lesion tracking.
"""

import os
from typing import Tuple, Optional
import numpy as np
import nibabel as nib


def load_nifti(filepath: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load a NIfTI image file.
    
    Args:
        filepath: Path to the NIfTI file (.nii or .nii.gz)
        
    Returns:
        Tuple of (image data as numpy array, NIfTI image object)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img


def save_nifti(
    data: np.ndarray, 
    reference_img: nib.Nifti1Image, 
    filepath: str,
    dtype: Optional[np.dtype] = None
) -> None:
    """
    Save a numpy array as a NIfTI image.
    
    Args:
        data: Image data to save
        reference_img: Reference NIfTI image for affine and header
        filepath: Output file path
        dtype: Optional data type for the output
    """
    if dtype is not None:
        data = data.astype(dtype)
    
    new_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(new_img, filepath)


def get_voxel_volume(img: nib.Nifti1Image) -> float:
    """
    Calculate the volume of a single voxel in mm³.
    
    Args:
        img: NIfTI image object
        
    Returns:
        Voxel volume in mm³
    """
    voxel_dims = img.header.get_zooms()[:3]
    return float(np.prod(voxel_dims))


def binarize_mask(
    mask: np.ndarray, 
    threshold: float = 0.5
) -> np.ndarray:
    """
    Binarize a mask array.
    
    Args:
        mask: Input mask array
        threshold: Threshold value for binarization
        
    Returns:
        Binary mask (0 or 1)
    """
    return (mask > threshold).astype(np.uint8)


def normalize_intensity(
    image: np.ndarray, 
    percentile_low: float = 1.0,
    percentile_high: float = 99.0
) -> np.ndarray:
    """
    Normalize image intensity using percentile-based scaling.
    
    Args:
        image: Input image array
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping
        
    Returns:
        Normalized image with values in [0, 1]
    """
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    
    image_clipped = np.clip(image, p_low, p_high)
    image_normalized = (image_clipped - p_low) / (p_high - p_low + 1e-8)
    
    return image_normalized


def compute_dice_coefficient(
    mask1: np.ndarray, 
    mask2: np.ndarray
) -> float:
    """
    Compute Dice similarity coefficient between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dice coefficient (0 to 1)
    """
    mask1_binary = binarize_mask(mask1)
    mask2_binary = binarize_mask(mask2)
    
    intersection = np.sum(mask1_binary & mask2_binary)
    total = np.sum(mask1_binary) + np.sum(mask2_binary)
    
    if total == 0:
        return 1.0  # Both masks are empty
    
    return 2.0 * intersection / total


def compute_overlap_ratio(
    mask1: np.ndarray, 
    mask2: np.ndarray
) -> float:
    """
    Compute the overlap ratio between two binary masks.
    Overlap is defined as intersection / min(volume1, volume2).
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Overlap ratio (0 to 1)
    """
    mask1_binary = binarize_mask(mask1)
    mask2_binary = binarize_mask(mask2)
    
    intersection = np.sum(mask1_binary & mask2_binary)
    min_volume = min(np.sum(mask1_binary), np.sum(mask2_binary))
    
    if min_volume == 0:
        return 0.0
    
    return intersection / min_volume


def get_bounding_box(
    mask: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Get the bounding box of a binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    """
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        return ((0, 0), (0, 0), (0, 0))
    
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def get_centroid(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the centroid of a binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of (x, y, z) centroid coordinates
    """
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        return (0.0, 0.0, 0.0)
    
    centroid_x = np.mean(coords[0])
    centroid_y = np.mean(coords[1])
    centroid_z = np.mean(coords[2])
    
    return (centroid_x, centroid_y, centroid_z)


def compute_distance(
    point1: Tuple[float, float, float],
    point2: Tuple[float, float, float],
    voxel_spacing: Optional[Tuple[float, float, float]] = None
) -> float:
    """Compute Euclidean distance between two points in mm."""
    spacing = voxel_spacing or (1.0, 1.0, 1.0)
    diff = [(point1[i] - point2[i]) * spacing[i] for i in range(3)]
    return float(np.sqrt(sum(d**2 for d in diff)))


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    # Simple test cases for utility functions
    mask1 = "/mnt/data/MSLesSeg Dataset/train/P2/T1/P2_T1_MASK.nii.gz"
    mask2 = "/mnt/data/MSLesSeg Dataset/train/P2/T2/P2_T2_MASK.nii.gz"
    data1, img1 = load_nifti(mask1)
    data2, img2 = load_nifti(mask2)
    print("Loaded NIfTI shape:", data1.shape)
    print("Loaded NIfTI shape:", data2.shape)
    print("Voxel volume (mm³):", get_voxel_volume(img1))
    print("Voxel volume (mm³):", get_voxel_volume(img2))
    print("Binarized mask unique values:", np.unique(binarize_mask(data1)))
    print("Dice coefficient (self):", compute_dice_coefficient(data1, data2))
    print("Overlap ratio (self):", compute_overlap_ratio(data1, data2))
