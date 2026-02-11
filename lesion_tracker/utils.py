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
