"""
Registration utilities for lesion tracking.

This module wraps ANTsPy functionality and is intentionally small, so it can be
mocked or replaced more easily when testing.
"""

from typing import Dict, Any, List

import numpy as np

try:
    import ants
except ImportError:  # pragma: no cover - soft dependency
    ants = None


def register_to_baseline(
    baseline_path: str, followup_path: str, transform_type: str = "SyN"
) -> Dict[str, Any]:
    """
    Register follow-up image to baseline space using ANTs.

    Returns dict with:
        - ``transforms``: list of forward transform paths
        - ``registered``: warped moving ANTs image
        - ``fixed``: fixed ANTs image (baseline)
    """
    if ants is None:
        raise ImportError(
            "ANTsPy is required for registration. Install with: pip install antspyx"
        )

    fixed = ants.image_read(baseline_path)
    moving = ants.image_read(followup_path)
    result = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform_type)

    return {
        "transforms": result["fwdtransforms"],
        "registered": result["warpedmovout"],
        "fixed": fixed,
    }


def apply_transform(
    image_path: str,
    reference: "ants.ANTsImage",
    transforms: List[str],
    interpolation: str = "nearestNeighbor",
) -> np.ndarray:
    """Apply transforms to an image (e.g., lesion mask) and return a NumPy array."""
    if ants is None:
        raise ImportError(
            "ANTsPy is required for applying transforms. Install with: pip install antspyx"
        )

    image = ants.image_read(image_path)
    transformed = ants.apply_transforms(
        fixed=reference,
        moving=image,
        transformlist=transforms,
        interpolator=interpolation,
    )
    return transformed.numpy()

