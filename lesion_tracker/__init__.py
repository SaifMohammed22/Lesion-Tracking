"""
Lesion Tracker - Simple MS Lesion Tracking Between Timepoints

Track lesions between baseline and follow-up MRI scans to detect:
- New lesions
- Disappeared lesions
- Enlarged lesions
- Shrunk lesions
- Stable lesions
"""

from .core import (
    # Main functions
    run_tracking,
    track,
    # Core functions (if needed separately)
    track_lesions,
    label_lesions,
    # Registration
    register_to_baseline,
    apply_transform,
)

# I/O utilities
from .utils import load_nifti, save_nifti, dice_score, lesion_dice_score

# Optional visualization
try:
    from .visualization import visualize_tracking
except ImportError:
    visualize_tracking = None

__version__ = "2.0.0"

__all__ = [
    # Main API
    "run_tracking",
    "track",
    # Core functions
    "track_lesions",
    "label_lesions",
    # I/O
    "load_nifti",
    "save_nifti",
    "dice_score",
    "lesion_dice_score",
    # Registration
    "register_to_baseline",
    "apply_transform",
    # Visualization
    "visualize_tracking",
]
