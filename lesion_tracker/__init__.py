"""
Lesion Tracker - Simple MS Lesion Tracking Between Timepoints

Track lesions between baseline and follow-up MRI scans to detect:
- New lesions
- Disappeared lesions
- Enlarged lesions
- Shrunk lesions
- Stable lesions
"""

from .main import (
    # Main functions
    run_tracking,
)

# Labeling and Tracking operation
from .lesion_ops import label_lesions, track_lesions

# Registration Functionality
from .registration import register_to_baseline, apply_transform

# I/O utilities
from .utils import load_nifti, save_nifti, dice_score, lesion_dice_score

# Reporting and Saving
from .reporting import print_summary, save_results

# Optional visualization
try:
    from .visualization import visualize_tracking
except ImportError:
    visualize_tracking = None

__version__ = "2.0.0"

__all__ = [
    # Main API
    "run_tracking",
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
    # Reporting and Saving
    "print_summary",
    "save_results"
]
