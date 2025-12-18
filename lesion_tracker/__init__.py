"""
MRI Lesion Tracking Tool

A comprehensive tool for tracking and labeling lesions in baseline and follow-up MRI scans.
Designed for the MSLesSeg dataset and similar MS lesion datasets.
"""

from .tracker import LesionTracker
from .labeler import LesionLabeler
from .metrics import LesionMetrics
from .visualization import LesionVisualizer

__version__ = "0.1.0"
__author__ = "Lesion Tracking Team"

__all__ = [
    "LesionTracker",
    "LesionLabeler",
    "LesionMetrics",
    "LesionVisualizer",
]
