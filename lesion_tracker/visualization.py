"""
Lesion Tracking Visualization

Output: PNG image showing lesion tracking between baseline and follow-up.
- Left column: Baseline with lesion outlines and IDs
- Right column: Follow-up with colored lesions (new=red, stable=yellow, disappeared=green)

Supports single slice or multi-slice grid view.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from typing import Optional, List

from utils import load_nifti

# Color scheme
COLORS = {
    "new": (0.91, 0.30, 0.24),  # Red
    "stable": (0.95, 0.77, 0.06),  # Yellow
    "disappeared": (0.18, 0.80, 0.44),  # Green
}


def visualize_tracking(
    flair_baseline_path: str,
    flair_followup_path: str,
    baseline_labeled_path: str,
    followup_labeled_path: str,
    slice_idx: Optional[int] = None,
    num_slices: int = 1,
    axis: int = 2,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Create lesion tracking visualization and save as PNG.

    Args:
        flair_baseline_path: Path to baseline FLAIR
        flair_followup_path: Path to follow-up FLAIR (registered)
        baseline_labeled_path: Path to labeled baseline lesions
        followup_labeled_path: Path to labeled follow-up lesions
        slice_idx: Slice to display (auto-selected if None, ignored if num_slices > 1)
        num_slices: Number of slices to show (1 = single best, >1 = grid)
        axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
        save_path: Path to save figure (e.g., "output.png")
        show: Whether to display figure
        figsize: Figure size (auto-calculated if None)

    Returns:
        matplotlib Figure
    """
    # Load data
    flair_bl, _ = load_nifti(flair_baseline_path)
    flair_fu, _ = load_nifti(flair_followup_path)
    labels_bl, _ = load_nifti(baseline_labeled_path)
    labels_fu, _ = load_nifti(followup_labeled_path)

    labels_bl = labels_bl.astype(np.int16)
    labels_fu = labels_fu.astype(np.int16)

    # Normalize FLAIR for display
    flair_bl = _normalize(flair_bl)
    flair_fu = _normalize(flair_fu)

    # Get lesion categories
    bl_ids = set(np.unique(labels_bl)) - {0}
    fu_ids = set(np.unique(labels_fu)) - {0}
    stable = bl_ids & fu_ids
    disappeared = bl_ids - fu_ids
    new = fu_ids - bl_ids

    # Determine slices to show
    if num_slices == 1:
        if slice_idx is None:
            slice_idx = _find_best_slice(labels_bl, labels_fu, axis)
        slice_indices = [slice_idx]
    else:
        slice_indices = _find_best_slices(labels_bl, labels_fu, axis, num_slices)

    # Create figure
    n_rows = len(slice_indices)
    if figsize is None:
        figsize = (10, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, facecolor="black")

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("black")

    # Draw each slice
    for row, sidx in enumerate(slice_indices):
        # Extract slices
        sl_flair_bl = np.rot90(_get_slice(flair_bl, sidx, axis))
        sl_flair_fu = np.rot90(_get_slice(flair_fu, sidx, axis))
        sl_labels_bl = np.rot90(_get_slice(labels_bl, sidx, axis))
        sl_labels_fu = np.rot90(_get_slice(labels_fu, sidx, axis))

        # Left: Baseline with cyan outlines
        axes[row, 0].imshow(sl_flair_bl, cmap="gray", vmin=0, vmax=1)
        _draw_lesions(axes[row, 0], sl_labels_bl, color="cyan")
        if row == 0:
            axes[row, 0].set_title(
                "Baseline", fontsize=14, color="white", fontweight="bold"
            )
        axes[row, 0].axis("off")

        # Add slice number on the left
        axes[row, 0].text(
            5,
            15,
            f"Slice {sidx}",
            fontsize=10,
            color="white",
            fontweight="bold",
            va="top",
        )

        # Right: Follow-up with colored lesions
        axes[row, 1].imshow(sl_flair_fu, cmap="gray", vmin=0, vmax=1)
        _draw_colored_lesions(
            axes[row, 1], sl_labels_bl, sl_labels_fu, stable, disappeared, new
        )
        if row == 0:
            axes[row, 1].set_title(
                "Follow-up", fontsize=14, color="white", fontweight="bold"
            )
        axes[row, 1].axis("off")

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS["stable"], edgecolor="white", label="Stable"),
        Patch(facecolor=COLORS["new"], edgecolor="white", label="New"),
        Patch(facecolor=COLORS["disappeared"], edgecolor="white", label="Disappeared"),
    ]
    legend = fig.legend(
        handles=legend_elements, loc="lower center", ncol=3, fontsize=11, frameon=False
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved visualization: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _draw_lesions(ax, labels, color="cyan"):
    """Draw lesion outlines with IDs."""
    for lid in np.unique(labels):
        if lid == 0:
            continue
        mask = labels == lid
        if not np.any(mask):
            continue
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)
        cy, cx = ndimage.center_of_mass(mask)
        ax.text(
            cx,
            cy,
            str(lid),
            fontsize=9,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor="white"),
        )


def _draw_colored_lesions(ax, labels_bl, labels_fu, stable, disappeared, new):
    """Draw lesions colored by category."""
    # Stable (yellow)
    for lid in stable:
        mask = labels_fu == lid
        if np.any(mask):
            _draw_single_lesion(ax, mask, lid, COLORS["stable"])

    # Disappeared (green) - show from baseline
    for lid in disappeared:
        mask = labels_bl == lid
        if np.any(mask):
            _draw_single_lesion(ax, mask, lid, COLORS["disappeared"])

    # New (red)
    for lid in new:
        mask = labels_fu == lid
        if np.any(mask):
            _draw_single_lesion(ax, mask, lid, COLORS["new"])


def _draw_single_lesion(ax, mask, lid, color):
    """Draw a single lesion with fill, outline, and ID."""
    # Semi-transparent fill
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask, :3] = color
    overlay[mask, 3] = 0.6
    ax.imshow(overlay)

    # Outline
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=2)

    # ID label
    cy, cx = ndimage.center_of_mass(mask)
    ax.text(
        cx,
        cy,
        str(lid),
        fontsize=9,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor="white"),
    )


def _find_best_slice(labels_bl, labels_fu, axis) -> int:
    """Find single slice with most lesion activity."""
    n = labels_bl.shape[axis]
    best_slice, best_count = n // 2, 0

    for s in range(n):
        bl = _get_slice(labels_bl, s, axis)
        fu = _get_slice(labels_fu, s, axis)
        count = len(set(np.unique(bl)) | set(np.unique(fu))) - 1
        if count > best_count:
            best_count = count
            best_slice = s

    return best_slice


def _find_best_slices(labels_bl, labels_fu, axis, num_slices: int) -> List[int]:
    """Find multiple slices with lesion activity, evenly distributed."""
    n = labels_bl.shape[axis]

    # Score each slice
    slice_scores = []
    for s in range(n):
        bl = _get_slice(labels_bl, s, axis)
        fu = _get_slice(labels_fu, s, axis)

        bl_ids = set(np.unique(bl)) - {0}
        fu_ids = set(np.unique(fu)) - {0}

        # Score based on lesion presence and change activity
        stable = len(bl_ids & fu_ids)
        disappeared = len(bl_ids - fu_ids)
        new = len(fu_ids - bl_ids)

        # Prioritize slices with changes
        score = stable + disappeared * 3 + new * 3
        if score > 0:
            slice_scores.append((s, score))

    if not slice_scores:
        # No lesions found, return evenly spaced slices
        return list(np.linspace(n // 4, 3 * n // 4, num_slices, dtype=int))

    # Sort by score and pick top slices, but ensure they're spread out
    slice_scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    min_distance = max(5, n // (num_slices + 1))  # Minimum spacing between slices

    for s, score in slice_scores:
        if len(selected) >= num_slices:
            break
        # Check if this slice is far enough from already selected
        if all(abs(s - sel) >= min_distance for sel in selected):
            selected.append(s)

    # If we couldn't get enough spread-out slices, just take the top ones
    if len(selected) < num_slices:
        for s, score in slice_scores:
            if s not in selected:
                selected.append(s)
            if len(selected) >= num_slices:
                break

    return sorted(selected)


def _normalize(img):
    """Normalize to 0-1 with percentile clipping."""
    if not np.any(img > 0):
        return img
    p1, p99 = np.percentile(img[img > 0], [1, 99])
    return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)


def _get_slice(vol, idx, axis=2):
    """Extract 2D slice from 3D volume."""
    slices = [slice(None)] * 3
    slices[axis] = idx
    return vol[tuple(slices)]
