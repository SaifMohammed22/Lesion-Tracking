"""
Lesion Tracking Visualization

Output: PNG image showing lesion tracking between baseline and follow-up.

Layout (2 rows per slice):
- Top row: Original FLAIR images (baseline and follow-up) for reference
- Bottom row: Labeled results with color-coded lesions by status

Status Color Scheme:
- Yellow: Present (stable, ±25% volume)
- Orange: Enlarged (>+25% volume)
- Light Blue: Shrinking (>-25% volume)
- Green: Absent (disappeared)
- Purple: Merged
- Cyan: Split
- Red: New

Supports single slice or multi-slice grid view.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage
from typing import Optional, List, Dict, Any

try:
    from .utils import load_nifti
except ImportError:
    from utils import load_nifti


# Color scheme for each status
COLORS = {
    "present": (0.95, 0.77, 0.06),  # Yellow
    "enlarged": (1.0, 0.55, 0.0),  # Orange
    "shrinking": (0.53, 0.81, 0.92),  # Light Blue
    "absent": (0.18, 0.80, 0.44),  # Green
    "merged": (0.58, 0.40, 0.74),  # Purple
    "split": (0.0, 0.75, 0.75),  # Cyan/Teal
    "new": (0.91, 0.30, 0.24),  # Red
    "outline": "cyan",  # For baseline outlines
}


def visualize_tracking(
    flair_baseline_path: str,
    flair_followup_path: str,
    baseline_labeled: Optional[np.ndarray] = None,
    followup_labeled: Optional[np.ndarray] = None,
    lesions: Optional[List[Dict[str, Any]]] = None,
    baseline_labeled_path: Optional[str] = None,
    followup_labeled_path: Optional[str] = None,
    slice_idx: Optional[int] = None,
    num_slices: int = 1,
    axis: int = 2,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Create lesion tracking visualization and save as PNG.

    Layout:
    - Top row: Original FLAIR images (baseline | follow-up) for visual comparison
    - Bottom row: Tracking results with color-coded lesions by status

    Args:
        flair_baseline_path: Path to baseline FLAIR
        flair_followup_path: Path to follow-up FLAIR (registered)
        baseline_labeled: Numpy array of labeled baseline lesions (preferred)
        followup_labeled: Numpy array of labeled follow-up lesions (preferred)
        lesions: List of lesion dicts with 'id' and 'status' keys (from track_lesions)
        baseline_labeled_path: Path to labeled baseline lesions (fallback)
        followup_labeled_path: Path to labeled follow-up lesions (fallback)
        slice_idx: Slice to display (auto-selected if None)
        num_slices: Number of slices to show (1 = single best, >1 = grid)
        axis: Axis for slicing (0=sagittal, 1=coronal, 2=axial)
        save_path: Path to save figure (e.g., "output.png")
        show: Whether to display figure
        figsize: Figure size (auto-calculated if None)

    Returns:
        matplotlib Figure
    """
    # Load FLAIR data
    flair_bl, _ = load_nifti(flair_baseline_path)
    flair_fu, _ = load_nifti(flair_followup_path)

    # Use provided arrays or load from paths
    if baseline_labeled is not None:
        labels_bl = baseline_labeled
    elif baseline_labeled_path is not None:
        labels_bl, _ = load_nifti(baseline_labeled_path)
    else:
        raise ValueError(
            "Must provide either baseline_labeled array or baseline_labeled_path"
        )

    if followup_labeled is not None:
        labels_fu = followup_labeled
    elif followup_labeled_path is not None:
        labels_fu, _ = load_nifti(followup_labeled_path)
    else:
        raise ValueError(
            "Must provide either followup_labeled array or followup_labeled_path"
        )

    labels_bl = labels_bl.astype(np.int16)
    labels_fu = labels_fu.astype(np.int16)

    # Normalize FLAIR for display
    flair_bl_norm = _normalize(flair_bl)
    flair_fu_norm = _normalize(flair_fu)

    # Build status lookup from lesions list
    status_map = {}
    if lesions:
        for lesion in lesions:
            status_map[lesion["id"]] = lesion["status"].lower()

    # If no lesions list provided, infer from labels (backward compatibility)
    if not status_map:
        bl_ids = set(np.unique(labels_bl)) - {0}
        fu_ids = set(np.unique(labels_fu)) - {0}
        for lid in bl_ids & fu_ids:
            status_map[lid] = "present"
        for lid in bl_ids - fu_ids:
            status_map[lid] = "absent"
        for lid in fu_ids - bl_ids:
            status_map[lid] = "new"

    # Determine slices to show
    if num_slices == 1:
        if slice_idx is None:
            slice_idx = _find_best_slice(labels_bl, labels_fu, axis)
        slice_indices = [slice_idx]
    else:
        slice_indices = _find_best_slices(labels_bl, labels_fu, axis, num_slices)

    # Create figure: 2 columns (baseline, follow-up) x (2 rows per slice)
    n_slices = len(slice_indices)
    n_rows = n_slices * 2  # 2 rows per slice (original + labeled)

    if figsize is None:
        figsize = (10, 4 * n_slices)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, facecolor="black")

    # Handle single slice case (2 rows)
    if n_rows == 2:
        axes = axes.reshape(2, 2)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("black")

    # Draw each slice (2 rows per slice)
    for slice_num, sidx in enumerate(slice_indices):
        row_original = slice_num * 2
        row_labeled = slice_num * 2 + 1

        # Extract slices
        sl_flair_bl = np.rot90(_get_slice(flair_bl_norm, sidx, axis))
        sl_flair_fu = np.rot90(_get_slice(flair_fu_norm, sidx, axis))
        sl_labels_bl = np.rot90(_get_slice(labels_bl, sidx, axis))
        sl_labels_fu = np.rot90(_get_slice(labels_fu, sidx, axis))

        # ==================== TOP ROW: Original FLAIR images ====================
        axes[row_original, 0].imshow(sl_flair_bl, cmap="gray", vmin=0, vmax=1)
        if slice_num == 0:
            axes[row_original, 0].set_title(
                "Baseline (Original)", fontsize=14, color="white", fontweight="bold"
            )
        axes[row_original, 0].axis("off")

        # Add slice number
        axes[row_original, 0].text(
            5,
            15,
            f"Slice {sidx}",
            fontsize=10,
            color="white",
            fontweight="bold",
            va="top",
        )

        axes[row_original, 1].imshow(sl_flair_fu, cmap="gray", vmin=0, vmax=1)
        if slice_num == 0:
            axes[row_original, 1].set_title(
                "Follow-up (Original)", fontsize=14, color="white", fontweight="bold"
            )
        axes[row_original, 1].axis("off")

        # ==================== BOTTOM ROW: Labeled tracking results ====================
        # Left: Baseline with lesion outlines colored by status
        axes[row_labeled, 0].imshow(sl_flair_bl, cmap="gray", vmin=0, vmax=1)
        _draw_baseline_lesions(axes[row_labeled, 0], sl_labels_bl, status_map)
        if slice_num == 0:
            axes[row_labeled, 0].set_title(
                "Baseline (Labeled)", fontsize=14, color="white", fontweight="bold"
            )
        axes[row_labeled, 0].axis("off")

        # Right: Follow-up with color-coded lesions
        axes[row_labeled, 1].imshow(sl_flair_fu, cmap="gray", vmin=0, vmax=1)
        _draw_followup_lesions(
            axes[row_labeled, 1], sl_labels_bl, sl_labels_fu, status_map
        )
        if slice_num == 0:
            axes[row_labeled, 1].set_title(
                "Follow-up (Tracking Results)",
                fontsize=14,
                color="white",
                fontweight="bold",
            )
        axes[row_labeled, 1].axis("off")

    # Legend with all statuses
    legend_elements = [
        Patch(facecolor=COLORS["present"], edgecolor="white", label="Present"),
        Patch(facecolor=COLORS["enlarged"], edgecolor="white", label="Enlarged"),
        Patch(facecolor=COLORS["shrinking"], edgecolor="white", label="Shrinking"),
        Patch(facecolor=COLORS["absent"], edgecolor="white", label="Absent"),
        Patch(facecolor=COLORS["merged"], edgecolor="white", label="Merged"),
        Patch(facecolor=COLORS["split"], edgecolor="white", label="Split"),
        Patch(facecolor=COLORS["new"], edgecolor="white", label="New"),
    ]
    legend = fig.legend(
        handles=legend_elements, loc="lower center", ncol=7, fontsize=9, frameon=False
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


def _draw_baseline_lesions(ax, labels_bl, status_map: Dict[int, str]):
    """Draw baseline lesions with outlines colored by status."""
    for lid in np.unique(labels_bl):
        if lid == 0:
            continue
        mask = labels_bl == lid
        if not np.any(mask):
            continue

        status = status_map.get(lid, "present")
        color = COLORS.get(status, COLORS["outline"])

        # Draw outline
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0)

        # Draw ID label
        cy, cx = ndimage.center_of_mass(mask)
        ax.text(
            cx,
            cy,
            str(lid),
            fontsize=6,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="circle,pad=0.1",
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            ),
        )


def _draw_followup_lesions(ax, labels_bl, labels_fu, status_map: Dict[int, str]):
    """Draw follow-up lesions colored by status."""
    # Get all unique IDs from both baseline and follow-up
    all_ids = set(np.unique(labels_bl)) | set(np.unique(labels_fu))
    all_ids.discard(0)

    for lid in all_ids:
        status = status_map.get(lid, "present")
        color = COLORS.get(status, COLORS["present"])

        # Determine which mask to use based on status
        if status == "absent":
            # Absent lesions: show from baseline with dashed outline
            mask = labels_bl == lid
            if np.any(mask):
                ax.contour(
                    mask,
                    levels=[0.5],
                    colors=[color],
                    linewidths=1.0,
                    linestyles="dashed",
                )
                cy, cx = ndimage.center_of_mass(mask)
                ax.text(
                    cx,
                    cy,
                    str(lid),
                    fontsize=6,
                    fontweight="bold",
                    color="white",
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="circle,pad=0.1",
                        facecolor=color,
                        edgecolor="white",
                        linewidth=0.5,
                    ),
                )
        elif status == "new":
            # New lesions: show from follow-up
            mask = labels_fu == lid
            if np.any(mask):
                _draw_single_lesion(ax, mask, lid, color)
        else:
            # Present, Enlarged, Shrinking, Merged, Split: show from follow-up
            mask = labels_fu == lid
            if np.any(mask):
                _draw_single_lesion(ax, mask, lid, color)


def _draw_single_lesion(ax, mask, lid, color):
    """Draw a single lesion with fill, outline, and small ID."""
    if not np.any(mask):
        return

    # Semi-transparent fill
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask, :3] = color
    overlay[mask, 3] = 0.5
    ax.imshow(overlay)

    # Outline
    ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.5)

    # ID label (small)
    cy, cx = ndimage.center_of_mass(mask)
    ax.text(
        cx,
        cy,
        str(lid),
        fontsize=6,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="circle,pad=0.1",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        ),
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
        present = len(bl_ids & fu_ids)
        absent = len(bl_ids - fu_ids)
        new = len(fu_ids - bl_ids)

        # Prioritize slices with changes
        score = present + absent * 3 + new * 3
        if score > 0:
            slice_scores.append((s, score))

    if not slice_scores:
        return list(np.linspace(n // 4, 3 * n // 4, num_slices, dtype=int))

    # Sort by score and pick top slices, spread out
    slice_scores.sort(key=lambda x: x[1], reverse=True)

    selected = []
    min_distance = max(5, n // (num_slices + 1))

    for s, score in slice_scores:
        if len(selected) >= num_slices:
            break
        if all(abs(s - sel) >= min_distance for sel in selected):
            selected.append(s)

    # Fill remaining slots
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
