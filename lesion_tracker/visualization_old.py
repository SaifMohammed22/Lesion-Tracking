"""
Visualization module for lesion tracking.

Provides tools for visualizing lesions, tracking results, and generating reports.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

from .utils import normalize_intensity, create_output_directory, get_centroid
from .labeler import Lesion
from scipy import ndimage


def get_2d_centroids_from_labeled_slice(labeled_slice: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Get 2D centroids for each label in a 2D labeled slice.
    
    Args:
        labeled_slice: 2D array with integer labels
        
    Returns:
        Dictionary mapping label -> (x, y) centroid
    """
    centroids = {}
    unique_labels = np.unique(labeled_slice)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        coords = np.where(labeled_slice == label)
        if len(coords[0]) > 0:
            centroids[label] = (np.mean(coords[0]), np.mean(coords[1]))
    
    return centroids


class LesionVisualizer:
    """
    Class for visualizing lesion tracking results.
    """
    
    # Color map for lesion categories
    CATEGORY_COLORS = {
        'stable': '#4CAF50',      # Green
        'growing': '#F44336',      # Red
        'shrinking': '#2196F3',    # Blue
        'new': '#FF9800',          # Orange
        'resolved': '#9C27B0',     # Purple
        'unmatched': '#757575'     # Gray
    }
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def plot_lesion_overlay(
        self,
        flair: np.ndarray,
        lesion_mask: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        ax: Optional[plt.Axes] = None,
        title: str = '',
        alpha: float = 0.5,
        cmap: str = 'hot',
        show_labels: bool = False,
        labeled_mask: Optional[np.ndarray] = None,
        label_fontsize: int = 10,
        label_color: str = 'white'
    ) -> plt.Axes:
        """
        Plot FLAIR image with lesion mask overlay.
        
        Args:
            flair: FLAIR image array
            lesion_mask: Lesion mask array
            slice_idx: Slice index to display (auto-select if None)
            axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            alpha: Overlay transparency
            cmap: Colormap for lesion overlay
            show_labels: Whether to show lesion number labels
            labeled_mask: Labeled mask with unique integer for each lesion (required if show_labels=True)
            label_fontsize: Font size for lesion labels
            label_color: Color for label text
            
        Returns:
            Matplotlib Axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Auto-select slice with most lesion voxels
        if slice_idx is None:
            lesion_sums = np.sum(lesion_mask > 0, axis=tuple(
                i for i in range(3) if i != axis
            ))
            slice_idx = np.argmax(lesion_sums)
        
        # Extract slices
        slices = [slice(None)] * 3
        slices[axis] = slice_idx
        flair_slice = flair[tuple(slices)]
        mask_slice = lesion_mask[tuple(slices)]
        
        # Normalize FLAIR
        flair_norm = normalize_intensity(flair_slice)
        
        # Plot FLAIR
        ax.imshow(flair_norm.T, cmap='gray', origin='lower')
        
        # Overlay mask
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(mask_overlay.T, cmap=cmap, alpha=alpha, origin='lower')
        
        # Add lesion labels if requested
        if show_labels and labeled_mask is not None:
            labeled_slice = labeled_mask[tuple(slices)]
            centroids = get_2d_centroids_from_labeled_slice(labeled_slice)
            
            for label, (cx, cy) in centroids.items():
                ax.text(
                    cx, cy, str(label),
                    fontsize=label_fontsize,
                    color=label_color,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none')
                )
        
        ax.set_title(title)
        ax.axis('off')
        
        return ax
    
    def plot_labeled_lesions(
        self,
        flair: np.ndarray,
        labeled_mask: np.ndarray,
        lesions: Optional[List[Lesion]] = None,
        slice_indices: Optional[List[int]] = None,
        axis: int = 2,
        num_slices: int = 6,
        save_path: Optional[str] = None,
        title: str = 'Labeled Lesions',
        cmap: str = 'tab20',
        label_fontsize: int = 9,
        show_volume: bool = True
    ) -> plt.Figure:
        """
        Plot multiple slices with numbered lesion labels.
        
        Args:
            flair: FLAIR image array
            labeled_mask: Labeled mask with unique integer for each lesion
            lesions: Optional list of Lesion objects for volume info
            slice_indices: Specific slice indices to show (auto-select if None)
            axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
            num_slices: Number of slices to show if auto-selecting
            save_path: Path to save figure
            title: Figure title
            cmap: Colormap for lesions
            label_fontsize: Font size for lesion labels
            show_volume: Whether to show volume in legend
            
        Returns:
            Matplotlib Figure object
        """
        # Auto-select slices with lesions
        if slice_indices is None:
            lesion_sums = np.sum(labeled_mask > 0, axis=tuple(
                i for i in range(3) if i != axis
            ))
            # Get slices with lesions, sorted by lesion content
            nonzero_slices = np.where(lesion_sums > 0)[0]
            if len(nonzero_slices) == 0:
                slice_indices = [labeled_mask.shape[axis] // 2]
            elif len(nonzero_slices) <= num_slices:
                slice_indices = list(nonzero_slices)
            else:
                # Sample evenly from slices with lesions
                step = len(nonzero_slices) // num_slices
                slice_indices = [nonzero_slices[i * step] for i in range(num_slices)]
        
        n_slices = len(slice_indices)
        n_cols = min(3, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_slices == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Get unique labels
        unique_labels = np.unique(labeled_mask)
        unique_labels = unique_labels[unique_labels > 0]
        n_labels = len(unique_labels)
        
        # Create color map
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, max(n_labels, 1)))
        label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for idx, slice_idx in enumerate(slice_indices):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            # Extract slices
            slices = [slice(None)] * 3
            slices[axis] = slice_idx
            flair_slice = flair[tuple(slices)]
            labeled_slice = labeled_mask[tuple(slices)]
            
            # Normalize and display FLAIR
            flair_norm = normalize_intensity(flair_slice)
            ax.imshow(flair_norm.T, cmap='gray', origin='lower')
            
            # Overlay each label with its color
            for label in unique_labels:
                if np.any(labeled_slice == label):
                    mask = (labeled_slice == label).astype(float)
                    mask_overlay = np.ma.masked_where(mask == 0, mask)
                    ax.imshow(mask_overlay.T, cmap=ListedColormap([label_colors[label]]), 
                             alpha=0.5, origin='lower')
            
            # Add number labels at centroids
            centroids = get_2d_centroids_from_labeled_slice(labeled_slice)
            for label, (cx, cy) in centroids.items():
                ax.text(
                    cx, cy, str(label),
                    fontsize=label_fontsize,
                    color='white',
                    ha='center',
                    va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none')
                )
            
            ax.set_title(f'Slice {slice_idx}')
            ax.axis('off')
        
        # Hide empty axes
        for idx in range(n_slices, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        # Create legend with lesion info
        legend_elements = []
        for label in unique_labels:
            label_text = f'Lesion {label}'
            if show_volume and lesions is not None:
                # Find the lesion object
                lesion_obj = next((l for l in lesions if l.label == label), None)
                if lesion_obj is not None:
                    label_text += f' ({lesion_obj.volume_mm3:.1f} mm³)'
            legend_elements.append(
                mpatches.Patch(color=label_colors[label], label=label_text)
            )
        
        if legend_elements:
            fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=min(5, len(legend_elements)), 
                      bbox_to_anchor=(0.5, 0.02),
                      fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1 + 0.03 * ((len(legend_elements) - 1) // 5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

    def plot_comparison(
        self,
        baseline_flair: np.ndarray,
        baseline_mask: np.ndarray,
        followup_flair: np.ndarray,
        followup_mask: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot side-by-side comparison of baseline and follow-up.
        
        Args:
            baseline_flair: Baseline FLAIR image
            baseline_mask: Baseline lesion mask
            followup_flair: Follow-up FLAIR image
            followup_mask: Follow-up lesion mask
            slice_idx: Slice index (auto-select if None)
            axis: Axis to slice
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Auto-select slice showing most lesion change
        if slice_idx is None:
            combined_mask = (baseline_mask > 0) | (followup_mask > 0)
            mask_sums = np.sum(combined_mask, axis=tuple(
                i for i in range(3) if i != axis
            ))
            slice_idx = np.argmax(mask_sums)
        
        self.plot_lesion_overlay(
            baseline_flair, baseline_mask,
            slice_idx=slice_idx, axis=axis,
            ax=axes[0], title='Baseline'
        )
        
        self.plot_lesion_overlay(
            followup_flair, followup_mask,
            slice_idx=slice_idx, axis=axis,
            ax=axes[1], title='Follow-up'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_tracking_results(
        self,
        baseline_flair: np.ndarray,
        followup_flair: np.ndarray,
        tracking_results: Dict[str, Any],
        baseline_labeled: np.ndarray,
        followup_labeled: np.ndarray,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize tracking results with color-coded categories.
        
        Args:
            baseline_flair: Baseline FLAIR image
            followup_flair: Follow-up FLAIR image
            tracking_results: Results from tracker
            baseline_labeled: Labeled baseline mask
            followup_labeled: Labeled follow-up mask
            slice_idx: Slice index
            axis: Slice axis
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Create category masks
        baseline_category_mask = np.zeros_like(baseline_labeled, dtype=np.float32)
        followup_category_mask = np.zeros_like(followup_labeled, dtype=np.float32)
        
        # Category to value mapping
        category_values = {
            'stable': 1,
            'growing': 2,
            'shrinking': 3,
            'new': 4,
            'resolved': 5
        }
        
        # Fill category masks from correspondences
        for corr in tracking_results.get('correspondences', []):
            cat_val = category_values.get(corr['category'], 0)
            baseline_category_mask[baseline_labeled == corr['baseline_label']] = cat_val
            followup_category_mask[followup_labeled == corr['followup_label']] = cat_val
        
        # Add new lesions
        for lesion_info in tracking_results.get('new_lesions', []):
            followup_category_mask[followup_labeled == lesion_info['label']] = category_values['new']
        
        # Add resolved lesions
        for lesion_info in tracking_results.get('resolved_lesions', []):
            baseline_category_mask[baseline_labeled == lesion_info['label']] = category_values['resolved']
        
        # Auto-select slice
        if slice_idx is None:
            combined = (baseline_category_mask > 0) | (followup_category_mask > 0)
            mask_sums = np.sum(combined, axis=tuple(i for i in range(3) if i != axis))
            slice_idx = np.argmax(mask_sums)
        
        # Create custom colormap
        colors = ['white', '#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0']
        cmap = ListedColormap(colors)
        
        # Extract slices
        slices = [slice(None)] * 3
        slices[axis] = slice_idx
        
        baseline_flair_slice = normalize_intensity(baseline_flair[tuple(slices)])
        followup_flair_slice = normalize_intensity(followup_flair[tuple(slices)])
        baseline_cat_slice = baseline_category_mask[tuple(slices)]
        followup_cat_slice = followup_category_mask[tuple(slices)]
        
        # Plot baseline
        axes[0].imshow(baseline_flair_slice.T, cmap='gray', origin='lower')
        mask_overlay = np.ma.masked_where(baseline_cat_slice == 0, baseline_cat_slice)
        axes[0].imshow(mask_overlay.T, cmap=cmap, alpha=0.6, origin='lower', vmin=0, vmax=5)
        axes[0].set_title('Baseline')
        axes[0].axis('off')
        
        # Plot follow-up
        axes[1].imshow(followup_flair_slice.T, cmap='gray', origin='lower')
        mask_overlay = np.ma.masked_where(followup_cat_slice == 0, followup_cat_slice)
        axes[1].imshow(mask_overlay.T, cmap=cmap, alpha=0.6, origin='lower', vmin=0, vmax=5)
        axes[1].set_title('Follow-up')
        axes[1].axis('off')
        
        # Create difference visualization
        diff_mask = followup_cat_slice - baseline_cat_slice
        axes[2].imshow(followup_flair_slice.T, cmap='gray', origin='lower')
        
        # Show new lesions in orange, resolved in purple
        new_mask = np.ma.masked_where(followup_cat_slice != 4, followup_cat_slice)
        resolved_mask = np.ma.masked_where(baseline_cat_slice != 5, baseline_cat_slice)
        
        axes[2].imshow(new_mask.T, cmap=ListedColormap(['white', '#FF9800']), 
                       alpha=0.7, origin='lower', vmin=0, vmax=4)
        axes[2].set_title('Changes (New/Resolved)')
        axes[2].axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#4CAF50', label='Stable'),
            mpatches.Patch(color='#F44336', label='Growing'),
            mpatches.Patch(color='#2196F3', label='Shrinking'),
            mpatches.Patch(color='#FF9800', label='New'),
            mpatches.Patch(color='#9C27B0', label='Resolved'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                   bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_tracking_multi_slice(
        self,
        baseline_flair: np.ndarray,
        followup_flair: np.ndarray,
        tracking_results: Dict[str, Any],
        baseline_labeled: np.ndarray,
        followup_labeled: np.ndarray,
        slice_indices: Optional[List[int]] = None,
        axis: int = 2,
        num_slices: int = 4,
        save_path: Optional[str] = None,
        show_labels: bool = True,
        label_fontsize: int = 9
    ) -> plt.Figure:
        """
        Plot multiple slices showing tracking results with category-based colors.
        
        Each lesion is colored based on its tracking result:
        - Stable (green): Lesion matched with <20% volume change
        - Growing (red): Lesion matched with >20% volume increase
        - Shrinking (blue): Lesion matched with >20% volume decrease
        - New (orange): Lesion only in follow-up
        - Resolved (purple): Lesion only in baseline
        
        Args:
            baseline_flair: Baseline FLAIR image
            followup_flair: Follow-up FLAIR image
            tracking_results: Results from tracker containing correspondences, new_lesions, resolved_lesions
            baseline_labeled: Labeled baseline mask (each lesion has unique integer)
            followup_labeled: Labeled follow-up mask
            slice_indices: Specific slice indices to show (auto-select if None)
            axis: Axis to slice (0=sagittal, 1=coronal, 2=axial)
            num_slices: Number of slices to show if auto-selecting
            save_path: Path to save figure
            show_labels: Whether to show lesion number labels
            label_fontsize: Font size for lesion labels
            
        Returns:
            Matplotlib Figure object
        """
        # Category to color mapping
        category_colors = {
            'stable': '#4CAF50',     # Green
            'growing': '#F44336',    # Red
            'shrinking': '#2196F3',  # Blue
            'new': '#FF9800',        # Orange
            'resolved': '#9C27B0'    # Purple
        }
        
        # Build lesion-to-category mappings
        baseline_lesion_category = {}  # label -> category
        followup_lesion_category = {}  # label -> category
        
        # Matched lesions (stable/growing/shrinking)
        for corr in tracking_results.get('correspondences', []):
            category = corr['category']
            baseline_lesion_category[corr['baseline_label']] = category
            followup_lesion_category[corr['followup_label']] = category
        
        # New lesions (only in follow-up)
        for lesion_info in tracking_results.get('new_lesions', []):
            followup_lesion_category[lesion_info['label']] = 'new'
        
        # Resolved lesions (only in baseline)
        for lesion_info in tracking_results.get('resolved_lesions', []):
            baseline_lesion_category[lesion_info['label']] = 'resolved'
        
        # Auto-select slices with lesions from both timepoints
        if slice_indices is None:
            combined_mask = (baseline_labeled > 0) | (followup_labeled > 0)
            lesion_sums = np.sum(combined_mask > 0, axis=tuple(
                i for i in range(3) if i != axis
            ))
            nonzero_slices = np.where(lesion_sums > 0)[0]
            if len(nonzero_slices) == 0:
                slice_indices = [combined_mask.shape[axis] // 2]
            elif len(nonzero_slices) <= num_slices:
                slice_indices = list(nonzero_slices)
            else:
                step = len(nonzero_slices) // num_slices
                slice_indices = [nonzero_slices[i * step] for i in range(num_slices)]
        
        n_slices = len(slice_indices)
        
        # Create figure: 2 rows (baseline, followup) x n_slices columns
        fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))
        if n_slices == 1:
            axes = axes.reshape(2, 1)
        
        for col_idx, slice_idx in enumerate(slice_indices):
            slices = [slice(None)] * 3
            slices[axis] = slice_idx
            
            # --- Baseline row ---
            ax_base = axes[0, col_idx]
            baseline_flair_slice = normalize_intensity(baseline_flair[tuple(slices)])
            baseline_labeled_slice = baseline_labeled[tuple(slices)]
            
            ax_base.imshow(baseline_flair_slice.T, cmap='gray', origin='lower')
            
            # Overlay each lesion with its category color
            for label, category in baseline_lesion_category.items():
                if np.any(baseline_labeled_slice == label):
                    mask = (baseline_labeled_slice == label).astype(float)
                    mask_overlay = np.ma.masked_where(mask == 0, mask)
                    color = category_colors.get(category, '#757575')
                    ax_base.imshow(mask_overlay.T, cmap=ListedColormap([color]), 
                                   alpha=0.6, origin='lower')
            
            # Add lesion labels
            if show_labels:
                centroids = get_2d_centroids_from_labeled_slice(baseline_labeled_slice)
                for label, (cx, cy) in centroids.items():
                    ax_base.text(
                        cx, cy, str(label),
                        fontsize=label_fontsize,
                        color='white',
                        ha='center', va='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none')
                    )
            
            ax_base.set_title(f'Slice {slice_idx}' if col_idx == 0 else f'{slice_idx}')
            ax_base.axis('off')
            
            # --- Follow-up row ---
            ax_fu = axes[1, col_idx]
            followup_flair_slice = normalize_intensity(followup_flair[tuple(slices)])
            followup_labeled_slice = followup_labeled[tuple(slices)]
            
            ax_fu.imshow(followup_flair_slice.T, cmap='gray', origin='lower')
            
            # Overlay each lesion with its category color
            for label, category in followup_lesion_category.items():
                if np.any(followup_labeled_slice == label):
                    mask = (followup_labeled_slice == label).astype(float)
                    mask_overlay = np.ma.masked_where(mask == 0, mask)
                    color = category_colors.get(category, '#757575')
                    ax_fu.imshow(mask_overlay.T, cmap=ListedColormap([color]), 
                                 alpha=0.6, origin='lower')
            
            # Add lesion labels
            if show_labels:
                centroids = get_2d_centroids_from_labeled_slice(followup_labeled_slice)
                for label, (cx, cy) in centroids.items():
                    ax_fu.text(
                        cx, cy, str(label),
                        fontsize=label_fontsize,
                        color='white',
                        ha='center', va='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none')
                    )
            
            ax_fu.axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Baseline', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Follow-up', fontsize=12, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#4CAF50', label='Stable'),
            mpatches.Patch(color='#F44336', label='Growing'),
            mpatches.Patch(color='#2196F3', label='Shrinking'),
            mpatches.Patch(color='#FF9800', label='New'),
            mpatches.Patch(color='#9C27B0', label='Resolved'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)
        
        fig.suptitle('Tracking Results by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_volume_changes(
        self,
        tracking_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot volume changes for tracked lesions.
        
        Args:
            tracking_results: Results from tracker
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        correspondences = tracking_results.get('correspondences', [])
        
        if not correspondences:
            axes[0].text(0.5, 0.5, 'No matched lesions', ha='center', va='center')
            axes[1].text(0.5, 0.5, 'No matched lesions', ha='center', va='center')
            return fig
        
        # Extract data
        baseline_volumes = [c['baseline_volume_mm3'] for c in correspondences]
        followup_volumes = [c['followup_volume_mm3'] for c in correspondences]
        categories = [c['category'] for c in correspondences]
        
        colors = [self.CATEGORY_COLORS.get(cat, '#757575') for cat in categories]
        
        # Scatter plot: baseline vs follow-up volume
        axes[0].scatter(baseline_volumes, followup_volumes, c=colors, s=100, alpha=0.7)
        
        # Add identity line
        max_vol = max(max(baseline_volumes), max(followup_volumes))
        axes[0].plot([0, max_vol], [0, max_vol], 'k--', alpha=0.5, label='No change')
        
        axes[0].set_xlabel('Baseline Volume (mm³)')
        axes[0].set_ylabel('Follow-up Volume (mm³)')
        axes[0].set_title('Lesion Volume: Baseline vs Follow-up')
        axes[0].legend()
        
        # Bar plot: volume change by category
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Add new and resolved
        new_count = len(tracking_results.get('new_lesions', []))
        resolved_count = len(tracking_results.get('resolved_lesions', []))
        
        if new_count > 0:
            category_counts['new'] = new_count
        if resolved_count > 0:
            category_counts['resolved'] = resolved_count
        
        cats = list(category_counts.keys())
        counts = list(category_counts.values())
        bar_colors = [self.CATEGORY_COLORS.get(cat, '#757575') for cat in cats]
        
        axes[1].bar(cats, counts, color=bar_colors)
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Lesion Count by Category')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_multi_slice(
        self,
        flair: np.ndarray,
        lesion_mask: np.ndarray,
        n_slices: int = 9,
        axis: int = 2,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot multiple slices with lesion overlay.
        
        Args:
            flair: FLAIR image array
            lesion_mask: Lesion mask array
            n_slices: Number of slices to show
            axis: Slice axis
            save_path: Path to save figure
            
        Returns:
            Matplotlib Figure object
        """
        # Find slices with lesions
        lesion_slices = np.where(np.sum(lesion_mask > 0, axis=tuple(
            i for i in range(3) if i != axis
        )) > 0)[0]
        
        if len(lesion_slices) == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, 'No lesions found', ha='center', va='center')
            return fig
        
        # Select evenly spaced slices
        if len(lesion_slices) <= n_slices:
            selected_slices = lesion_slices
        else:
            indices = np.linspace(0, len(lesion_slices) - 1, n_slices, dtype=int)
            selected_slices = lesion_slices[indices]
        
        n_cols = min(3, len(selected_slices))
        n_rows = int(np.ceil(len(selected_slices) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        
        for idx, slice_idx in enumerate(selected_slices):
            row = idx // n_cols
            col = idx % n_cols
            
            self.plot_lesion_overlay(
                flair, lesion_mask,
                slice_idx=slice_idx, axis=axis,
                ax=axes[row, col],
                title=f'Slice {slice_idx}'
            )
        
        # Hide unused axes
        for idx in range(len(selected_slices), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report_figures(
        self,
        baseline_flair: np.ndarray,
        baseline_mask: np.ndarray,
        followup_flair: np.ndarray,
        followup_mask: np.ndarray,
        tracking_results: Dict[str, Any],
        baseline_labeled: np.ndarray,
        followup_labeled: np.ndarray,
        output_dir: str,
        baseline_lesions: Optional[List[Lesion]] = None,
        followup_lesions: Optional[List[Lesion]] = None
    ) -> List[str]:
        """
        Generate all report figures.
        
        Args:
            baseline_flair: Baseline FLAIR image
            baseline_mask: Baseline mask
            followup_flair: Follow-up FLAIR image
            followup_mask: Follow-up mask
            tracking_results: Tracking results
            baseline_labeled: Labeled baseline mask
            followup_labeled: Labeled follow-up mask
            output_dir: Output directory
            baseline_lesions: Optional list of baseline Lesion objects
            followup_lesions: Optional list of follow-up Lesion objects
            
        Returns:
            List of saved figure paths
        """
        create_output_directory(output_dir)
        saved_paths = []
        
        # Comparison figure
        comp_path = os.path.join(output_dir, 'comparison.png')
        self.plot_comparison(
            baseline_flair, baseline_mask,
            followup_flair, followup_mask,
            save_path=comp_path
        )
        plt.close()
        saved_paths.append(comp_path)
        
        # Tracking results figure
        track_path = os.path.join(output_dir, 'tracking_results.png')
        self.plot_tracking_results(
            baseline_flair, followup_flair,
            tracking_results,
            baseline_labeled, followup_labeled,
            save_path=track_path
        )
        plt.close()
        saved_paths.append(track_path)
        
        # Multi-slice tracking comparison (category-based colors)
        track_multi_path = os.path.join(output_dir, 'tracking_multi_slice.png')
        self.plot_tracking_multi_slice(
            baseline_flair, followup_flair,
            tracking_results,
            baseline_labeled, followup_labeled,
            save_path=track_multi_path
        )
        plt.close()
        saved_paths.append(track_multi_path)
        
        # Volume changes figure
        vol_path = os.path.join(output_dir, 'volume_changes.png')
        self.plot_volume_changes(tracking_results, save_path=vol_path)
        plt.close()
        saved_paths.append(vol_path)
        
        # Multi-slice baseline
        base_multi_path = os.path.join(output_dir, 'baseline_multi_slice.png')
        self.plot_multi_slice(baseline_flair, baseline_mask, save_path=base_multi_path)
        plt.close()
        saved_paths.append(base_multi_path)
        
        # Multi-slice follow-up
        fu_multi_path = os.path.join(output_dir, 'followup_multi_slice.png')
        self.plot_multi_slice(followup_flair, followup_mask, save_path=fu_multi_path)
        plt.close()
        saved_paths.append(fu_multi_path)
        
        # Labeled lesions - baseline
        base_labeled_path = os.path.join(output_dir, 'baseline_labeled_lesions.png')
        self.plot_labeled_lesions(
            baseline_flair, baseline_labeled,
            lesions=baseline_lesions,
            save_path=base_labeled_path,
            title='Baseline - Labeled Lesions'
        )
        plt.close()
        saved_paths.append(base_labeled_path)
        
        # Labeled lesions - follow-up
        fu_labeled_path = os.path.join(output_dir, 'followup_labeled_lesions.png')
        self.plot_labeled_lesions(
            followup_flair, followup_labeled,
            lesions=followup_lesions,
            save_path=fu_labeled_path,
            title='Follow-up - Labeled Lesions'
        )
        plt.close()
        saved_paths.append(fu_labeled_path)
        
        return saved_paths
    
    def plot_comprehensive_view(
        self,
        flair: np.ndarray,
        labeled_mask: np.ndarray,
        lesions: Optional[List] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Best of both worlds: overview + detail views.
        
        Layout:
        ┌─────────────────────────────────────┐
        │  OVERVIEW (6 spread slices)         │
        ├─────────────────────────────────────┤
        │  DETAIL: 6 contiguous around peak   │
        └─────────────────────────────────────┘
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Get slice info
        z_coords = np.where(labeled_mask > 0)[2]
        if len(z_coords) == 0:
            return fig
        
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # Find peak slice (most lesion activity)
        lesion_counts = np.sum(labeled_mask > 0, axis=(0, 1))
        peak_slice = int(np.argmax(lesion_counts))
        
        # OVERVIEW: spread slices
        spread_slices = np.linspace(z_min, z_max, 6, dtype=int)
        
        # DETAIL: contiguous around peak
        half = 3
        start = max(0, peak_slice - half)
        end = min(labeled_mask.shape[2], start + 6)
        contiguous_slices = list(range(start, end))
        
        # Color map for lesions
        num_lesions = int(labeled_mask.max())
        colors = plt.cm.tab20(np.linspace(0, 1, max(num_lesions, 1)))
        
        # Plot OVERVIEW (top row)
        for i, slice_idx in enumerate(spread_slices):
            ax = fig.add_subplot(2, 6, i + 1)
            self._plot_single_slice(ax, flair, labeled_mask, slice_idx, colors, num_lesions)
            ax.set_title(f'Slice {slice_idx}', fontsize=10)
            if i == 0:
                ax.set_ylabel('OVERVIEW\n(spread)', fontsize=12, fontweight='bold')
        
        # Plot DETAIL (bottom row)
        for i, slice_idx in enumerate(contiguous_slices):
            ax = fig.add_subplot(2, 6, i + 7)
            self._plot_single_slice(ax, flair, labeled_mask, slice_idx, colors, num_lesions)
            ax.set_title(f'Slice {slice_idx}', fontsize=10)
            if i == 0:
                ax.set_ylabel('DETAIL\n(contiguous)', fontsize=12, fontweight='bold')
        
        # Add legend
        if lesions:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[i], markersize=8,
                          label=f'L{l.label}: {l.volume_mm3:.0f}mm³')
                for i, l in enumerate(lesions[:10])  # Max 10 in legend
            ]
            fig.legend(handles=legend_elements, loc='lower center', 
                      ncol=min(5, len(lesions)), fontsize=9)
        
        plt.suptitle('Lesion Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


    def _plot_single_slice(
        self, 
        ax: plt.Axes, 
        flair: np.ndarray, 
        labeled_mask: np.ndarray, 
        slice_idx: int,
        colors: np.ndarray,
        num_lesions: int
    ) -> None:
        """Helper to plot a single slice with colored lesions."""
        from matplotlib.colors import ListedColormap
        
        flair_slice = flair[:, :, slice_idx]
        label_slice = labeled_mask[:, :, slice_idx]
        
        # Show FLAIR
        ax.imshow(flair_slice.T, cmap='gray', origin='lower')
        
        # Overlay each lesion
        for label in range(1, num_lesions + 1):
            lesion_mask = (label_slice == label)
            if lesion_mask.any():
                masked = np.ma.masked_where(~lesion_mask, np.ones_like(label_slice))
                ax.imshow(masked.T, cmap=ListedColormap([colors[label-1]]),
                         alpha=0.6, origin='lower', vmin=0, vmax=1)
                
                # Add number label
                coords = np.where(lesion_mask)
                if len(coords[0]) > 0:
                    cx, cy = np.mean(coords[0]), np.mean(coords[1])
                    ax.text(cy, cx, str(label), color='white', fontsize=8,
                           ha='center', va='center', fontweight='bold',
                           bbox=dict(boxstyle='circle,pad=0.1', 
                                    facecolor=colors[label-1][:3], alpha=0.8))
        
        ax.axis('off')
