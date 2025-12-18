"""
Visualization module for lesion tracking.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

from .utils import normalize_intensity, create_output_directory
from .labeler import Lesion


def get_2d_centroids(labeled_slice: np.ndarray) -> Dict[int, tuple]:
    """Get 2D centroids for each label in a slice."""
    centroids = {}
    for label in np.unique(labeled_slice):
        if label == 0:
            continue
        coords = np.where(labeled_slice == label)
        if len(coords[0]) > 0:
            centroids[label] = (np.mean(coords[0]), np.mean(coords[1]))
    return centroids


class LesionVisualizer:
    """Visualize lesion tracking results."""
    
    CATEGORY_COLORS = {
        'stable': '#4CAF50', 'growing': '#F44336', 'shrinking': '#2196F3',
        'new': '#FF9800', 'resolved': '#9C27B0', 'unmatched': '#757575'
    }
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
    
    def plot_lesion_overlay(
        self, flair: np.ndarray, lesion_mask: np.ndarray,
        slice_idx: Optional[int] = None, axis: int = 2,
        ax: Optional[plt.Axes] = None, title: str = '', alpha: float = 0.5
    ) -> plt.Axes:
        """Plot FLAIR with lesion mask overlay."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        if slice_idx is None:
            sums = np.sum(lesion_mask > 0, axis=tuple(i for i in range(3) if i != axis))
            slice_idx = np.argmax(sums)
        
        slices = [slice(None)] * 3
        slices[axis] = slice_idx
        flair_slice = normalize_intensity(flair[tuple(slices)])
        mask_slice = lesion_mask[tuple(slices)]
        
        ax.imshow(flair_slice.T, cmap='gray', origin='lower')
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(mask_overlay.T, cmap='hot', alpha=alpha, origin='lower')
        ax.set_title(title)
        ax.axis('off')
        return ax
    
    def plot_comparison(
        self, baseline_flair: np.ndarray, baseline_mask: np.ndarray,
        followup_flair: np.ndarray, followup_mask: np.ndarray,
        slice_idx: Optional[int] = None, axis: int = 2, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot side-by-side comparison of baseline and follow-up."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        if slice_idx is None:
            combined = (baseline_mask > 0) | (followup_mask > 0)
            sums = np.sum(combined, axis=tuple(i for i in range(3) if i != axis))
            slice_idx = np.argmax(sums)
        
        self.plot_lesion_overlay(baseline_flair, baseline_mask, slice_idx, axis, axes[0], 'Baseline')
        self.plot_lesion_overlay(followup_flair, followup_mask, slice_idx, axis, axes[1], 'Follow-up')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_tracking_results(
        self, baseline_flair: np.ndarray, followup_flair: np.ndarray,
        tracking_results: Dict[str, Any], baseline_labeled: np.ndarray, followup_labeled: np.ndarray,
        slice_idx: Optional[int] = None, axis: int = 2, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize tracking results with color-coded categories."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        cat_values = {'stable': 1, 'growing': 2, 'shrinking': 3, 'new': 4, 'resolved': 5}
        colors = ['white', '#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0']
        cmap = ListedColormap(colors)
        
        # Build category masks
        bl_cat = np.zeros_like(baseline_labeled, dtype=np.float32)
        fu_cat = np.zeros_like(followup_labeled, dtype=np.float32)
        
        for corr in tracking_results.get('correspondences', []):
            v = cat_values.get(corr['category'], 0)
            bl_cat[baseline_labeled == corr['baseline_label']] = v
            fu_cat[followup_labeled == corr['followup_label']] = v
        for info in tracking_results.get('new_lesions', []):
            fu_cat[followup_labeled == info['label']] = cat_values['new']
        for info in tracking_results.get('resolved_lesions', []):
            bl_cat[baseline_labeled == info['label']] = cat_values['resolved']
        
        if slice_idx is None:
            combined = (bl_cat > 0) | (fu_cat > 0)
            sums = np.sum(combined, axis=tuple(i for i in range(3) if i != axis))
            slice_idx = np.argmax(sums)
        
        slices = [slice(None)] * 3
        slices[axis] = slice_idx
        
        for i, (flair, cat_mask, title) in enumerate([
            (baseline_flair, bl_cat, 'Baseline'), (followup_flair, fu_cat, 'Follow-up')
        ]):
            axes[i].imshow(normalize_intensity(flair[tuple(slices)]).T, cmap='gray', origin='lower')
            mask = np.ma.masked_where(cat_mask[tuple(slices)] == 0, cat_mask[tuple(slices)])
            axes[i].imshow(mask.T, cmap=cmap, alpha=0.6, origin='lower', vmin=0, vmax=5)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        legend = [mpatches.Patch(color=c, label=l) for l, c in self.CATEGORY_COLORS.items() if l != 'unmatched']
        fig.legend(handles=legend, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_volume_changes(self, tracking_results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot volume changes for tracked lesions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        correspondences = tracking_results.get('correspondences', [])
        
        if not correspondences:
            axes[0].text(0.5, 0.5, 'No matched lesions', ha='center', va='center')
            axes[1].text(0.5, 0.5, 'No matched lesions', ha='center', va='center')
            return fig
        
        bl_vols = [c['baseline_volume_mm3'] for c in correspondences]
        fu_vols = [c['followup_volume_mm3'] for c in correspondences]
        colors = [self.CATEGORY_COLORS.get(c['category'], '#757575') for c in correspondences]
        
        axes[0].scatter(bl_vols, fu_vols, c=colors, s=100, alpha=0.7)
        max_vol = max(max(bl_vols), max(fu_vols))
        axes[0].plot([0, max_vol], [0, max_vol], 'k--', alpha=0.5)
        axes[0].set_xlabel('Baseline Volume (mm³)')
        axes[0].set_ylabel('Follow-up Volume (mm³)')
        axes[0].set_title('Volume: Baseline vs Follow-up')
        
        # Category counts
        cats = {}
        for c in correspondences:
            cats[c['category']] = cats.get(c['category'], 0) + 1
        cats['new'] = len(tracking_results.get('new_lesions', []))
        cats['resolved'] = len(tracking_results.get('resolved_lesions', []))
        cats = {k: v for k, v in cats.items() if v > 0}
        
        axes[1].bar(cats.keys(), cats.values(), color=[self.CATEGORY_COLORS.get(k, '#757575') for k in cats])
        axes[1].set_ylabel('Count')
        axes[1].set_title('Lesion Count by Category')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_multi_slice(
        self, flair: np.ndarray, lesion_mask: np.ndarray,
        n_slices: int = 9, axis: int = 2, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot multiple slices with lesion overlay."""
        lesion_slices = np.where(np.sum(lesion_mask > 0, axis=tuple(i for i in range(3) if i != axis)) > 0)[0]
        
        if len(lesion_slices) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No lesions found', ha='center', va='center')
            return fig
        
        indices = np.linspace(0, len(lesion_slices) - 1, min(n_slices, len(lesion_slices)), dtype=int)
        selected = lesion_slices[indices]
        
        n_cols = min(3, len(selected))
        n_rows = (len(selected) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        
        for idx, sl in enumerate(selected):
            self.plot_lesion_overlay(flair, lesion_mask, sl, axis, axes[idx // n_cols, idx % n_cols], f'Slice {sl}')
        
        for idx in range(len(selected), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_labeled_lesions(
        self, flair: np.ndarray, labeled_mask: np.ndarray,
        lesions: Optional[List[Lesion]] = None, num_slices: int = 6,
        axis: int = 2, save_path: Optional[str] = None, title: str = 'Labeled Lesions'
    ) -> plt.Figure:
        """Plot multiple slices with numbered lesion labels."""
        sums = np.sum(labeled_mask > 0, axis=tuple(i for i in range(3) if i != axis))
        nonzero = np.where(sums > 0)[0]
        
        if len(nonzero) == 0:
            slice_indices = [labeled_mask.shape[axis] // 2]
        elif len(nonzero) <= num_slices:
            slice_indices = list(nonzero)
        else:
            step = len(nonzero) // num_slices
            slice_indices = [nonzero[i * step] for i in range(num_slices)]
        
        n_cols = min(3, len(slice_indices))
        n_rows = (len(slice_indices) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.atleast_2d(axes)
        
        unique_labels = [l for l in np.unique(labeled_mask) if l > 0]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))
        label_colors = {l: colors[i] for i, l in enumerate(unique_labels)}
        
        for idx, sl in enumerate(slice_indices):
            ax = axes[idx // n_cols, idx % n_cols]
            slices = [slice(None)] * 3
            slices[axis] = sl
            
            ax.imshow(normalize_intensity(flair[tuple(slices)]).T, cmap='gray', origin='lower')
            lbl_slice = labeled_mask[tuple(slices)]
            
            for lbl in unique_labels:
                if np.any(lbl_slice == lbl):
                    mask = (lbl_slice == lbl).astype(float)
                    ax.imshow(np.ma.masked_where(mask == 0, mask).T, 
                             cmap=ListedColormap([label_colors[lbl]]), alpha=0.5, origin='lower')
            
            for lbl, (cx, cy) in get_2d_centroids(lbl_slice).items():
                ax.text(cx, cy, str(lbl), fontsize=9, color='white', ha='center', va='center',
                       fontweight='bold', bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.7))
            
            ax.set_title(f'Slice {sl}')
            ax.axis('off')
        
        for idx in range(len(slice_indices), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].axis('off')
        
        # Legend
        legend_items = [mpatches.Patch(color=label_colors[l], 
                        label=f'L{l}: {next((x.volume_mm3 for x in (lesions or []) if x.label == l), "?")}mm³')
                       for l in unique_labels[:10]]
        if legend_items:
            fig.legend(handles=legend_items, loc='lower center', ncol=min(5, len(legend_items)), fontsize=9)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def generate_report_figures(
        self, baseline_flair: np.ndarray, baseline_mask: np.ndarray,
        followup_flair: np.ndarray, followup_mask: np.ndarray,
        tracking_results: Dict[str, Any], baseline_labeled: np.ndarray, followup_labeled: np.ndarray,
        output_dir: str, baseline_lesions: Optional[List[Lesion]] = None,
        followup_lesions: Optional[List[Lesion]] = None
    ) -> List[str]:
        """Generate all report figures."""
        create_output_directory(output_dir)
        paths = []
        
        # Comparison
        p = os.path.join(output_dir, 'comparison.png')
        self.plot_comparison(baseline_flair, baseline_mask, followup_flair, followup_mask, save_path=p)
        plt.close(); paths.append(p)
        
        # Tracking results
        p = os.path.join(output_dir, 'tracking_results.png')
        self.plot_tracking_results(baseline_flair, followup_flair, tracking_results, baseline_labeled, followup_labeled, save_path=p)
        plt.close(); paths.append(p)
        
        # Volume changes
        p = os.path.join(output_dir, 'volume_changes.png')
        self.plot_volume_changes(tracking_results, save_path=p)
        plt.close(); paths.append(p)
        
        # Multi-slice views
        for name, flair, mask in [('baseline', baseline_flair, baseline_mask), ('followup', followup_flair, followup_mask)]:
            p = os.path.join(output_dir, f'{name}_multi_slice.png')
            self.plot_multi_slice(flair, mask, save_path=p)
            plt.close(); paths.append(p)
        
        # Labeled lesions
        p = os.path.join(output_dir, 'baseline_labeled_lesions.png')
        self.plot_labeled_lesions(baseline_flair, baseline_labeled, baseline_lesions, save_path=p, title='Baseline - Labeled')
        plt.close(); paths.append(p)
        
        p = os.path.join(output_dir, 'followup_labeled_lesions.png')
        self.plot_labeled_lesions(followup_flair, followup_labeled, followup_lesions, save_path=p, title='Follow-up - Labeled')
        plt.close(); paths.append(p)
        
        return paths
