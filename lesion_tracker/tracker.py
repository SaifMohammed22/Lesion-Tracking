"""
Main lesion tracking module.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import os
from datetime import datetime

from .labeler import LesionLabeler, Lesion
from .registration import Registration
from .metrics import LesionMetrics
from .visualization import LesionVisualizer
from .utils import load_nifti, save_nifti, get_voxel_volume, compute_overlap_ratio, compute_distance, create_output_directory


class LesionTracker:
    """Track lesions between baseline and follow-up scans."""
    
    def __init__(
        self,
        overlap_threshold: float = 0.3,
        distance_threshold_mm: float = 10.0,
        growth_threshold: float = 0.20,
        shrink_threshold: float = 0.20,
        min_lesion_voxels: int = 3,
        connectivity: int = 26
    ):
        self.overlap_threshold = overlap_threshold
        self.distance_threshold_mm = distance_threshold_mm
        self.growth_threshold = growth_threshold
        self.shrink_threshold = shrink_threshold
        
        self.labeler = LesionLabeler(connectivity=connectivity, min_lesion_voxels=min_lesion_voxels)
        self.registration = Registration(transform_type='Affine')
        self.metrics = LesionMetrics()
        self.visualizer = LesionVisualizer()
        
        # Data storage
        self.baseline_flair = self.baseline_mask = self.baseline_labeled = self.baseline_lesions = self.baseline_img = None
        self.followup_flair = self.followup_mask = self.followup_labeled = self.followup_lesions = self.followup_img = None
        self.voxel_volume_mm3 = self.voxel_spacing = self.tracking_results = None
    
    def load_baseline(self, flair_path: str, mask_path: str) -> Dict[str, Any]:
        """Load baseline FLAIR scan and segmentation mask."""
        self.baseline_flair, self.baseline_img = load_nifti(flair_path)
        self.baseline_mask, _ = load_nifti(mask_path)
        self.voxel_volume_mm3 = get_voxel_volume(self.baseline_img)
        self.voxel_spacing = tuple(self.baseline_img.header.get_zooms()[:3])
        self.baseline_labeled, self.baseline_lesions = self.labeler.label_lesions(self.baseline_mask, self.voxel_volume_mm3)
        stats = self.labeler.get_lesion_statistics(self.baseline_lesions)
        return {'status': 'success', 'timepoint': 'baseline', 'flair_shape': self.baseline_flair.shape, **stats}
    
    def load_followup(self, flair_path: str, mask_path: str) -> Dict[str, Any]:
        """Load follow-up FLAIR scan and segmentation mask."""
        self.followup_flair, self.followup_img = load_nifti(flair_path)
        self.followup_mask, _ = load_nifti(mask_path)
        if self.voxel_volume_mm3 is None:
            self.voxel_volume_mm3 = get_voxel_volume(self.followup_img)
            self.voxel_spacing = tuple(self.followup_img.header.get_zooms()[:3])
        self.followup_labeled, self.followup_lesions = self.labeler.label_lesions(self.followup_mask, self.voxel_volume_mm3)
        stats = self.labeler.get_lesion_statistics(self.followup_lesions)
        return {'status': 'success', 'timepoint': 'followup', 'flair_shape': self.followup_flair.shape, **stats}
    
    def _match_lesions(self) -> List[Tuple[int, int, float, float]]:
        """Match baseline lesions to follow-up using overlap + distance."""
        n_bl, n_fu = len(self.baseline_lesions), len(self.followup_lesions)
        
        # Compute overlap and distance matrices
        overlap = np.zeros((n_bl, n_fu))
        distance = np.zeros((n_bl, n_fu))
        for i, bl in enumerate(self.baseline_lesions):
            for j, fu in enumerate(self.followup_lesions):
                overlap[i, j] = compute_overlap_ratio(bl.mask, fu.mask)
                distance[i, j] = compute_distance(bl.centroid, fu.centroid, self.voxel_spacing)
        
        # Score: prioritize overlap, fallback to distance
        score = np.zeros((n_bl, n_fu))
        for i in range(n_bl):
            for j in range(n_fu):
                if overlap[i, j] >= self.overlap_threshold:
                    score[i, j] = overlap[i, j]
                elif distance[i, j] <= self.distance_threshold_mm:
                    score[i, j] = 0.1 * (1 - distance[i, j] / self.distance_threshold_mm)
        
        # Greedy matching
        matched_bl, matched_fu, matches = set(), set(), []
        while True:
            best_score, best_pair = 0, None
            for i in range(n_bl):
                if i in matched_bl: continue
                for j in range(n_fu):
                    if j in matched_fu: continue
                    if score[i, j] > best_score:
                        best_score, best_pair = score[i, j], (i, j)
            if best_pair is None or best_score <= 0:
                break
            i, j = best_pair
            matches.append((i, j, overlap[i, j], distance[i, j]))
            matched_bl.add(i)
            matched_fu.add(j)
        return matches
    
    def track_lesions(self, image1_path: str, image2_path: str, check_registration: bool = True) -> Dict[str, Any]:
        """Perform lesion tracking between baseline and follow-up."""
        if self.baseline_lesions is None or self.followup_lesions is None:
            raise ValueError("Both baseline and follow-up must be loaded first")
        
        registration_info = self.registration.check_alignment(image1_path, image2_path) if check_registration else None
        matches = self._match_lesions()
        
        correspondences, matched_bl_labels, matched_fu_labels = [], set(), set()
        for bl_idx, fu_idx, ovlp, dist in matches:
            bl, fu = self.baseline_lesions[bl_idx], self.followup_lesions[fu_idx]
            matched_bl_labels.add(bl.label)
            matched_fu_labels.add(fu.label)
            vol_change = fu.volume_mm3 - bl.volume_mm3
            vol_pct = (vol_change / bl.volume_mm3 * 100) if bl.volume_mm3 > 0 else float('inf')
            category = self.metrics.classify_lesion_change(bl.volume_mm3, fu.volume_mm3, self.growth_threshold, self.shrink_threshold)
            correspondences.append({
                'baseline_label': bl.label, 'followup_label': fu.label,
                'baseline_volume_mm3': bl.volume_mm3, 'followup_volume_mm3': fu.volume_mm3,
                'volume_change_mm3': vol_change, 'volume_change_percent': vol_pct,
                'category': category, 'overlap_ratio': ovlp, 'centroid_distance_mm': dist,
            })
        
        new_lesions = [{'label': l.label, 'volume_mm3': l.volume_mm3, 'centroid': l.centroid} 
                       for l in self.followup_lesions if l.label not in matched_fu_labels]
        resolved_lesions = [{'label': l.label, 'volume_mm3': l.volume_mm3, 'centroid': l.centroid}
                           for l in self.baseline_lesions if l.label not in matched_bl_labels]
        
        self.tracking_results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {'overlap_threshold': self.overlap_threshold, 'distance_threshold_mm': self.distance_threshold_mm,
                          'growth_threshold': self.growth_threshold, 'shrink_threshold': self.shrink_threshold},
            'registration_check': registration_info,
            'baseline_lesion_count': len(self.baseline_lesions), 'followup_lesion_count': len(self.followup_lesions),
            'correspondences': correspondences, 'new_lesions': new_lesions, 'resolved_lesions': resolved_lesions
        }
        return self.tracking_results
    
    def generate_report(self, output_dir: str, include_visualizations: bool = True) -> Dict[str, str]:
        """Generate comprehensive report with results and visualizations."""
        if self.tracking_results is None:
            raise ValueError("Must run track_lesions() before generating report")
        
        create_output_directory(output_dir)
        output_files = {}
        
        # JSON results
        with open(os.path.join(output_dir, 'tracking_results.json'), 'w') as f:
            json.dump(self.tracking_results, f, indent=2, default=str)
        output_files['json'] = os.path.join(output_dir, 'tracking_results.json')
        
        # Summary
        summary = self.metrics.compute_tracking_summary(self.tracking_results)
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['summary'] = os.path.join(output_dir, 'summary.json')
        
        # CSV
        df = self.metrics.generate_lesion_dataframe(self.tracking_results)
        df.to_csv(os.path.join(output_dir, 'lesion_details.csv'), index=False)
        output_files['csv'] = os.path.join(output_dir, 'lesion_details.csv')
        
        # Labeled masks
        save_nifti(self.baseline_labeled, self.baseline_img, os.path.join(output_dir, 'baseline_labeled.nii.gz'), dtype=np.int16)
        save_nifti(self.followup_labeled, self.followup_img, os.path.join(output_dir, 'followup_labeled.nii.gz'), dtype=np.int16)
        
        # Visualizations
        if include_visualizations:
            output_files['figures'] = self.visualizer.generate_report_figures(
                self.baseline_flair, self.baseline_mask, self.followup_flair, self.followup_mask,
                self.tracking_results, self.baseline_labeled, self.followup_labeled, output_dir,
                baseline_lesions=self.baseline_lesions, followup_lesions=self.followup_lesions)
        
        # Text report
        self._write_text_report(os.path.join(output_dir, 'report.txt'), summary)
        output_files['report'] = os.path.join(output_dir, 'report.txt')
        return output_files
    
    def _write_text_report(self, filepath: str, summary: Dict[str, Any]) -> None:
        """Write a human-readable text report."""
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\nLESION TRACKING REPORT\n" + "=" * 60 + "\n\n")
            f.write(f"Generated: {self.tracking_results['timestamp']}\n\n")
            f.write(f"Baseline: {summary['baseline_lesion_count']} lesions, Follow-up: {summary['followup_lesion_count']} lesions\n")
            f.write(f"Matched: {summary['matched_lesions']}\n\n")
            for cat, count in summary['categories'].items():
                f.write(f"  {cat.capitalize()}: {count}\n")
            f.write(f"\nVolume change: {summary['volume_change']['percent_change']:.1f}%\n")

