"""
Lesion metrics module for volumetric and statistical analysis.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from .labeler import Lesion


class LesionMetrics:
    """Compute lesion metrics and statistics."""
    
    def compute_volume_change(self, baseline_volume: float, followup_volume: float) -> Dict[str, float]:
        """Compute volume change between baseline and follow-up."""
        absolute = followup_volume - baseline_volume
        relative = (absolute / baseline_volume) if baseline_volume > 0 else (float('inf') if followup_volume > 0 else 0.0)
        return {'absolute_change_mm3': absolute, 'relative_change': relative, 'percent_change': relative * 100}
    
    def classify_lesion_change(
        self, baseline_volume: float, followup_volume: float,
        growth_threshold: float = 0.20, shrink_threshold: float = 0.20
    ) -> str:
        """Classify lesion change: 'new', 'resolved', 'growing', 'shrinking', or 'stable'."""
        if baseline_volume == 0 and followup_volume > 0:
            return 'new'
        if baseline_volume > 0 and followup_volume == 0:
            return 'resolved'
        if baseline_volume == 0:
            return 'absent'
        relative = (followup_volume - baseline_volume) / baseline_volume
        if relative > growth_threshold:
            return 'growing'
        if relative < -shrink_threshold:
            return 'shrinking'
        return 'stable'
    
    def compute_lesion_load(self, lesions: List[Lesion]) -> Dict[str, float]:
        """Compute total lesion load statistics."""
        if not lesions:
            return {'total_lesion_count': 0, 'total_lesion_volume_mm3': 0.0}
        volumes = [l.volume_mm3 for l in lesions]
        return {
            'total_lesion_count': len(lesions),
            'total_lesion_volume_mm3': sum(volumes),
            'total_lesion_volume_ml': sum(volumes) / 1000.0,
            'mean_lesion_volume_mm3': np.mean(volumes),
        }
    
    def compute_tracking_summary(self, tracking_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics from tracking results."""
        correspondences = tracking_results.get('correspondences', [])
        new_lesions = tracking_results.get('new_lesions', [])
        resolved_lesions = tracking_results.get('resolved_lesions', [])
        
        categories = {'stable': 0, 'growing': 0, 'shrinking': 0, 'new': len(new_lesions), 'resolved': len(resolved_lesions)}
        total_baseline = sum(c['baseline_volume_mm3'] for c in correspondences) + sum(l['volume_mm3'] for l in resolved_lesions)
        total_followup = sum(c['followup_volume_mm3'] for c in correspondences) + sum(l['volume_mm3'] for l in new_lesions)
        
        for corr in correspondences:
            categories[corr['category']] += 1
        
        return {
            'baseline_lesion_count': len(correspondences) + len(resolved_lesions),
            'followup_lesion_count': len(correspondences) + len(new_lesions),
            'matched_lesions': len(correspondences),
            'categories': categories,
            'total_baseline_volume_mm3': total_baseline,
            'total_followup_volume_mm3': total_followup,
            'volume_change': self.compute_volume_change(total_baseline, total_followup)
        }
    
    def generate_lesion_dataframe(self, tracking_results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a pandas DataFrame with detailed lesion information."""
        records = []
        for corr in tracking_results.get('correspondences', []):
            records.append({
                'baseline_label': corr['baseline_label'], 'followup_label': corr['followup_label'],
                'category': corr['category'],
                'baseline_volume_mm3': corr['baseline_volume_mm3'], 'followup_volume_mm3': corr['followup_volume_mm3'],
                'volume_change_mm3': corr['volume_change_mm3'], 'volume_change_percent': corr['volume_change_percent'],
            })
        for info in tracking_results.get('new_lesions', []):
            records.append({
                'baseline_label': None, 'followup_label': info['label'], 'category': 'new',
                'baseline_volume_mm3': 0.0, 'followup_volume_mm3': info['volume_mm3'],
                'volume_change_mm3': info['volume_mm3'], 'volume_change_percent': float('inf'),
            })
        for info in tracking_results.get('resolved_lesions', []):
            records.append({
                'baseline_label': info['label'], 'followup_label': None, 'category': 'resolved',
                'baseline_volume_mm3': info['volume_mm3'], 'followup_volume_mm3': 0.0,
                'volume_change_mm3': -info['volume_mm3'], 'volume_change_percent': -100.0,
            })
        return pd.DataFrame(records)

