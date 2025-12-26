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
        """Classify lesion change using user-specified status names.

        Returns one of: 'present', 'absent', 'enlarged', 'shrinking'.
        - baseline 0 and followup >0 => 'present' (new)
        - baseline >0 and followup ==0 => 'absent'
        - relative >= growth_threshold => 'enlarged'
        - relative <= -shrink_threshold => 'shrinking'
        - otherwise => 'present'
        """
        if baseline_volume == 0 and followup_volume > 0:
            return 'present'
        if baseline_volume > 0 and followup_volume == 0:
            return 'absent'
        if baseline_volume == 0:
            return 'present'
        relative = (followup_volume - baseline_volume) / baseline_volume
        if relative >= growth_threshold:
            return 'enlarged'
        if relative <= -shrink_threshold:
            return 'shrinking'
        return 'present'
    
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
        categories = {'present': 0, 'enlarged': 0, 'shrinking': 0, 'absent': len(resolved_lesions), 'merged': 0, 'split': 0}
        total_baseline = sum(c.get('baseline_volume_mm3', 0.0) for c in correspondences) + sum(l['volume_mm3'] for l in resolved_lesions)
        total_followup = sum(c.get('followup_volume_mm3', 0.0) for c in correspondences) + sum(l['volume_mm3'] for l in new_lesions)

        for corr in correspondences:
            cat = corr.get('category', 'present')
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
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
        df = pd.DataFrame(records)

        if df.empty:
            return df

        # Aggregate split rows (same baseline_label mapped to multiple followups) into single row
        # Only aggregate rows that have a non-null baseline_label
        agg_rows = []
        mask = df['baseline_label'].notnull()
        if mask.any():
            grouped = df[mask].groupby('baseline_label', sort=False)
            for baseline_label, group in grouped:
                baseline_vol = group['baseline_volume_mm3'].iloc[0]
                followup_vol_sum = group['followup_volume_mm3'].sum()
                # reconstruct followup_label(s)
                followup_labels = ','.join(map(str, group['followup_label'].astype(str).unique()))
                vol_change = followup_vol_sum - baseline_vol
                vol_pct = (vol_change / baseline_vol * 100) if baseline_vol > 0 else float('inf')
                # choose category: if any 'split' present, mark 'split'; if any 'merged' present, mark 'merged'; else pick first
                if (group['category'] == 'split').any():
                    category = 'split'
                elif (group['category'] == 'merged').any():
                    category = 'merged'
                else:
                    category = group['category'].iloc[0]
                agg_rows.append({
                    'baseline_label': baseline_label,
                    'followup_label': followup_labels,
                    'category': category,
                    'baseline_volume_mm3': baseline_vol,
                    'followup_volume_mm3': followup_vol_sum,
                    'volume_change_mm3': vol_change,
                    'volume_change_percent': vol_pct,
                })

        # Keep rows without a baseline_label (pure new lesions)
        new_rows = df[~mask].to_dict('records') if (~mask).any() else []

        final_records = agg_rows + new_rows
        return pd.DataFrame(final_records)

