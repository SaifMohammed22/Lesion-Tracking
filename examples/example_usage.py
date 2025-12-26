"""
Example usage of the lesion tracking tool.
"""

import os
from lesion_tracker import LesionTracker, LesionLabeler, LesionVisualizer


def example_full_tracking():
    """
    Example: Full lesion tracking pipeline.
    
    This demonstrates the complete workflow for tracking lesions
    between baseline and follow-up scans.
    """
    # Define paths (replace with your actual file paths)
    baseline_flair = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz"
    baseline_mask = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
    followup_flair = "/mnt/data/MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz"
    followup_mask = "/mnt/data/MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz"
    output_dir = "./tracking_results"
    
    # Initialize the tracker with custom parameters
    tracker = LesionTracker(
        overlap_threshold=0.3,      # 30% overlap to match lesions
        distance_threshold_mm=10.0,  # Max 10mm centroid distance
        growth_threshold=0.25,       # 25% volume increase = growing
        shrink_threshold=0.25,       # 25% volume decrease = shrinking
        min_lesion_voxels=3          # Minimum 3 voxels per lesion
    )
    
    # Load baseline scan
    print("Loading baseline...")
    baseline_info = tracker.load_baseline(baseline_flair, baseline_mask)
    print(f"Found {baseline_info['num_lesions']} baseline lesions")
    print(f"Total volume: {baseline_info['total_volume_mm3']:.2f} mm³")
    
    # Load follow-up scan
    print("\nLoading follow-up...")
    followup_info = tracker.load_followup(followup_flair, followup_mask)
    print(f"Found {followup_info['num_lesions']} follow-up lesions")
    print(f"Total volume: {followup_info['total_volume_mm3']:.2f} mm³")
    
    # Perform tracking
    print("\nTracking lesions...")
    results = tracker.track_lesions(baseline_flair, followup_flair)
    
    # Print results
    print("\n" + "="*50)
    print("TRACKING RESULTS")
    print("="*50)
    print(f"Matched lesions: {len(results['correspondences'])}")
    print(f"New lesions: {len(results['new_lesions'])}")
    print(f"Resolved lesions: {len(results['resolved_lesions'])}")
    
    # Print category breakdown
    print("\nLesion categories:")
    categories = {}
    for corr in results['correspondences']:
        cat = corr['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat.capitalize()}: {count}")
    
    # Generate report
    print(f"\nGenerating report in: {output_dir}")
    output_files = tracker.generate_report(output_dir)
    
    print("\nGenerated files:")
    for key, path in output_files.items():
        if isinstance(path, list):
            for p in path:
                print(f"  - {os.path.basename(p)}")
        else:
            print(f"  - {os.path.basename(path)}")


if __name__ == "__main__":
    # Run the example you want to test
    example_full_tracking()