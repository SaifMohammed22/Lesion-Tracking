"""
Command-line interface for lesion tracking.
"""

import click
import os
import sys
import yaml
from typing import Optional

from .tracker import LesionTracker, track_lesions_from_files
from .labeler import LesionLabeler
from .utils import load_nifti, save_nifti, get_voxel_volume


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    MRI Lesion Tracking Tool
    
    Track and label lesions in baseline and follow-up MRI scans.
    Designed for the MSLesSeg dataset and similar MS lesion datasets.
    """
    pass


@cli.command()
@click.option('--baseline-flair', '-bf', required=True, type=click.Path(exists=True),
              help='Path to baseline FLAIR NIfTI file')
@click.option('--baseline-mask', '-bm', required=True, type=click.Path(exists=True),
              help='Path to baseline lesion mask NIfTI file')
@click.option('--followup-flair', '-ff', required=True, type=click.Path(exists=True),
              help='Path to follow-up FLAIR NIfTI file')
@click.option('--followup-mask', '-fm', required=True, type=click.Path(exists=True),
              help='Path to follow-up lesion mask NIfTI file')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for results')
@click.option('--overlap-threshold', default=0.3, type=float,
              help='Minimum overlap ratio for lesion matching (default: 0.3)')
@click.option('--distance-threshold', default=10.0, type=float,
              help='Maximum centroid distance in mm (default: 10.0)')
@click.option('--growth-threshold', default=0.20, type=float,
              help='Relative growth threshold (default: 0.20)')
@click.option('--shrink-threshold', default=0.20, type=float,
              help='Relative shrink threshold (default: 0.20)')
@click.option('--min-lesion-voxels', default=3, type=int,
              help='Minimum voxels to consider as lesion (default: 3)')
@click.option('--no-viz', is_flag=True, default=False,
              help='Skip visualization generation')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
def track(baseline_flair, baseline_mask, followup_flair, followup_mask, output,
          overlap_threshold, distance_threshold, growth_threshold, shrink_threshold,
          min_lesion_voxels, no_viz, config):
    """
    Track lesions between baseline and follow-up scans.
    
    Identifies matching lesions, new lesions, and resolved lesions.
    Classifies changes as stable, growing, or shrinking.
    
    Example:
        lesion-tracker track -bf baseline.nii.gz -bm baseline_mask.nii.gz 
                            -ff followup.nii.gz -fm followup_mask.nii.gz -o ./results
    """
    # Load config if provided
    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
            overlap_threshold = cfg.get('overlap_threshold', overlap_threshold)
            distance_threshold = cfg.get('distance_threshold_mm', distance_threshold)
            growth_threshold = cfg.get('growth_threshold', growth_threshold)
            shrink_threshold = cfg.get('shrink_threshold', shrink_threshold)
            min_lesion_voxels = cfg.get('min_lesion_voxels', min_lesion_voxels)
    
    click.echo("=" * 50)
    click.echo("MRI Lesion Tracking")
    click.echo("=" * 50)
    
    # Initialize tracker
    tracker = LesionTracker(
        overlap_threshold=overlap_threshold,
        distance_threshold_mm=distance_threshold,
        growth_threshold=growth_threshold,
        shrink_threshold=shrink_threshold,
        min_lesion_voxels=min_lesion_voxels
    )
    
    # Load baseline
    click.echo("\nLoading baseline scan...")
    baseline_info = tracker.load_baseline(baseline_flair, baseline_mask)
    click.echo(f"  Found {baseline_info['num_lesions']} lesions")
    click.echo(f"  Total volume: {baseline_info['total_volume_mm3']:.2f} mm³")
    
    # Load follow-up
    click.echo("\nLoading follow-up scan...")
    followup_info = tracker.load_followup(followup_flair, followup_mask)
    click.echo(f"  Found {followup_info['num_lesions']} lesions")
    click.echo(f"  Total volume: {followup_info['total_volume_mm3']:.2f} mm³")
    
    # Perform tracking
    click.echo("\nTracking lesions...")
    results = tracker.track_lesions()
    
    # Print summary
    click.echo("\n" + "-" * 40)
    click.echo("TRACKING RESULTS")
    click.echo("-" * 40)
    click.echo(f"Matched lesions: {len(results['correspondences'])}")
    click.echo(f"New lesions: {len(results['new_lesions'])}")
    click.echo(f"Resolved lesions: {len(results['resolved_lesions'])}")
    
    # Count categories
    categories = {}
    for corr in results['correspondences']:
        cat = corr['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    click.echo("\nLesion categories:")
    for cat, count in sorted(categories.items()):
        click.echo(f"  {cat.capitalize()}: {count}")
    
    # Generate report
    click.echo(f"\nGenerating report in: {output}")
    output_files = tracker.generate_report(output, include_visualizations=not no_viz)
    
    click.echo("\nGenerated files:")
    for key, path in output_files.items():
        if isinstance(path, list):
            for p in path:
                click.echo(f"  - {os.path.basename(p)}")
        else:
            click.echo(f"  - {os.path.basename(path)}")
    
    click.echo("\n" + "=" * 50)
    click.echo("Tracking complete!")
    click.echo("=" * 50)


@cli.command()
@click.option('--flair', '-f', type=click.Path(exists=True),
              help='Path to FLAIR NIfTI file (optional)')
@click.option('--mask', '-m', required=True, type=click.Path(exists=True),
              help='Path to lesion mask NIfTI file')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output path for labeled mask (.nii.gz)')
@click.option('--min-lesion-voxels', default=3, type=int,
              help='Minimum voxels to consider as lesion (default: 3)')
@click.option('--connectivity', default=26, type=click.Choice(['6', '18', '26']),
              help='Connectivity for component analysis (default: 26)')
@click.option('--stats-output', '-s', type=click.Path(),
              help='Output path for lesion statistics (JSON)')
def label(flair, mask, output, min_lesion_voxels, connectivity, stats_output):
    """
    Label individual lesions in a single scan.
    
    Uses connected component analysis to identify separate lesions
    and assigns unique integer labels.
    
    Example:
        lesion-tracker label -m mask.nii.gz -o labeled_mask.nii.gz
    """
    click.echo("=" * 50)
    click.echo("Lesion Labeling")
    click.echo("=" * 50)
    
    # Initialize labeler
    labeler = LesionLabeler(
        connectivity=int(connectivity),
        min_lesion_voxels=min_lesion_voxels
    )
    
    # Load mask
    click.echo(f"\nLoading mask: {mask}")
    mask_data, mask_img = load_nifti(mask)
    voxel_volume = get_voxel_volume(mask_img)
    
    # Label lesions
    click.echo("Labeling lesions...")
    labeled_mask, lesions = labeler.label_lesions(mask_data, voxel_volume_mm3=voxel_volume)
    
    # Save labeled mask
    click.echo(f"Saving labeled mask: {output}")
    save_nifti(labeled_mask, mask_img, output, dtype='int16')
    
    # Get statistics
    stats = labeler.get_lesion_statistics(lesions)
    
    click.echo("\n" + "-" * 40)
    click.echo("STATISTICS")
    click.echo("-" * 40)
    click.echo(f"Number of lesions: {stats['num_lesions']}")
    click.echo(f"Total volume: {stats['total_volume_mm3']:.2f} mm³")
    click.echo(f"Mean volume: {stats['mean_volume_mm3']:.2f} mm³")
    click.echo(f"Min volume: {stats['min_volume_mm3']:.2f} mm³")
    click.echo(f"Max volume: {stats['max_volume_mm3']:.2f} mm³")
    
    # Save statistics if requested
    if stats_output:
        import json
        lesion_details = [l.to_dict() for l in lesions]
        output_data = {
            'statistics': stats,
            'lesions': lesion_details
        }
        with open(stats_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\nStatistics saved to: {stats_output}")
    
    click.echo("\n" + "=" * 50)
    click.echo("Labeling complete!")
    click.echo("=" * 50)


@cli.command()
@click.option('--flair', '-f', required=True, type=click.Path(exists=True),
              help='Path to FLAIR NIfTI file')
@click.option('--mask', '-m', required=True, type=click.Path(exists=True),
              help='Path to lesion mask NIfTI file')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for visualizations')
@click.option('--slices', '-n', default=9, type=int,
              help='Number of slices to display (default: 9)')
def visualize(flair, mask, output, slices):
    """
    Generate visualizations for a single scan.
    
    Creates multi-slice views with lesion overlays.
    
    Example:
        lesion-tracker visualize -f flair.nii.gz -m mask.nii.gz -o ./viz
    """
    from .visualization import LesionVisualizer
    from .utils import create_output_directory
    import matplotlib.pyplot as plt
    
    click.echo("=" * 50)
    click.echo("Lesion Visualization")
    click.echo("=" * 50)
    
    # Load data
    click.echo(f"\nLoading FLAIR: {flair}")
    flair_data, _ = load_nifti(flair)
    
    click.echo(f"Loading mask: {mask}")
    mask_data, _ = load_nifti(mask)
    
    # Create visualizer
    visualizer = LesionVisualizer()
    
    # Create output directory
    create_output_directory(output)
    
    # Generate multi-slice view
    click.echo("\nGenerating visualizations...")
    
    for axis, axis_name in [(0, 'sagittal'), (1, 'coronal'), (2, 'axial')]:
        output_path = os.path.join(output, f'multi_slice_{axis_name}.png')
        visualizer.plot_multi_slice(flair_data, mask_data, n_slices=slices, 
                                    axis=axis, save_path=output_path)
        plt.close()
        click.echo(f"  Saved: {os.path.basename(output_path)}")
    
    click.echo("\n" + "=" * 50)
    click.echo("Visualization complete!")
    click.echo("=" * 50)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='./config.yaml',
              help='Output path for config file')
def init_config(output):
    """
    Generate a default configuration file.
    
    Example:
        lesion-tracker init-config -o my_config.yaml
    """
    default_config = {
        'tracking': {
            'overlap_threshold': 0.3,
            'distance_threshold_mm': 10.0,
            'growth_threshold': 0.20,
            'shrink_threshold': 0.20,
        },
        'labeling': {
            'min_lesion_voxels': 3,
            'connectivity': 26,
            'sort_by_size': True,
        },
        'visualization': {
            'figsize': [12, 8],
            'dpi': 150,
            'n_slices': 9,
        },
        'output': {
            'save_labeled_masks': True,
            'generate_visualizations': True,
            'save_csv': True,
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    click.echo(f"Configuration file saved to: {output}")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
