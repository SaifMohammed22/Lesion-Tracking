"""
Lesion Tracking - Simplified Core Module

Track MS lesions between two timepoints (baseline and follow-up).
Detects: new, disappeared, enlarged, shrunk, and stable lesions.
"""

import os
import json
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Dict, Any, Optional, Tuple

try:
    import ants
except ImportError:
    ants = None

from utils import load_nifti, save_nifti


# =============================================================================
# Registration
# =============================================================================


def register_to_baseline(
    baseline_path: str, followup_path: str, transform_type: str = "Affine"
) -> Dict[str, Any]:
    """
    Register follow-up image to baseline space using ANTs.

    Returns dict with 'transforms' and 'registered' ANTs image.
    """
    if ants is None:
        raise ImportError(
            "ANTsPy is required for registration. Install with: pip install antspyx"
        )

    fixed = ants.image_read(baseline_path)
    moving = ants.image_read(followup_path)
    result = ants.registration(
        fixed=fixed, moving=moving, type_of_transform=transform_type
    )

    return {
        "transforms": result["fwdtransforms"],
        "registered": result["warpedmovout"],
        "fixed": fixed,
    }


def apply_transform(
    image_path: str,
    reference: "ants.ANTsImage",
    transforms: list,
    interpolation: str = "nearestNeighbor",
) -> np.ndarray:
    """Apply transforms to an image (e.g., lesion mask)."""
    image = ants.image_read(image_path)
    transformed = ants.apply_transforms(
        fixed=reference,
        moving=image,
        transformlist=transforms,
        interpolator=interpolation,
    )
    return transformed.numpy()


# =============================================================================
# Lesion Tracking Core
# =============================================================================


def label_lesions(mask: np.ndarray, min_size: int = 3) -> Tuple[np.ndarray, int]:
    """
    Label connected components in mask, filter by minimum size.
    Returns (labeled_array, num_lesions).
    """
    struct = ndimage.generate_binary_structure(3, 3)  # 26-connectivity
    labeled, num = ndimage.label(mask > 0, structure=struct)

    # Filter small lesions and relabel sequentially
    filtered = np.zeros_like(labeled)
    new_label = 1

    for old_label in range(1, num + 1):
        component = labeled == old_label
        if component.sum() >= min_size:
            filtered[component] = new_label
            new_label += 1

    return filtered, new_label - 1


def track_lesions(
    baseline_mask: np.ndarray,
    followup_mask: np.ndarray,
    min_lesion_size: int = 3,
    change_threshold: float = 0.2,
    max_distance_mm: float = 15.0,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Robust lesion tracking using centroid distance + overlap + size similarity.

    This algorithm is more robust to registration errors than pure overlap matching.

    Args:
        baseline_mask: Binary lesion mask at baseline
        followup_mask: Binary lesion mask at follow-up (registered to baseline space)
        min_lesion_size: Minimum voxels to count as a lesion
        change_threshold: Volume change ratio to classify as enlarged/shrunk (default 20%)
        max_distance_mm: Maximum distance (mm) to consider lesions as matching (default 15mm)
        voxel_spacing: Voxel size in mm (x, y, z)

    Returns:
        Dict with tracking results including labeled masks and statistics
    """
    # Label lesions in each timepoint
    bl_labeled, num_bl = label_lesions(baseline_mask, min_lesion_size)
    fu_labeled, num_fu = label_lesions(followup_mask, min_lesion_size)

    # Get lesion properties (centroid, volume) for each lesion
    bl_props = _get_lesion_properties(bl_labeled, num_bl, voxel_spacing)
    fu_props = _get_lesion_properties(fu_labeled, num_fu, voxel_spacing)

    # Build matching score matrix
    # Score = weighted combination of: overlap, distance, size similarity
    matches = _match_lesions(
        bl_labeled, fu_labeled, bl_props, fu_props, max_distance_mm, voxel_spacing
    )

    # Classify lesions based on matches
    stable, enlarged, shrunk, disappeared = [], [], [], []
    matched_fu_ids = set()

    for bl_id, match_info in matches.items():
        if match_info is None:
            # No match found - lesion disappeared
            disappeared.append(
                {
                    "id": bl_id,
                    "volume": int(bl_props[bl_id]["volume"]),
                    "centroid": bl_props[bl_id]["centroid_mm"],
                }
            )
        else:
            fu_id = match_info["fu_id"]
            matched_fu_ids.add(fu_id)

            bl_vol = bl_props[bl_id]["volume"]
            fu_vol = fu_props[fu_id]["volume"]
            change_ratio = (fu_vol - bl_vol) / bl_vol if bl_vol > 0 else 0

            info = {
                "id": bl_id,
                "baseline_volume": int(bl_vol),
                "followup_volume": int(fu_vol),
                "change_ratio": float(change_ratio),
                "distance_mm": float(match_info["distance_mm"]),
                "overlap_ratio": float(match_info["overlap_ratio"]),
            }

            if change_ratio > change_threshold:
                enlarged.append(info)
            elif change_ratio < -change_threshold:
                shrunk.append(info)
            else:
                stable.append(info)

    # Find new lesions (not matched to any baseline lesion)
    new_lesions = []
    next_id = num_bl + 1

    for fu_id in range(1, num_fu + 1):
        if fu_id not in matched_fu_ids:
            new_lesions.append(
                {
                    "id": next_id,
                    "volume": int(fu_props[fu_id]["volume"]),
                    "centroid": fu_props[fu_id]["centroid_mm"],
                }
            )
            next_id += 1

    # Create tracked follow-up labels
    fu_tracked = np.zeros_like(fu_labeled)

    # Assign baseline IDs to matched lesions
    for bl_id, match_info in matches.items():
        if match_info is not None:
            fu_id = match_info["fu_id"]
            fu_tracked[fu_labeled == fu_id] = bl_id

    # Assign new IDs to new lesions
    next_id = num_bl + 1
    for fu_id in range(1, num_fu + 1):
        if fu_id not in matched_fu_ids:
            fu_tracked[fu_labeled == fu_id] = next_id
            next_id += 1

    return {
        "baseline_labeled": bl_labeled,
        "followup_labeled": fu_tracked,
        "stable": stable,
        "enlarged": enlarged,
        "shrunk": shrunk,
        "disappeared": disappeared,
        "new": new_lesions,
        "summary": {
            "num_baseline": num_bl,
            "num_followup": num_fu,
            "num_stable": len(stable),
            "num_enlarged": len(enlarged),
            "num_shrunk": len(shrunk),
            "num_disappeared": len(disappeared),
            "num_new": len(new_lesions),
        },
    }


def _get_lesion_properties(
    labeled: np.ndarray, num_lesions: int, voxel_spacing: tuple
) -> Dict:
    """Extract centroid and volume for each lesion."""
    props = {}
    for lid in range(1, num_lesions + 1):
        region = labeled == lid
        volume = region.sum()

        # Centroid in voxel coordinates
        coords = np.array(np.where(region))
        centroid_vox = coords.mean(axis=1)

        # Centroid in mm
        centroid_mm = tuple(centroid_vox[i] * voxel_spacing[i] for i in range(3))

        props[lid] = {
            "volume": volume,
            "centroid_vox": tuple(centroid_vox),
            "centroid_mm": centroid_mm,
        }
    return props


def _match_lesions(
    bl_labeled: np.ndarray,
    fu_labeled: np.ndarray,
    bl_props: Dict,
    fu_props: Dict,
    max_distance_mm: float,
    voxel_spacing: tuple,
) -> Dict:
    """
    Match baseline lesions to follow-up lesions using multiple criteria.

    Matching criteria (in priority order):
    1. Overlap - if lesions overlap, they're likely the same
    2. Distance - closest unmatched lesion within max_distance_mm
    3. Size similarity - prefer similar sized lesions
    """
    matches = {}  # bl_id -> {fu_id, distance_mm, overlap_ratio} or None
    used_fu_ids = set()

    # First pass: match by overlap (most reliable)
    for bl_id in bl_props:
        bl_region = bl_labeled == bl_id

        # Find overlapping follow-up lesions
        overlap_ids = np.unique(fu_labeled[bl_region])
        overlap_ids = [fid for fid in overlap_ids if fid > 0 and fid not in used_fu_ids]

        if overlap_ids:
            # Find best overlap
            best_fu_id = None
            best_overlap = 0
            best_overlap_ratio = 0.0

            for fu_id in overlap_ids:
                fu_region = fu_labeled == fu_id
                overlap = (bl_region & fu_region).sum()
                union = (bl_region | fu_region).sum()
                overlap_ratio = overlap / union if union > 0 else 0

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_fu_id = fu_id
                    best_overlap_ratio = overlap_ratio

            if best_fu_id is not None:
                # Calculate distance for reporting
                bl_cent = np.array(bl_props[bl_id]["centroid_mm"])
                fu_cent = np.array(fu_props[best_fu_id]["centroid_mm"])
                distance = np.sqrt(np.sum((bl_cent - fu_cent) ** 2))

                matches[bl_id] = {
                    "fu_id": best_fu_id,
                    "distance_mm": distance,
                    "overlap_ratio": best_overlap_ratio,
                    "match_type": "overlap",
                }
                used_fu_ids.add(best_fu_id)

    # Second pass: match remaining by distance (for registration errors)
    unmatched_bl = [bl_id for bl_id in bl_props if bl_id not in matches]
    unmatched_fu = [fu_id for fu_id in fu_props if fu_id not in used_fu_ids]

    for bl_id in unmatched_bl:
        bl_cent = np.array(bl_props[bl_id]["centroid_mm"])
        bl_vol = bl_props[bl_id]["volume"]

        best_fu_id = None
        best_score = float("inf")
        best_distance = 0.0

        for fu_id in unmatched_fu:
            if fu_id in used_fu_ids:
                continue

            fu_cent = np.array(fu_props[fu_id]["centroid_mm"])
            fu_vol = fu_props[fu_id]["volume"]

            # Distance in mm
            distance = np.sqrt(np.sum((bl_cent - fu_cent) ** 2))

            if distance > max_distance_mm:
                continue

            # Size similarity (penalize large volume changes)
            vol_ratio = max(bl_vol, fu_vol) / max(min(bl_vol, fu_vol), 1)
            if vol_ratio > 5:  # More than 5x size change is suspicious
                continue

            # Combined score (lower is better)
            score = distance + (vol_ratio - 1) * 5  # Penalize size mismatch

            if score < best_score:
                best_score = score
                best_fu_id = fu_id
                best_distance = distance

        if best_fu_id is not None:
            matches[bl_id] = {
                "fu_id": best_fu_id,
                "distance_mm": best_distance,
                "overlap_ratio": 0.0,  # No overlap
                "match_type": "distance",
            }
            used_fu_ids.add(best_fu_id)
        else:
            matches[bl_id] = None  # Disappeared

    return matches


# =============================================================================
# Main Pipeline
# =============================================================================


def run_tracking(
    baseline_flair: str,
    baseline_mask: str,
    followup_flair: str,
    followup_mask: str,
    output_dir: Optional[str] = None,
    registration_type: str = "Affine",
    min_lesion_size: int = 3,
    change_threshold: float = 0.2,
    save_visualization: bool = True,
    num_slices: int = 1,
) -> Dict[str, Any]:
    """
    Main entry point - track lesions between two timepoints.

    Args:
        baseline_flair: Path to baseline FLAIR image
        baseline_mask: Path to baseline lesion mask
        followup_flair: Path to follow-up FLAIR image
        followup_mask: Path to follow-up lesion mask
        output_dir: Optional directory to save results
        registration_type: ANTs registration type ('Rigid', 'Affine', 'SyN')
        min_lesion_size: Minimum lesion size in voxels
        change_threshold: Volume change ratio for enlarged/shrunk classification
        save_visualization: Whether to save PNG visualization (default: True)
        num_slices: Number of slices in visualization (1=single best, >1=grid)

    Returns:
        Dict with tracking results and statistics
    """
    print("=" * 60)
    print("Lesion Tracking Pipeline")
    print("=" * 60)

    # 1. Load baseline mask
    print("\n[1/4] Loading data...")
    bl_mask_data, bl_mask_img = load_nifti(baseline_mask)
    bl_mask_data = (bl_mask_data > 0).astype(np.uint8)

    # 2. Register follow-up to baseline
    print(f"[2/4] Registering follow-up to baseline ({registration_type})...")
    reg_result = register_to_baseline(baseline_flair, followup_flair, registration_type)

    # 3. Apply transform to follow-up mask
    print("[3/4] Transforming follow-up mask...")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_mask_path = tmp.name

    try:
        # Save follow-up mask temporarily for ANTs
        fu_mask_data, _ = load_nifti(followup_mask)
        nib.save(
            nib.Nifti1Image(fu_mask_data, nib.load(followup_flair).affine),
            tmp_mask_path,
        )

        # Apply registration transform
        fu_mask_registered = apply_transform(
            tmp_mask_path,
            reg_result["fixed"],
            reg_result["transforms"],
            interpolation="nearestNeighbor",
        )
        fu_mask_registered = (fu_mask_registered > 0.5).astype(np.uint8)
    finally:
        if os.path.exists(tmp_mask_path):
            os.remove(tmp_mask_path)

    # 4. Track lesions
    print("[4/4] Tracking lesions...")
    voxel_spacing = bl_mask_img.header.get_zooms()[:3]
    results = track_lesions(
        bl_mask_data,
        fu_mask_registered,
        min_lesion_size=min_lesion_size,
        change_threshold=change_threshold,
        max_distance_mm=15.0,
        voxel_spacing=voxel_spacing,
    )

    # Add voxel volume for mm3 calculations
    voxel_vol = np.prod(voxel_spacing)
    results["voxel_volume_mm3"] = float(voxel_vol)

    # Print summary
    _print_summary(results)

    # Save results if output_dir specified
    if output_dir:
        _save_results(
            results,
            bl_mask_img,
            fu_mask_registered,
            output_dir,
            baseline_flair,
            followup_flair,
            save_visualization,
            num_slices,
        )

    return results


def _print_summary(results: Dict[str, Any]):
    """Print tracking results summary."""
    s = results["summary"]
    print("\n" + "=" * 60)
    print("TRACKING RESULTS")
    print("=" * 60)
    print(f"Baseline lesions:  {s['num_baseline']}")
    print(f"Follow-up lesions: {s['num_followup']}")
    print(f"")
    print(f"  Stable:      {s['num_stable']}")
    print(f"  Enlarged:    {s['num_enlarged']}")
    print(f"  Shrunk:      {s['num_shrunk']}")
    print(f"  Disappeared: {s['num_disappeared']}")
    print(f"  New:         {s['num_new']}")
    print("=" * 60)


def _save_results(
    results: Dict[str, Any],
    reference_img: nib.Nifti1Image,
    fu_mask_registered: np.ndarray,
    output_dir: str,
    baseline_flair: str,
    followup_flair: str,
    save_visualization: bool = True,
    num_slices: int = 1,
):
    """Save tracking results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    # Save labeled masks (NIfTI)
    save_nifti(
        results["baseline_labeled"],
        reference_img,
        os.path.join(output_dir, "baseline_lesions_labeled.nii.gz"),
    )
    save_nifti(
        results["followup_labeled"],
        reference_img,
        os.path.join(output_dir, "followup_lesions_labeled.nii.gz"),
    )

    # Save registered follow-up mask
    save_nifti(
        fu_mask_registered,
        reference_img,
        os.path.join(output_dir, "followup_mask_registered.nii.gz"),
    )

    # Save statistics as JSON
    stats = {
        "summary": results["summary"],
        "stable": results["stable"],
        "enlarged": results["enlarged"],
        "shrunk": results["shrunk"],
        "disappeared": results["disappeared"],
        "new": results["new"],
        "voxel_volume_mm3": results["voxel_volume_mm3"],
    }
    with open(os.path.join(output_dir, "tracking_results.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("  - baseline_lesions_labeled.nii.gz")
    print("  - followup_lesions_labeled.nii.gz")
    print("  - followup_mask_registered.nii.gz")
    print("  - tracking_results.json")

    # Save PNG visualization
    if save_visualization:
        try:
            from visualization import visualize_tracking

            png_path = os.path.join(output_dir, "tracking_visualization.png")
            visualize_tracking(
                flair_baseline_path=baseline_flair,
                flair_followup_path=followup_flair,
                baseline_labeled_path=os.path.join(
                    output_dir, "baseline_lesions_labeled.nii.gz"
                ),
                followup_labeled_path=os.path.join(
                    output_dir, "followup_lesions_labeled.nii.gz"
                ),
                num_slices=num_slices,
                save_path=png_path,
                show=False,
            )
            print("  - tracking_visualization.png")
        except ImportError as e:
            print(e)


# =============================================================================
# Convenience function for MSLesSeg dataset format
# =============================================================================


def track_mslesseg(
    baseline_dir: str,
    followup_dir: str,
    output_dir: str,
    patient_id: str = "P1",
    baseline_tp: str = "T1",
    followup_tp: str = "T2",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for MSLesSeg dataset format.

    Expects files named: {patient_id}_{timepoint}_FLAIR.nii.gz and {patient_id}_{timepoint}_MASK.nii.gz
    """
    return run_tracking(
        baseline_flair=os.path.join(
            baseline_dir, f"{patient_id}_{baseline_tp}_FLAIR.nii.gz"
        ),
        baseline_mask=os.path.join(
            baseline_dir, f"{patient_id}_{baseline_tp}_MASK.nii.gz"
        ),
        followup_flair=os.path.join(
            followup_dir, f"{patient_id}_{followup_tp}_FLAIR.nii.gz"
        ),
        followup_mask=os.path.join(
            followup_dir, f"{patient_id}_{followup_tp}_MASK.nii.gz"
        ),
        output_dir=output_dir,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage
    results = track_mslesseg(
        baseline_dir="MSLesSeg Dataset/train/P1/T1",
        followup_dir="MSLesSeg Dataset/train/P1/T2",
        output_dir="./output/P1_T1_T2",
        patient_id="P1",
        baseline_tp="T1",
        followup_tp="T2",
    )
