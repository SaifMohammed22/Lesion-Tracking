"""
Lesion Tracking - Core Module

Track MS lesions between two timepoints (baseline and follow-up).

Lesion Statuses:
- Present: Lesion exists at both timepoints with minimal volume change (±25%)
- Enlarged: Volume increased by >25%
- Shrinking: Volume decreased by >25%
- Absent: Lesion existed at baseline but not at follow-up
- Merged: Two or more baseline lesions merged into one follow-up lesion
- Split: One baseline lesion split into two or more follow-up lesions
- New: Lesion only exists at follow-up (no baseline match)
"""

import os
import json
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage
from typing import Dict, Any, Optional, Tuple, List, Set

try:
    import ants
except ImportError:
    ants = None

try:
    from .utils import load_nifti, save_nifti
except ImportError:
    from utils import load_nifti, save_nifti


# =============================================================================
# Constants
# =============================================================================

# Minimum overlap ratio (IoU) to consider lesions as overlapping
MIN_OVERLAP_RATIO = 0.10

# Volume change threshold for enlarged/shrinking classification
CHANGE_THRESHOLD = 0.20


# =============================================================================
# Registration
# =============================================================================


def register_to_baseline(
    baseline_path: str, followup_path: str, transform_type: str = "SyN"
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
    change_threshold: float = CHANGE_THRESHOLD,
    max_distance_mm: float = 20.0,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """
    Track lesions between baseline and follow-up with comprehensive status classification.

    Statuses: Present, Enlarged, Shrinking, Absent, Merged, Split, New

    Detection order:
    1. Detect merges (multiple baseline → one follow-up)
    2. Detect splits (one baseline → multiple follow-up)
    3. Standard 1:1 matching for remaining lesions
    4. Classify matched lesions by volume change
    5. Mark unmatched baseline as Absent
    6. Mark unmatched follow-up as New

    Args:
        baseline_mask: Binary lesion mask at baseline
        followup_mask: Binary lesion mask at follow-up (registered to baseline space)
        min_lesion_size: Minimum voxels to count as a lesion
        change_threshold: Volume change ratio for enlarged/shrinking (default 25%)
        max_distance_mm: Base distance (mm) for matching (default 20mm)
        voxel_spacing: Voxel size in mm (x, y, z)

    Returns:
        Dict with 'lesions' list and 'summary' statistics
    """
    # Label lesions in each timepoint
    bl_labeled, num_bl = label_lesions(baseline_mask, min_lesion_size)
    fu_labeled, num_fu = label_lesions(followup_mask, min_lesion_size)

    # Get lesion properties (centroid, volume) for each lesion
    bl_props = _get_lesion_properties(bl_labeled, num_bl, voxel_spacing)
    fu_props = _get_lesion_properties(fu_labeled, num_fu, voxel_spacing)

    # Build overlap matrix (IMPROVEMENT: now includes distance filtering)
    overlap_matrix = _build_overlap_matrix(
        bl_labeled, fu_labeled, num_bl, num_fu, bl_props, fu_props, max_distance_mm
    )

    # Track which lesions have been matched
    matched_bl: Set[int] = set()
    matched_fu: Set[int] = set()

    # Results: list of lesion records
    lesions: List[Dict[str, Any]] = []

    # =========================================================================
    # Step 1: Detect MERGES (multiple baseline → one follow-up)
    # =========================================================================
    merge_groups = _detect_merges(overlap_matrix, bl_props, fu_props, num_bl, num_fu)

    for fu_id, bl_ids in merge_groups.items():
        # Find the largest baseline lesion (will carry the merged volume)
        largest_bl = max(bl_ids, key=lambda x: bl_props[x]["volume"])
        fu_vol = fu_props[fu_id]["volume"]

        for bl_id in bl_ids:
            bl_vol = bl_props[bl_id]["volume"]
            if bl_id == largest_bl:
                # Largest baseline lesion carries the follow-up volume
                lesions.append(
                    {
                        "id": bl_id,
                        "status": "Merged",
                        "baseline_volume": int(bl_vol),
                        "followup_volume": int(fu_vol),
                        "merged_with": [x for x in bl_ids if x != bl_id],
                        "centroid": list(bl_props[bl_id]["centroid_mm"]),
                    }
                )
            else:
                # Other merged lesions have 0 follow-up volume
                lesions.append(
                    {
                        "id": bl_id,
                        "status": "Merged",
                        "baseline_volume": int(bl_vol),
                        "followup_volume": 0,
                        "merged_with": [x for x in bl_ids if x != bl_id],
                        "centroid": list(bl_props[bl_id]["centroid_mm"]),
                    }
                )
            matched_bl.add(bl_id)
        matched_fu.add(fu_id)

    # =========================================================================
    # Step 2: Detect SPLITS (one baseline → multiple follow-up)
    # =========================================================================
    split_groups = _detect_splits(
        overlap_matrix, bl_props, fu_props, num_bl, num_fu, matched_bl, matched_fu
    )

    for bl_id, fu_ids in split_groups.items():
        bl_vol = bl_props[bl_id]["volume"]
        # Sum of all split parts
        fu_vol_total = sum(fu_props[fid]["volume"] for fid in fu_ids)

        lesions.append(
            {
                "id": bl_id,
                "status": "Split",
                "baseline_volume": int(bl_vol),
                "followup_volume": int(fu_vol_total),
                "split_count": len(fu_ids),
                "centroid": list(bl_props[bl_id]["centroid_mm"]),
            }
        )
        matched_bl.add(bl_id)
        matched_fu.update(fu_ids)

    # =========================================================================
    # Step 3: Standard 1:1 matching for remaining lesions
    # =========================================================================
    matches = _match_lesions_1to1(
        bl_labeled,
        fu_labeled,
        bl_props,
        fu_props,
        matched_bl,
        matched_fu,
        max_distance_mm,
        voxel_spacing,
    )

    # =========================================================================
    # Step 4: Classify matched lesions (Present, Enlarged, Shrinking)
    # =========================================================================
    for bl_id, match_info in matches.items():
        if match_info is None:
            continue

        fu_id = match_info["fu_id"]
        bl_vol = bl_props[bl_id]["volume"]
        fu_vol = fu_props[fu_id]["volume"]
        change_ratio = (fu_vol - bl_vol) / bl_vol if bl_vol > 0 else 0

        if change_ratio > change_threshold:
            status = "Enlarged"
        elif change_ratio < -change_threshold:
            status = "Shrinking"
        else:
            status = "Present"

        lesions.append(
            {
                "id": bl_id,
                "status": status,
                "baseline_volume": int(bl_vol),
                "followup_volume": int(fu_vol),
                "change_ratio": round(change_ratio, 4),
                "centroid": list(bl_props[bl_id]["centroid_mm"]),
            }
        )
        matched_bl.add(bl_id)
        matched_fu.add(fu_id)

    # =========================================================================
    # Step 5: Mark unmatched baseline lesions as ABSENT
    # =========================================================================
    for bl_id in range(1, num_bl + 1):
        if bl_id not in matched_bl:
            lesions.append(
                {
                    "id": bl_id,
                    "status": "Absent",
                    "baseline_volume": int(bl_props[bl_id]["volume"]),
                    "followup_volume": 0,
                    "centroid": list(bl_props[bl_id]["centroid_mm"]),
                }
            )

    # =========================================================================
    # Step 6: Mark unmatched follow-up lesions as NEW
    # =========================================================================
    next_id = num_bl + 1
    for fu_id in range(1, num_fu + 1):
        if fu_id not in matched_fu:
            lesions.append(
                {
                    "id": next_id,
                    "status": "New",
                    "baseline_volume": 0,
                    "followup_volume": int(fu_props[fu_id]["volume"]),
                    "centroid": list(fu_props[fu_id]["centroid_mm"]),
                }
            )
            next_id += 1

    # Sort lesions by ID
    lesions.sort(key=lambda x: x["id"])

    # =========================================================================
    # Create tracked follow-up label map
    # =========================================================================
    fu_tracked = _create_tracked_labels(
        fu_labeled,
        bl_labeled,
        matches,
        merge_groups,
        split_groups,
        matched_fu,
        num_bl,
        num_fu,
    )

    # Build summary
    summary = _build_summary(lesions, num_bl, num_fu)

    return {
        "lesions": lesions,
        "summary": summary,
        "baseline_labeled": bl_labeled,
        "followup_labeled": fu_tracked,
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


def _build_overlap_matrix(
    bl_labeled: np.ndarray,
    fu_labeled: np.ndarray,
    num_bl: int,
    num_fu: int,
    bl_props: Dict = None,
    fu_props: Dict = None,
    max_distance_mm: float = 20.0,
) -> Dict[Tuple[int, int], float]:
    """
    Build overlap matrix between baseline and follow-up lesions.
    Returns dict of (bl_id, fu_id) -> IoU ratio.
    Only includes pairs with IoU >= MIN_OVERLAP_RATIO AND distance <= max_distance_mm.

    IMPROVEMENT: Added distance filtering to prevent false matches between
    distant lesions that happen to have some overlap.
    """
    overlaps = {}

    for bl_id in range(1, num_bl + 1):
        bl_region = bl_labeled == bl_id

        # Find all follow-up lesions that overlap
        fu_ids_in_region = np.unique(fu_labeled[bl_region])

        for fu_id in fu_ids_in_region:
            if fu_id == 0:
                continue

            fu_region = fu_labeled == fu_id
            intersection = (bl_region & fu_region).sum()
            union = (bl_region | fu_region).sum()
            iou = intersection / union if union > 0 else 0

            # IMPROVEMENT: Add distance check - only match if close enough
            if bl_props is not None and fu_props is not None:
                bl_centroid = bl_props[bl_id]["centroid_mm"]
                fu_centroid = fu_props[fu_id]["centroid_mm"]
                distance = np.sqrt(
                    sum((a - b) ** 2 for a, b in zip(bl_centroid, fu_centroid))
                )
                # Only include if both IoU AND distance thresholds are met
                if iou >= MIN_OVERLAP_RATIO and distance <= max_distance_mm:
                    overlaps[(bl_id, fu_id)] = iou
            else:
                # Fallback to IoU only if props not provided
                if iou >= MIN_OVERLAP_RATIO:
                    overlaps[(bl_id, fu_id)] = iou

    return overlaps


def _detect_merges(
    overlap_matrix: Dict[Tuple[int, int], float],
    bl_props: Dict,
    fu_props: Dict,
    num_bl: int,
    num_fu: int,
) -> Dict[int, List[int]]:
    """
    Detect merged lesions: multiple baseline lesions → one follow-up lesion.

    Returns: {fu_id: [bl_id1, bl_id2, ...]} for each merge group.
    """
    # For each follow-up lesion, find all baseline lesions it overlaps with
    fu_to_bl: Dict[int, List[int]] = {}

    for (bl_id, fu_id), iou in overlap_matrix.items():
        if fu_id not in fu_to_bl:
            fu_to_bl[fu_id] = []
        fu_to_bl[fu_id].append(bl_id)

    # Merge groups: fu_id maps to 2+ baseline lesions
    merge_groups = {
        fu_id: bl_ids for fu_id, bl_ids in fu_to_bl.items() if len(bl_ids) >= 2
    }

    return merge_groups


def _detect_splits(
    overlap_matrix: Dict[Tuple[int, int], float],
    bl_props: Dict,
    fu_props: Dict,
    num_bl: int,
    num_fu: int,
    matched_bl: Set[int],
    matched_fu: Set[int],
) -> Dict[int, List[int]]:
    """
    Detect split lesions: one baseline lesion → multiple follow-up lesions.

    Returns: {bl_id: [fu_id1, fu_id2, ...]} for each split group.
    Only considers unmatched lesions.
    """
    # For each baseline lesion, find all follow-up lesions it overlaps with
    bl_to_fu: Dict[int, List[int]] = {}

    for (bl_id, fu_id), iou in overlap_matrix.items():
        # Skip if already matched
        if bl_id in matched_bl or fu_id in matched_fu:
            continue

        if bl_id not in bl_to_fu:
            bl_to_fu[bl_id] = []
        bl_to_fu[bl_id].append(fu_id)

    # Split groups: bl_id maps to 2+ follow-up lesions
    split_groups = {
        bl_id: fu_ids for bl_id, fu_ids in bl_to_fu.items() if len(fu_ids) >= 2
    }

    return split_groups


def _match_lesions_1to1(
    bl_labeled: np.ndarray,
    fu_labeled: np.ndarray,
    bl_props: Dict,
    fu_props: Dict,
    matched_bl: Set[int],
    matched_fu: Set[int],
    max_distance_mm: float,
    voxel_spacing: tuple,
) -> Dict[int, Optional[Dict]]:
    """
    Match remaining baseline lesions to follow-up lesions 1:1.

    Uses overlap-based matching first, then distance-based fallback.
    Returns: {bl_id: {fu_id, distance_mm, overlap_ratio} or None}
    """
    matches = {}
    local_used_fu = set(matched_fu)

    # Get unmatched baseline lesions
    unmatched_bl = [bl_id for bl_id in bl_props.keys() if bl_id not in matched_bl]

    # First pass: match by overlap
    for bl_id in unmatched_bl:
        bl_region = bl_labeled == bl_id

        # Find overlapping follow-up lesions (not already used)
        overlap_ids = np.unique(fu_labeled[bl_region])
        overlap_ids = [
            fid for fid in overlap_ids if fid > 0 and fid not in local_used_fu
        ]

        if overlap_ids:
            # Find best overlap
            best_fu_id = None
            best_overlap = 0
            best_iou = 0.0

            for fu_id in overlap_ids:
                fu_region = fu_labeled == fu_id
                intersection = (bl_region & fu_region).sum()
                union = (bl_region | fu_region).sum()
                iou = intersection / union if union > 0 else 0

                if iou >= MIN_OVERLAP_RATIO and intersection > best_overlap:
                    best_overlap = intersection
                    best_fu_id = fu_id
                    best_iou = iou

            if best_fu_id is not None:
                bl_cent = np.array(bl_props[bl_id]["centroid_mm"])
                fu_cent = np.array(fu_props[best_fu_id]["centroid_mm"])
                distance = np.sqrt(np.sum((bl_cent - fu_cent) ** 2))

                matches[bl_id] = {
                    "fu_id": best_fu_id,
                    "distance_mm": distance,
                    "overlap_ratio": best_iou,
                }
                local_used_fu.add(best_fu_id)

    # Second pass: match remaining by distance
    still_unmatched_bl = [bl_id for bl_id in unmatched_bl if bl_id not in matches]
    unmatched_fu_list = [
        fu_id for fu_id in fu_props.keys() if fu_id not in local_used_fu
    ]

    for bl_id in still_unmatched_bl:
        bl_cent = np.array(bl_props[bl_id]["centroid_mm"])
        bl_vol = bl_props[bl_id]["volume"]

        best_fu_id = None
        best_score = float("inf")
        best_distance = 0.0

        # Size-weighted distance threshold
        size_factor = max(1.0, 2.0 - bl_vol / 100.0)
        effective_max_dist = max_distance_mm * size_factor

        for fu_id in unmatched_fu_list:
            if fu_id in local_used_fu:
                continue

            fu_cent = np.array(fu_props[fu_id]["centroid_mm"])
            fu_vol = fu_props[fu_id]["volume"]

            distance = np.sqrt(np.sum((bl_cent - fu_cent) ** 2))

            if distance > effective_max_dist:
                continue

            # Volume ratio for scoring
            vol_ratio = max(bl_vol, fu_vol) / max(min(bl_vol, fu_vol), 1)

            # Size-weighted scoring
            size_weight = min(1.0, bl_vol / 50.0)
            distance_penalty = distance * size_weight
            vol_penalty = np.log1p(vol_ratio - 1) * 3

            score = distance_penalty + vol_penalty

            if score < best_score:
                best_score = score
                best_fu_id = fu_id
                best_distance = distance

        if best_fu_id is not None:
            matches[bl_id] = {
                "fu_id": best_fu_id,
                "distance_mm": best_distance,
                "overlap_ratio": 0.0,
            }
            local_used_fu.add(best_fu_id)
        else:
            matches[bl_id] = None

    return matches


def _create_tracked_labels(
    fu_labeled: np.ndarray,
    bl_labeled: np.ndarray,
    matches: Dict[int, Optional[Dict]],
    merge_groups: Dict[int, List[int]],
    split_groups: Dict[int, List[int]],
    matched_fu: Set[int],
    num_bl: int,
    num_fu: int,
) -> np.ndarray:
    """Create follow-up label map with baseline IDs for matched lesions."""
    fu_tracked = np.zeros_like(fu_labeled)

    # Handle merges: all merged baseline lesions map to the merged follow-up
    for fu_id, bl_ids in merge_groups.items():
        # Use the smallest bl_id as the label (or largest by volume - we use smallest for simplicity)
        label = min(bl_ids)
        fu_tracked[fu_labeled == fu_id] = label

    # Handle splits: all split follow-up lesions get the baseline ID
    for bl_id, fu_ids in split_groups.items():
        for fu_id in fu_ids:
            fu_tracked[fu_labeled == fu_id] = bl_id

    # Handle 1:1 matches
    for bl_id, match_info in matches.items():
        if match_info is not None:
            fu_id = match_info["fu_id"]
            fu_tracked[fu_labeled == fu_id] = bl_id

    # Handle new lesions (unmatched follow-up)
    next_id = num_bl + 1
    for fu_id in range(1, num_fu + 1):
        if fu_id not in matched_fu:
            # Check if not already labeled (from splits/merges handled above)
            mask = fu_labeled == fu_id
            if not np.any(fu_tracked[mask] > 0):
                fu_tracked[mask] = next_id
                next_id += 1

    return fu_tracked


def _build_summary(lesions: List[Dict], num_bl: int, num_fu: int) -> Dict[str, int]:
    """Build summary statistics from lesion list."""
    status_counts = {
        "present": 0,
        "enlarged": 0,
        "shrinking": 0,
        "absent": 0,
        "merged": 0,
        "split": 0,
        "new": 0,
    }

    for lesion in lesions:
        status = lesion["status"].lower()
        if status in status_counts:
            status_counts[status] += 1

    return {
        "total_baseline": num_bl,
        "total_followup": num_fu,
        **status_counts,
    }


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
    min_lesion_size: int = 5,
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
        max_distance_mm=20.0,
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
            baseline_mask,
            followup_mask,
        )

    return results


def _print_summary(results: Dict[str, Any]):
    """Print tracking results with detailed lesion table."""
    s = results["summary"]
    lesions = results["lesions"]
    voxel_vol = results.get("voxel_volume_mm3", 1.0)

    print("\n" + "=" * 80)
    print("LESION TRACKING RESULTS")
    print("=" * 80)

    # Overview
    print(f"\nOverview:")
    print(f"  Baseline lesions:  {s['total_baseline']}")
    print(f"  Follow-up lesions: {s['total_followup']}")

    # Status counts
    print(f"\nStatus Summary:")
    print(f"  Present:   {s['present']:3d}  (stable, ±25% volume)")
    print(f"  Enlarged:  {s['enlarged']:3d}  (>+25% volume)")
    print(f"  Shrinking: {s['shrinking']:3d}  (>-25% volume)")
    print(f"  Absent:    {s['absent']:3d}  (disappeared)")
    print(f"  Merged:    {s['merged']:3d}  (combined with other lesion)")
    print(f"  Split:     {s['split']:3d}  (divided into multiple)")
    print(f"  New:       {s['new']:3d}  (not in baseline)")

    # Detailed lesion table
    print("\n" + "-" * 80)
    print("DETAILED LESION TABLE")
    print("-" * 80)

    # Table header
    print(
        f"{'ID':>4} | {'Status':<10} | {'BL Vol':>8} | {'FU Vol':>8} | {'Change':>8} | {'Notes':<20}"
    )
    print("-" * 80)

    # Sort lesions by ID
    for lesion in sorted(lesions, key=lambda x: x["id"]):
        lid = lesion["id"]
        status = lesion["status"]
        bl_vol = lesion["baseline_volume"]
        fu_vol = lesion["followup_volume"]

        # Calculate change percentage
        if bl_vol > 0:
            change_pct = ((fu_vol - bl_vol) / bl_vol) * 100
            change_str = f"{change_pct:+.1f}%"
        elif fu_vol > 0:
            change_str = "NEW"
        else:
            change_str = "-"

        # Build notes
        notes = ""
        if "merged_with" in lesion:
            notes = f"merged with {lesion['merged_with']}"
        elif "split_count" in lesion:
            notes = f"split into {lesion['split_count']} parts"
        elif "change_ratio" in lesion:
            ratio = lesion["change_ratio"]
            if abs(ratio) < 0.25:
                notes = "stable"

        # Status-specific formatting
        status_display = status.capitalize()

        print(
            f"{lid:>4} | {status_display:<10} | {bl_vol:>8} | {fu_vol:>8} | {change_str:>8} | {notes:<20}"
        )

    print("-" * 80)

    # Volume summary
    total_bl_vol = sum(l["baseline_volume"] for l in lesions if l["status"] != "New")
    total_fu_vol = sum(l["followup_volume"] for l in lesions if l["status"] != "Absent")

    print(f"\nVolume Summary (voxels):")
    print(f"  Total baseline volume:  {total_bl_vol:,}")
    print(f"  Total follow-up volume: {total_fu_vol:,}")
    if total_bl_vol > 0:
        total_change = ((total_fu_vol - total_bl_vol) / total_bl_vol) * 100
        print(f"  Total volume change:    {total_change:+.1f}%")

    # Convert to mm³ if voxel volume is known
    if voxel_vol != 1.0:
        print(f"\nVolume Summary (mm³):")
        print(f"  Total baseline volume:  {total_bl_vol * voxel_vol:,.1f} mm³")
        print(f"  Total follow-up volume: {total_fu_vol * voxel_vol:,.1f} mm³")

    print("=" * 80)


def _save_results(
    results: Dict[str, Any],
    reference_img: nib.Nifti1Image,
    fu_mask_registered: np.ndarray,
    output_dir: str,
    baseline_flair: str,
    followup_flair: str,
    save_visualization: bool = True,
    num_slices: int = 10,
    baseline_mask_path: str = None,
    followup_mask_path: str = None,
):
    """Save tracking results to output directory (JSON, CSV, TXT, and PNG)."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")

    lesions = results["lesions"]
    summary = results["summary"]
    voxel_vol = results.get("voxel_volume_mm3", 1.0)

    # =========================================================================
    # 1. Save JSON (full data)
    # =========================================================================
    output_data = {
        "summary": summary,
        "lesions": lesions,
        "voxel_volume_mm3": voxel_vol,
    }
    with open(os.path.join(output_dir, "tracking_results.json"), "w") as f:
        json.dump(output_data, f, indent=2)
    print("  - tracking_results.json")

    # =========================================================================
    # 2. Save CSV (lesion table)
    # =========================================================================
    csv_path = os.path.join(output_dir, "lesion_table.csv")
    with open(csv_path, "w") as f:
        # Header
        f.write(
            "ID,Status,Baseline_Volume,Followup_Volume,Change_Percent,Baseline_Volume_mm3,Followup_Volume_mm3,Notes\n"
        )

        for lesion in sorted(lesions, key=lambda x: x["id"]):
            lid = lesion["id"]
            status = lesion["status"]
            bl_vol = lesion["baseline_volume"]
            fu_vol = lesion["followup_volume"]

            # Calculate change percentage
            if bl_vol > 0:
                change_pct = ((fu_vol - bl_vol) / bl_vol) * 100
            elif fu_vol > 0:
                change_pct = 100.0  # New lesion
            else:
                change_pct = 0.0

            # Build notes
            notes = ""
            if "merged_with" in lesion:
                notes = f"merged with {lesion['merged_with']}"
            elif "split_count" in lesion:
                notes = f"split into {lesion['split_count']} parts"

            # Write row
            f.write(
                f"{lid},{status},{bl_vol},{fu_vol},{change_pct:.2f},{bl_vol * voxel_vol:.2f},{fu_vol * voxel_vol:.2f},{notes}\n"
            )

    print("  - lesion_table.csv")

    # =========================================================================
    # 3. Save TXT (formatted summary report)
    # =========================================================================
    txt_path = os.path.join(output_dir, "tracking_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LESION TRACKING RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Overview
        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline lesions:  {summary['total_baseline']}\n")
        f.write(f"Follow-up lesions: {summary['total_followup']}\n\n")

        # Status counts
        f.write("STATUS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Present:   {summary['present']:3d}  (stable, ±25% volume)\n")
        f.write(f"  Enlarged:  {summary['enlarged']:3d}  (>+25% volume)\n")
        f.write(f"  Shrinking: {summary['shrinking']:3d}  (>-25% volume)\n")
        f.write(f"  Absent:    {summary['absent']:3d}  (disappeared)\n")
        f.write(f"  Merged:    {summary['merged']:3d}  (combined with other lesion)\n")
        f.write(f"  Split:     {summary['split']:3d}  (divided into multiple)\n")
        f.write(f"  New:       {summary['new']:3d}  (not in baseline)\n\n")

        # Detailed table
        f.write("=" * 80 + "\n")
        f.write("DETAILED LESION TABLE\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'ID':>4} | {'Status':<10} | {'BL Vol':>8} | {'FU Vol':>8} | {'Change':>8} | {'Notes':<20}\n"
        )
        f.write("-" * 80 + "\n")

        for lesion in sorted(lesions, key=lambda x: x["id"]):
            lid = lesion["id"]
            status = lesion["status"]
            bl_vol = lesion["baseline_volume"]
            fu_vol = lesion["followup_volume"]

            if bl_vol > 0:
                change_pct = ((fu_vol - bl_vol) / bl_vol) * 100
                change_str = f"{change_pct:+.1f}%"
            elif fu_vol > 0:
                change_str = "NEW"
            else:
                change_str = "-"

            notes = ""
            if "merged_with" in lesion:
                notes = f"merged with {lesion['merged_with']}"
            elif "split_count" in lesion:
                notes = f"split into {lesion['split_count']} parts"
            elif "change_ratio" in lesion and abs(lesion["change_ratio"]) < 0.25:
                notes = "stable"

            f.write(
                f"{lid:>4} | {status:<10} | {bl_vol:>8} | {fu_vol:>8} | {change_str:>8} | {notes:<20}\n"
            )

        f.write("-" * 80 + "\n\n")

        # Volume summary
        total_bl_vol = sum(
            l["baseline_volume"] for l in lesions if l["status"] != "New"
        )
        total_fu_vol = sum(
            l["followup_volume"] for l in lesions if l["status"] != "Absent"
        )

        f.write("VOLUME SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Total baseline volume:  {total_bl_vol:,} voxels ({total_bl_vol * voxel_vol:,.1f} mm³)\n"
        )
        f.write(
            f"Total follow-up volume: {total_fu_vol:,} voxels ({total_fu_vol * voxel_vol:,.1f} mm³)\n"
        )
        if total_bl_vol > 0:
            total_change = ((total_fu_vol - total_bl_vol) / total_bl_vol) * 100
            f.write(f"Total volume change:    {total_change:+.1f}%\n")

        f.write("\n" + "=" * 80 + "\n")

    print("  - tracking_report.txt")

    # =========================================================================
    # 4. Save .nii.gz files
    # =========================================================================
    save_nifti(fu_mask_registered, reference_img, output_dir)

    # Save labeled baseline mask
    bl_labeled = results["baseline_labeled"]
    bl_labeled_path = os.path.join(output_dir, "baseline_labeled.nii.gz")
    save_nifti(bl_labeled.astype(np.int32), reference_img, bl_labeled_path)
    print("  - baseline_labeled.nii.gz")

    # Save labeled/tracked follow-up mask
    fu_labeled = results["followup_labeled"]
    fu_labeled_path = os.path.join(output_dir, "followup_labeled.nii.gz")
    save_nifti(fu_labeled.astype(np.int32), reference_img, fu_labeled_path)
    print("  - followup_labeled.nii.gz")

    # =========================================================================
    # 5. Compute and save Dice scores (if ground truth masks provided)
    # =========================================================================
    dice_results = {}
    if baseline_mask_path and followup_mask_path:
        try:
            try:
                from .utils import lesion_dice_score
            except ImportError:
                from utils import lesion_dice_score

            gt_bl, _ = load_nifti(baseline_mask_path)
            gt_fu, _ = load_nifti(followup_mask_path)

            bl_dice = lesion_dice_score(bl_labeled, gt_bl)
            fu_dice = lesion_dice_score(fu_labeled, gt_fu)

            dice_results = {
                "baseline": bl_dice,
                "followup": fu_dice,
            }

            dice_path = os.path.join(output_dir, "dice_scores.json")
            with open(dice_path, "w") as f:
                json.dump(dice_results, f, indent=2)
            print("  - dice_scores.json")

            print(f"\n  Dice Score - Baseline: {bl_dice['overall_dice']:.4f}")
            print(f"  Dice Score - Followup: {fu_dice['overall_dice']:.4f}")

        except Exception as e:
            print(f"  Dice computation failed: {e}")

    # =========================================================================
    # 6. Save PNG visualization
    # =========================================================================
    if save_visualization:
        try:
            try:
                from .visualization import visualize_tracking
            except ImportError:
                from visualization import visualize_tracking

            png_path = os.path.join(output_dir, "tracking_visualization.png")
            visualize_tracking(
                flair_baseline_path=baseline_flair,
                flair_followup_path=followup_flair,
                baseline_labeled=results["baseline_labeled"],
                followup_labeled=results["followup_labeled"],
                lesions=results["lesions"],
                num_slices=num_slices,
                save_path=png_path,
                show=False,
            )
            print("  - tracking_visualization.png")
        except ImportError as e:
            print(f"Visualization failed: {e}")


# =============================================================================
# Convenience function for MSLesSeg dataset format
# =============================================================================


def track(
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
    results = track(
        baseline_dir="MSLesSeg Dataset/train/P5/T1",
        followup_dir="MSLesSeg Dataset/train/P5/T2",
        output_dir="./output/P5_T1_T2",
        patient_id="P5",
        baseline_tp="T1",
        followup_tp="T2",
    )