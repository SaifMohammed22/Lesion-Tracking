"""
Core lesion operations and tracking logic.

This module contains the pure numerical logic for:
- Labeling lesions
- Computing lesion properties
- Building overlap matrices
- Detecting merges and splits
- 1:1 matching
- Building tracked label maps and summary statistics
"""

from typing import Dict, Any, Optional, Tuple, List, Set

import numpy as np
from scipy import ndimage


# =============================================================================
# Constants
# =============================================================================

# Minimum overlap ratio (IoU) to consider lesions as overlapping
MIN_OVERLAP_RATIO = 0.10

# Volume change threshold for enlarged/shrinking classification
CHANGE_THRESHOLD = 0.20


def label_lesions(mask: np.ndarray, min_size: int = 3) -> Tuple[np.ndarray, int]:
    """
    Label connected components in mask, filter by minimum size.

    Returns
    -------
    labeled_array, num_lesions
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
    """
    # Label lesions in each timepoint
    bl_labeled, num_bl = label_lesions(baseline_mask, min_lesion_size)
    fu_labeled, num_fu = label_lesions(followup_mask, min_lesion_size)

    # Get lesion properties (centroid, volume) for each lesion
    bl_props = _get_lesion_properties(bl_labeled, num_bl, voxel_spacing)
    fu_props = _get_lesion_properties(fu_labeled, num_fu, voxel_spacing)

    # Build overlap matrix (includes distance filtering)
    overlap_matrix = _build_overlap_matrix(
        bl_labeled, fu_labeled, num_bl, num_fu, bl_props, fu_props, max_distance_mm
    )

    # Track which lesions have been matched
    matched_bl: Set[int] = set()
    matched_fu: Set[int] = set()

    # Results: list of lesion records
    lesions: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Step 1: Detect MERGES (multiple baseline → one follow-up)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 2: Detect SPLITS (one baseline → multiple follow-up)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 3: Standard 1:1 matching for remaining lesions
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 4: Classify matched lesions (Present, Enlarged, Shrinking)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 5: Mark unmatched baseline lesions as ABSENT
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 6: Mark unmatched follow-up lesions as NEW
    # -------------------------------------------------------------------------
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

    # Create tracked follow-up label map
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
) -> Dict[int, Dict[str, Any]]:
    """Extract centroid and volume for each lesion."""
    props: Dict[int, Dict[str, Any]] = {}
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
    bl_props: Dict[int, Dict[str, Any]] = None,
    fu_props: Dict[int, Dict[str, Any]] = None,
    max_distance_mm: float = 20.0,
) -> Dict[Tuple[int, int], float]:
    """
    Build overlap matrix between baseline and follow-up lesions.

    Returns dict of (bl_id, fu_id) -> IoU ratio.
    Only includes pairs with IoU >= MIN_OVERLAP_RATIO AND distance <= max_distance_mm.
    """
    overlaps: Dict[Tuple[int, int], float] = {}

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

            # Distance check - only match if close enough
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
    bl_props: Dict[int, Dict[str, Any]],
    fu_props: Dict[int, Dict[str, Any]],
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
    bl_props: Dict[int, Dict[str, Any]],
    fu_props: Dict[int, Dict[str, Any]],
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
    bl_props: Dict[int, Dict[str, Any]],
    fu_props: Dict[int, Dict[str, Any]],
    matched_bl: Set[int],
    matched_fu: Set[int],
    max_distance_mm: float,
    voxel_spacing: tuple,
) -> Dict[int, Optional[Dict[str, Any]]]:
    """
    Match remaining baseline lesions to follow-up lesions 1:1.

    Uses overlap-based matching first, then distance-based fallback.
    Returns: {bl_id: {fu_id, distance_mm, overlap_ratio} or None}
    """
    matches: Dict[int, Optional[Dict[str, Any]]] = {}
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
    unmatched_fu_list = [fu_id for fu_id in fu_props.keys() if fu_id not in local_used_fu]

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
    matches: Dict[int, Optional[Dict[str, Any]]],
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


def _build_summary(lesions: List[Dict[str, Any]], num_bl: int, num_fu: int) -> Dict[str, int]:
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

