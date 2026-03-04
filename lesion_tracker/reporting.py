import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj
"""
Reporting results and saving it
"""
import os
import json
from typing import Dict, Any

import nibabel as nib
import numpy as np

try:
    from .utils import lesion_dice_score, load_nifti, save_nifti
except ImportError:
    from utils import lesion_dice_score, load_nifti, save_nifti

def print_summary(results: Dict[str, Any]):
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


def save_results(
    results: Dict[str, Any],
    reference_img: nib.Nifti1Image,
    fu_mask_registered: np.ndarray,
    output_dir: str,
    baseline_flair: str,
    followup_flair: str,
    baseline_mask_path: str = None,
    followup_mask_path: str = None,
):
    """Save tracking results to output directory (JSON, CSV, TXT, and NIfTI)."""
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
    #save_nifti(fu_mask_registered, reference_img, output_dir)

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