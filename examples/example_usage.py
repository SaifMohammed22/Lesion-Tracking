"""
Example usage of the simplified Lesion Tracker.

Track MS lesions between baseline and follow-up MRI scans.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lesion_tracker import run_tracking, track_mslesseg, visualize_tracking


def example_basic():
    """
    Basic example - track lesions with explicit file paths.
    """
    print("=" * 60)
    print("Lesion Tracking - Basic Example")
    print("=" * 60)

    results = run_tracking(
        baseline_flair="MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz",
        baseline_mask="MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz",
        followup_flair="MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz",
        followup_mask="MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz",
        output_dir="./output/P1_T1_T2",
        registration_type="Affine",
        min_lesion_size=3,
        change_threshold=0.2,  # 20% volume change = enlarged/shrunk
    )

    # Print detailed results
    print("\nDetailed results:")
    print(f"  Stable lesions: {results['stable']}")
    print(f"  Enlarged lesions: {results['enlarged']}")
    print(f"  Shrunk lesions: {results['shrunk']}")
    print(f"  Disappeared lesions: {results['disappeared']}")
    print(f"  New lesions: {results['new']}")

    return results


def example_mslesseg():
    """
    Example using MSLesSeg dataset format (convenience function).
    """
    print("=" * 60)
    print("Lesion Tracking - MSLesSeg Dataset")
    print("=" * 60)

    results = track_mslesseg(
        baseline_dir="MSLesSeg Dataset/train/P1/T1",
        followup_dir="MSLesSeg Dataset/train/P1/T2",
        output_dir="./output/P1_T1_T2",
        patient_id="P1",
        baseline_tp="T1",
        followup_tp="T2",
    )

    return results


def example_with_visualization():
    """
    Track lesions and create visualization.
    """
    print("=" * 60)
    print("Lesion Tracking - With Visualization")
    print("=" * 60)

    output_dir = "./output/P1_T1_T2"

    # Run tracking
    results = track_mslesseg(
        baseline_dir="MSLesSeg Dataset/train/P1/T1",
        followup_dir="MSLesSeg Dataset/train/P1/T2",
        output_dir=output_dir,
        patient_id="P1",
        baseline_tp="T1",
        followup_tp="T2",
    )

    # Create visualization
    if visualize_tracking is not None:
        visualize_tracking(
            flair_baseline_path="MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz",
            flair_followup_path="MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz",
            baseline_labeled_path=f"{output_dir}/baseline_lesions_labeled.nii.gz",
            followup_labeled_path=f"{output_dir}/followup_lesions_labeled.nii.gz",
            save_path=f"{output_dir}/tracking_visualization.png",
            show=True,
        )
    else:
        print("Visualization requires matplotlib")

    return results


def process_all_patients():
    """
    Process all patients with multiple timepoints.
    """
    import glob

    dataset_root = "MSLesSeg Dataset/train"
    output_root = "./output/all_patients"
    os.makedirs(output_root, exist_ok=True)

    patient_dirs = sorted(glob.glob(f"{dataset_root}/P*"))
    results_summary = []

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        timepoints = sorted(
            [
                d
                for d in os.listdir(patient_dir)
                if os.path.isdir(os.path.join(patient_dir, d))
            ]
        )

        if len(timepoints) < 2:
            print(f"Skipping {patient_id}: only {len(timepoints)} timepoint(s)")
            continue

        # Process consecutive pairs
        for i in range(len(timepoints) - 1):
            baseline_tp = timepoints[i]
            followup_tp = timepoints[i + 1]

            print(f"\n{'=' * 50}")
            print(f"{patient_id}: {baseline_tp} -> {followup_tp}")
            print("=" * 50)

            try:
                results = track_mslesseg(
                    baseline_dir=f"{patient_dir}/{baseline_tp}",
                    followup_dir=f"{patient_dir}/{followup_tp}",
                    output_dir=f"{output_root}/{patient_id}_{baseline_tp}_{followup_tp}",
                    patient_id=patient_id,
                    baseline_tp=baseline_tp,
                    followup_tp=followup_tp,
                )

                s = results["summary"]
                results_summary.append(
                    {
                        "patient": patient_id,
                        "baseline": baseline_tp,
                        "followup": followup_tp,
                        "new": s["num_new"],
                        "disappeared": s["num_disappeared"],
                        "stable": s["num_stable"],
                    }
                )
            except Exception as e:
                print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results_summary:
        print(
            f"{r['patient']} ({r['baseline']}->{r['followup']}): "
            f"New={r['new']}, Disappeared={r['disappeared']}, Stable={r['stable']}"
        )

    return results_summary


if __name__ == "__main__":
    print("\nLesion Tracker Examples")
    print("-" * 40)
    print("1. Basic example")
    print("2. MSLesSeg format")
    print("3. With visualization")
    print("4. Process all patients")

    choice = input("\nChoice (1-4): ").strip()

    if choice == "1":
        example_basic()
    elif choice == "2":
        example_mslesseg()
    elif choice == "3":
        example_with_visualization()
    elif choice == "4":
        process_all_patients()
    else:
        print("Running basic example...")
        example_basic()
