from lesion_tracker import LesionTracker

base_flair = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz"
basse_mask = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
followup_flair = "/mnt/data/MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz"
followup_mask = "/mnt/data/MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz"


tracker = LesionTracker()
tracker.load_baseline(base_flair, basse_mask)
tracker.load_followup(followup_flair, followup_mask)
tracker.track_lesions(base_flair, followup_flair)

# Generate report (now includes labeled lesion figures)
tracker.generate_report("./results")

# Or plot labeled lesions manually:
tracker.visualizer.plot_labeled_lesions(
    tracker.baseline_flair,
    tracker.baseline_labeled,
    lesions=tracker.baseline_lesions,
    save_path="my_labeled_lesions.png"
)