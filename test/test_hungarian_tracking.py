from lesion_tracker import load_nifti, track_lesions_hungarian

# Paths to your masks
baseline_mask_path = "MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
followup_mask_path = "MSLesSeg Dataset/train/P1/T2/P1_T2_MASK.nii.gz"

# Load masks as NumPy arrays
baseline_mask, _ = load_nifti(baseline_mask_path)
followup_mask, _ = load_nifti(followup_mask_path)

# Run Hungarian tracking
result = track_lesions_hungarian(baseline_mask, followup_mask, min_lesion_size=3, min_overlap=0.05)

print("Assignments:", result["assignments"])
print(len(result["assignments"]))
print("Absent lesions:", result["absent"])
print("New lesions:", result["new"])