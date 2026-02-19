from lesion_tracker import register_to_baseline, save_nifti, load_nifti

base_line = "MSLesSeg_RAW/P1/T1/P1_T1_FLAIR.nii.gz"
follow_up = "MSLesSeg_RAW/P1/T2/P1_T2_FLAIR.nii.gz"

results = register_to_baseline(base_line, follow_up, transform_type="Affine")

transforms = results["transforms"]
registered = results["registered"]   # ants.ANTsImage in baseline space
fixed = results["fixed"]             # ants.ANTsImage (baseline)

# Load baseline as NIfTI to get affine/header
fixed_data, fixed_img = load_nifti(base_line)

# Convert registered ANTs image to NumPy
registered_np = registered.numpy()

# Save as NIfTI
save_nifti(registered_np, fixed_img, "P1_T2_T1_Registered.nii.gz")