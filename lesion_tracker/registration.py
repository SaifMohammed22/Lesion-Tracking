"""
Image registration module using ANTsPy.
"""

from typing import Optional, Dict, Any
import ants
import numpy as np


class Registration:
    """ANTs-based registration for aligning baseline and follow-up scans."""
    
    def __init__(self, transform_type: str = 'Rigid'):
        """Args: transform_type: 'Rigid', 'Affine', or 'SyN' (non-linear)"""
        self.transform_type = transform_type
        self.transforms = None
    
    def register(self, fixed_path: str, moving_path: str, output_path: Optional[str] = None) -> ants.ANTsImage:
        """Register moving image to fixed image."""
        fixed = ants.image_read(fixed_path)
        moving = ants.image_read(moving_path)
        result = ants.registration(fixed=fixed, moving=moving, type_of_transform=self.transform_type)
        self.transforms = result['fwdtransforms']
        registered = result['warpedmovout']
        if output_path:
            ants.image_write(registered, output_path)
        return registered
    
    def apply_transform(
        self, image_path: str, reference_path: str, 
        output_path: Optional[str] = None, interpolation: str = 'nearestNeighbor'
    ) -> ants.ANTsImage:
        """Apply stored transforms to another image (e.g., mask)."""
        if self.transforms is None:
            raise ValueError("No transforms available. Run register() first.")
        image = ants.image_read(image_path)
        reference = ants.image_read(reference_path)
        transformed = ants.apply_transforms(fixed=reference, moving=image, transformlist=self.transforms, interpolator=interpolation)
        if output_path:
            ants.image_write(transformed, output_path)
        return transformed
    
    def check_alignment(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """Check alignment quality between two images."""
        img1, img2 = ants.image_read(image1_path), ants.image_read(image2_path)
        mi = ants.image_mutual_information(img1, img2)
        correlation = float(np.corrcoef(img1.numpy().flatten(), img2.numpy().flatten())[0, 1])
        return {'correlation': correlation, 'mutual_information': mi, 'aligned': correlation > 0.8}

if __name__ == "__main__":

    baseline = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_FLAIR.nii.gz"
    follow_up = "/mnt/data/MSLesSeg Dataset/train/P1/T2/P1_T2_FLAIR.nii.gz"

    reg = Registration()
    #registered_image = reg.register(fixed_path=baseline, moving_path=follow_up, output_path="./registered_followup.nii.gz")
    metrics = reg.check_alignment(baseline, follow_up)
    print("Alignment metrics:", metrics)
    if metrics['aligned'] > 0.8:
        print("Images are well aligned.")
    else:
        print("Images are not well aligned.")
