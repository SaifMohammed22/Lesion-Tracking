"""
Lesion labeling module using connected component analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import ndimage

from .utils import load_nifti, save_nifti, binarize_mask, get_bounding_box, get_centroid, get_voxel_volume


class Lesion:
    """Class representing a single lesion."""
    
    def __init__(self, label: int, mask: np.ndarray, voxel_volume_mm3: float = 1.0):
        self.label = label
        self.mask = mask
        self.voxel_count = int(np.sum(mask > 0))
        self.volume_mm3 = self.voxel_count * voxel_volume_mm3
        self.centroid = get_centroid(mask)
        self.bounding_box = get_bounding_box(mask)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'label': self.label,
            'voxel_count': self.voxel_count,
            'volume_mm3': self.volume_mm3,
            'centroid': self.centroid,
            'bounding_box': self.bounding_box
        }
    
    def __repr__(self) -> str:
        return f"Lesion(label={self.label}, volume={self.volume_mm3:.2f}mm³)"


class LesionLabeler:
    """Label individual lesions using connected component analysis."""
    
    def __init__(self, connectivity: int = 26, min_lesion_voxels: int = 3, sort_by_size: bool = True):
        self.min_lesion_voxels = min_lesion_voxels
        self.sort_by_size = sort_by_size
        # 6=faces, 18=faces+edges, 26=faces+edges+corners
        self.structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}.get(connectivity, 3))
    
    def label_lesions(self, mask: np.ndarray, voxel_volume_mm3: float = 1.0) -> Tuple[np.ndarray, List[Lesion]]:
        """Label individual lesions in a binary mask."""
        binary_mask = binarize_mask(mask)
        labeled_array, num_features = ndimage.label(binary_mask, structure=self.structure)
        
        # Get component sizes and filter
        component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        valid_components = [(i, size) for i, size in enumerate(component_sizes, 1) if size >= self.min_lesion_voxels]
        
        if self.sort_by_size:
            valid_components.sort(key=lambda x: x[1], reverse=True)
        
        # Create new labeled mask with sequential labels
        new_labeled_mask = np.zeros_like(labeled_array)
        lesions = []
        
        for new_label, (old_label, _) in enumerate(valid_components, start=1):
            component_mask = (labeled_array == old_label).astype(np.uint8)
            new_labeled_mask[component_mask > 0] = new_label
            lesions.append(Lesion(label=new_label, mask=component_mask, voxel_volume_mm3=voxel_volume_mm3))
        
        return new_labeled_mask, lesions
    
    def label_lesions_from_file(self, mask_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, List[Lesion]]:
        """Label lesions from a NIfTI mask file."""
        mask_data, mask_img = load_nifti(mask_path)
        labeled_mask, lesions = self.label_lesions(mask_data, voxel_volume_mm3=get_voxel_volume(mask_img))
        if output_path:
            save_nifti(labeled_mask, mask_img, output_path, dtype=np.int16)
        return labeled_mask, lesions
    
    def get_lesion_statistics(self, lesions: List[Lesion]) -> Dict[str, Any]:
        """Compute statistics for a list of lesions."""
        if not lesions:
            return {'num_lesions': 0, 'total_volume_mm3': 0.0, 'mean_volume_mm3': 0.0}
        volumes = [l.volume_mm3 for l in lesions]
        return {
            'num_lesions': len(lesions),
            'total_volume_mm3': sum(volumes),
            'mean_volume_mm3': np.mean(volumes),
            'min_volume_mm3': min(volumes),
            'max_volume_mm3': max(volumes),
        }
    
if __name__ == "__main__":
    labeler = LesionLabeler(connectivity=26, min_lesion_voxels=3)
    mask_path = "/mnt/data/MSLesSeg Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
    output_path = "./labeled_lesions.nii.gz"
    
    labeled_mask, lesions = labeler.label_lesions_from_file(mask_path, output_path)
    stats = labeler.get_lesion_statistics(lesions)
    
    print(f"Labeled {stats['num_lesions']} lesions")
    print(f"Total volume: {stats['total_volume_mm3']:.2f} mm³")
    print(f"Mean volume: {stats['mean_volume_mm3']:.2f} mm³")
    for lesion in lesions:
        print(lesion)