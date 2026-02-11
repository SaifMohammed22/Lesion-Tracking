"""
Basic tests for the simplified lesion tracker.
"""

import numpy as np
from lesion_tracker import track_lesions, label_lesions


def test_label_lesions():
    """Test connected component labeling."""
    # Create a simple mask with 2 separate lesions
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1  # Lesion 1: 8 voxels
    mask[7:9, 7:9, 7:9] = 1  # Lesion 2: 8 voxels

    labeled, num = label_lesions(mask, min_size=3)

    assert num == 2, f"Expected 2 lesions, got {num}"
    assert labeled.max() == 2, "Should have labels 1 and 2"


def test_label_lesions_filters_small():
    """Test that small lesions are filtered out."""
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1  # Lesion 1: 8 voxels
    mask[7, 7, 7] = 1  # Lesion 2: 1 voxel (too small)

    labeled, num = label_lesions(mask, min_size=3)

    assert num == 1, f"Expected 1 lesion (small one filtered), got {num}"


def test_track_lesions_stable():
    """Test tracking identifies stable lesions."""
    # Same lesion in both timepoints
    baseline = np.zeros((10, 10, 10), dtype=np.uint8)
    baseline[2:5, 2:5, 2:5] = 1

    followup = np.zeros((10, 10, 10), dtype=np.uint8)
    followup[2:5, 2:5, 2:5] = 1

    results = track_lesions(baseline, followup, min_lesion_size=3)

    assert results["summary"]["num_stable"] == 1
    assert results["summary"]["num_new"] == 0
    assert results["summary"]["num_disappeared"] == 0


def test_track_lesions_new():
    """Test tracking identifies new lesions."""
    baseline = np.zeros((10, 10, 10), dtype=np.uint8)

    followup = np.zeros((10, 10, 10), dtype=np.uint8)
    followup[2:5, 2:5, 2:5] = 1

    results = track_lesions(baseline, followup, min_lesion_size=3)

    assert results["summary"]["num_new"] == 1
    assert results["summary"]["num_stable"] == 0


def test_track_lesions_disappeared():
    """Test tracking identifies disappeared lesions."""
    baseline = np.zeros((10, 10, 10), dtype=np.uint8)
    baseline[2:5, 2:5, 2:5] = 1

    followup = np.zeros((10, 10, 10), dtype=np.uint8)

    results = track_lesions(baseline, followup, min_lesion_size=3)

    assert results["summary"]["num_disappeared"] == 1
    assert results["summary"]["num_stable"] == 0


def test_track_lesions_enlarged():
    """Test tracking identifies enlarged lesions."""
    baseline = np.zeros((20, 20, 20), dtype=np.uint8)
    baseline[5:8, 5:8, 5:8] = 1  # 27 voxels

    followup = np.zeros((20, 20, 20), dtype=np.uint8)
    followup[4:10, 4:10, 4:10] = 1  # 216 voxels (8x larger)

    results = track_lesions(baseline, followup, min_lesion_size=3, change_threshold=0.2)

    assert results["summary"]["num_enlarged"] == 1
    assert results["summary"]["num_stable"] == 0


if __name__ == "__main__":
    test_label_lesions()
    test_label_lesions_filters_small()
    test_track_lesions_stable()
    test_track_lesions_new()
    test_track_lesions_disappeared()
    test_track_lesions_enlarged()
    print("All tests passed!")
