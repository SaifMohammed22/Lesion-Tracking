# MRI Lesion Tracking Tool

A comprehensive tool for tracking and labeling lesions in baseline and follow-up MRI scans, specifically designed for the MSLesSeg dataset.

## Features

- **Lesion Labeling**: Automatically label individual lesions using connected component analysis
- **Lesion Tracking**: Track lesions across baseline and follow-up scans
- **Change Detection**: Identify new, resolved, growing, and shrinking lesions
- **Visualization**: Generate visual reports comparing lesion changes
- **Metrics**: Calculate volumetric changes and statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from lesion_tracker import LesionTracker

# Initialize tracker
tracker = LesionTracker()

# Load scans and masks
tracker.load_baseline(
    flair_path="path/to/baseline_flair.nii.gz",
    mask_path="path/to/baseline_mask.nii.gz"
)

tracker.load_followup(
    flair_path="path/to/followup_flair.nii.gz",
    mask_path="path/to/followup_mask.nii.gz"
)

# Perform tracking
results = tracker.track_lesions()

# Generate report
tracker.generate_report(output_dir="./results")
```

### Command Line Interface

```bash
# Track lesions between baseline and follow-up
python -m lesion_tracker track \
    --baseline-flair baseline_flair.nii.gz \
    --baseline-mask baseline_mask.nii.gz \
    --followup-flair followup_flair.nii.gz \
    --followup-mask followup_mask.nii.gz \
    --output ./results

# Label lesions in a single scan
python -m lesion_tracker label \
    --flair scan.nii.gz \
    --mask mask.nii.gz \
    --output labeled_lesions.nii.gz
```

## Work flow
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LESION TRACKING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐         ┌──────────────┐
     │  Baseline    │         │  Follow-up   │
     │  FLAIR MRI   │         │  FLAIR MRI   │
     └──────┬───────┘         └──────┬───────┘
            │                        │
            ▼                        ▼
     ┌──────────────┐         ┌──────────────┐
     │  Baseline    │         │  Follow-up   │
     │  Lesion Mask │         │  Lesion Mask │
     │  (from nnU-Net│         │  (from nnU-Net│
     │   or manual) │         │   or manual) │
     └──────┬───────┘         └──────┬───────┘
            │                        │
            └───────────┬────────────┘
                        ▼
              ┌─────────────────┐
              │  REGISTRATION   │  ← Align scans if needed
              │ (registration.py)│
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ LESION LABELING │  ← Connected component analysis
              │  (labeler.py)   │    Identify individual lesions
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ LESION MATCHING │  ← Match lesions between timepoints
              │  (tracker.py)   │    Using overlap + distance
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ CLASSIFICATION  │  ← Categorize each lesion
              │  (tracker.py)   │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │  NEW    │   │ MATCHED  │   │ RESOLVED │
   │ lesions │   │ lesions  │   │ lesions  │
   └─────────┘   └────┬─────┘   └──────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌──────────┐
    │ GROWING │  │ STABLE  │  │SHRINKING │
    │  >20%   │  │  ±20%   │  │  >20%    │
    └─────────┘  └─────────┘  └──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    OUTPUTS      │
              ├─────────────────┤
              │ • tracking.json │
              │ • summary.csv   │
              │ • report.txt    │
              │ • visualizations│
              └─────────────────┘
```

## Output

The tool generates:
- Labeled lesion masks for baseline and follow-up
- Tracking results with lesion correspondences
- Volumetric statistics (CSV format)
- Visual comparison images (PNG format)
- Detailed JSON report

## Lesion Categories

After tracking, lesions are categorized as:
- **Stable**: Lesions present in both scans with minimal size change
- **Growing**: Lesions that increased in volume (>20% by default)
- **Shrinking**: Lesions that decreased in volume (>20% by default)
- **New**: Lesions only present in follow-up
- **Resolved**: Lesions only present in baseline

## Project Structure

```
lesion_tracking/
├── lesion_tracker/
│   ├── __init__.py
│   ├── tracker.py          # Main tracking logic
│   ├── labeler.py          # Lesion labeling utilities
│   ├── registration.py     # Image registration utilities
│   ├── metrics.py          # Volumetric metrics
│   ├── visualization.py    # Visualization tools
│   └── utils.py            # Helper functions
├── config/
│   └── default_config.yaml # Default configuration
├── tests/
│   └── test.py     # test file
├── examples/
│   └── example_usage.py    # Example scripts
├── environment.yml
├──pyproject.toml
└── README.md
```

## Configuration

Edit `config/default_config.yaml` to customize:
- Overlap threshold for lesion matching
- Volume change thresholds
- Registration parameters
- Visualization settings

## License

MIT License
