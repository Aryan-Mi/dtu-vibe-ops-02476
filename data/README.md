# Skin Cancer Dataset

This directory contains skin cancer image datasets for classification and segmentation tasks.

## Directory Structure

```
data/
├── raw/                          # Original, unprocessed data
│   ├── ham10000/                 # HAM10000 dataset
│   │   ├── images/               # 10,015 dermoscopic images
│   │   ├── segmentations/        # Lesion segmentation masks
│   │   └── metadata/             # HAM10000_metadata.csv
│   └── isic2018_test/            # ISIC 2018 Challenge Task 3 Test Set
│       ├── images/               # 1,511 test images
│       └── metadata/             # Ground truth and interaction benefit CSVs
└── processed/                    # Preprocessed data (train/val/test splits, augmentations, etc.)
```

## Datasets

### HAM10000 (Human Against Machine with 10000 training images)
- **Images**: 10,015 dermoscopic images of pigmented skin lesions
- **Classes**: 7 types of skin lesions
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)
- **Metadata**: Patient demographics, lesion location, diagnosis type
- **Segmentations**: Binary masks for lesion boundaries

### ISIC 2018 Task 3 Test Set
- **Images**: 1,511 test images
- **Purpose**: Test set for skin lesion classification
- **Ground Truth**: Available in metadata folder
- **Interaction Benefit**: Human-AI collaboration study data

## Usage

The datasets are version-controlled with DVC. To push/pull:

```bash
# Track data with DVC
uv run dvc add data

# Push to cloud storage
uv run dvc push

# Pull from cloud storage (on other machines)
uv run dvc pull
```

## References

- HAM10000: https://doi.org/10.1038/sdata.2018.161
- ISIC 2018 Challenge: https://challenge2018.isic-archive.com/
