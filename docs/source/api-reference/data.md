# Data Handling API Reference

Auto-generated documentation for data processing, loading, and augmentation components.

## Dataset Classes

### HAM10000Dataset

::: src.mlops_project.data.HAM10000Dataset

### HAM10000DataModule  

::: src.mlops_project.dataloader.HAM10000DataModule

## Data Processing Functions

### Image Preprocessing

::: src.mlops_project.data.preprocess_image

### Data Transforms

::: src.mlops_project.data.get_transforms

## Subsampling Utilities

::: src.mlops_project.subsample.create_subsample

::: src.mlops_project.subsample.save_subsample_metadata

## Usage Examples

### Basic Data Loading

```python
from src.mlops_project.data import HAM10000Dataset
from src.mlops_project.dataloader import HAM10000DataModule

# Direct dataset usage
dataset = HAM10000Dataset("data/ham10000", split="train")
image, label = dataset[0]

# Lightning DataModule usage
data_module = HAM10000DataModule(
    data_dir="data/ham10000",
    batch_size=32,
    image_size=224,
    num_workers=4
)
data_module.setup("fit")
train_loader = data_module.train_dataloader()
```

### Custom Data Transforms

```python
from torchvision import transforms
from src.mlops_project.data import HAM10000Dataset

# Define custom transforms
custom_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Use with dataset
dataset = HAM10000Dataset(
    data_dir="data/ham10000",
    split="train", 
    transform=custom_transforms
)
```

### Data Subsampling

```python
from src.mlops_project.subsample import create_subsample

# Create 10% subsample maintaining class balance
subsample_metadata = create_subsample(
    data_dir="data/ham10000",
    percentage=0.1,
    stratified=True,
    seed=42
)

# Use subsampled data
dataset = HAM10000Dataset(
    data_dir="data/ham10000",
    subsample_metadata=subsample_metadata
)
```

## Data Configuration

### DataModule Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | Required | Path to HAM10000 dataset |
| `batch_size` | int | 32 | Batch size for dataloaders |
| `image_size` | int | 224 | Target image size (square) |
| `num_workers` | int | 4 | Number of dataloader workers |
| `pin_memory` | bool | True | Pin memory for faster GPU transfer |
| `persistent_workers` | bool | False | Keep workers alive between epochs |

### Augmentation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizontal_flip` | float | 0.5 | Probability of horizontal flip |
| `vertical_flip` | float | 0.0 | Probability of vertical flip |
| `rotation_degrees` | int | 0 | Maximum rotation degrees |
| `color_jitter` | float | 0.0 | Color jitter strength |
| `normalize` | bool | True | Apply ImageNet normalization |

## Dataset Statistics

### HAM10000 Class Distribution

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| **nv** | 6705 | 67.0% | Melanocytic nevi (benign) |
| **mel** | 1113 | 11.1% | Melanoma (malignant) |
| **bkl** | 1099 | 11.0% | Benign keratosis (benign) |
| **bcc** | 514 | 5.1% | Basal cell carcinoma (malignant) |
| **akiec** | 327 | 3.3% | Actinic keratoses (malignant) |
| **vasc** | 142 | 1.4% | Vascular lesions (benign) |
| **df** | 115 | 1.1% | Dermatofibroma (benign) |

### Binary Classification Mapping

```python
# Mapping to binary classification (malignant vs benign)
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']  # ~19.5% of dataset
BENIGN_CLASSES = ['nv', 'bkl', 'vasc', 'df']  # ~80.5% of dataset
```

### Image Properties

- **Format**: JPEG
- **Resolution**: Variable (typically 450×600 to 1024×768)
- **Color Space**: RGB
- **File Size**: 50-500 KB per image
- **Total Dataset Size**: ~1.5 GB

## Data Validation

### Quality Checks

The data pipeline includes automatic validation:

```python
def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """Validate dataset integrity and properties"""
    validation_results = {
        "total_images": 0,
        "missing_images": [],
        "corrupt_images": [],
        "class_distribution": {},
        "image_size_stats": {},
    }
    
    # Implementation details in source code
    return validation_results
```

### Common Data Issues

1. **Missing Images**: Some image files referenced in metadata may be missing
2. **Corrupt Files**: Occasional JPEG corruption requiring validation
3. **Size Variation**: Images have different resolutions requiring consistent resizing
4. **Class Imbalance**: Significant imbalance between benign and malignant cases

## Performance Optimization

### DataLoader Configuration

```python
# Optimized dataloader settings
data_module = HAM10000DataModule(
    data_dir="data/ham10000",
    batch_size=32,
    num_workers=8,          # 2x number of CPU cores
    pin_memory=True,        # Faster GPU transfer
    persistent_workers=True, # Reduce worker startup time
    prefetch_factor=2       # Prefetch batches
)
```

### Memory Usage

| Configuration | Memory Usage | Loading Speed |
|---------------|--------------|---------------|
| Default | ~2GB | Baseline |
| Optimized Workers | ~4GB | +40% faster |
| Persistent Workers | ~6GB | +60% faster |
| + Prefetching | ~8GB | +80% faster |