# Data Handling API Reference

API documentation for data processing and loading.

## Dataset Classes

### CancerDataset

::: mlops_project.data.CancerDataset

## Data Functions

### get_transforms

::: mlops_project.data.get_transforms

### preprocess

::: mlops_project.data.preprocess

## DataLoader Functions

### create_dataloaders

::: mlops_project.dataloader.create_dataloaders

### split_dataset_indices

::: mlops_project.dataloader.split_dataset_indices

### subsample_dataloader

::: mlops_project.dataloader.subsample_dataloader

## Usage Examples

```python
from mlops_project.data import CancerDataset, get_transforms
from mlops_project.dataloader import create_dataloaders

# Create dataset
transform = get_transforms(image_size=224, augment=True)
dataset = CancerDataset(data_dir="data/", transform=transform)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/",
    batch_size=32,
    image_size=224
)
```
