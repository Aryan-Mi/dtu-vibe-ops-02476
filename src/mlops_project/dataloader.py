import random
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from mlops_project.data import CancerDataset, get_transforms

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset_indices(
    metadata_path: str,
    train_ratio: float = 0.525,
    val_ratio: float = 0.175,
    test_ratio: float = 0.30,
    random_seed: int = 42
) -> tuple[list[int], list[int], list[int]]:
    """Split dataset indices into train/val/test by lesion_id to avoid data leakage.

    Args:
        metadata_path: Path to the HAM10000_metadata.csv file
        train_ratio: Fraction of data for training (default: 52.5%)
        val_ratio: Fraction of data for validation (default: 17.5%)
        test_ratio: Fraction of data for testing (default: 30%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)

    Note:
        Splits by lesion_id rather than image_id to prevent the same lesion
        from appearing in multiple splits (data leakage).
    """
    # Set seed for reproducibility
    set_seed(random_seed)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Get unique lesion IDs
    unique_lesions = metadata['lesion_id'].unique()

    # First split: separate test set
    train_val_lesions, test_lesions = train_test_split(
        unique_lesions,
        test_size=test_ratio,
        random_state=random_seed
    )

    # Second split: separate train and validation from remaining data
    # Calculate validation ratio relative to train+val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    train_lesions, val_lesions = train_test_split(
        train_val_lesions,
        test_size=val_ratio_adjusted,
        random_state=random_seed
    )

    # Get indices for each split
    train_indices = metadata[metadata['lesion_id'].isin(train_lesions)].index.tolist()
    val_indices = metadata[metadata['lesion_id'].isin(val_lesions)].index.tolist()
    test_indices = metadata[metadata['lesion_id'].isin(test_lesions)].index.tolist()

    print(f"Dataset split:")
    print(f"  Train: {len(train_indices)} images ({len(train_lesions)} lesions) - {len(train_indices)/len(metadata)*100:.1f}%")
    print(f"  Val:   {len(val_indices)} images ({len(val_lesions)} lesions) - {len(val_indices)/len(metadata)*100:.1f}%")
    print(f"  Test:  {len(test_indices)} images ({len(test_lesions)} lesions) - {len(test_indices)/len(metadata)*100:.1f}%")

    return train_indices, val_indices, test_indices


def create_dataloaders(
    data_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.525,
    val_ratio: float = 0.175,
    test_ratio: float = 0.30,
    random_seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        data_path: Path to the data folder (e.g., 'data/raw/ham10000')
        image_size: Target size for image resizing
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for parallel data loading
        train_ratio: Fraction of data for training (default: 52.5%)
        val_ratio: Fraction of data for validation (default: 17.5%)
        test_ratio: Fraction of data for testing (default: 30%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set global seed for reproducibility
    set_seed(random_seed)

    data_path = data_path
    metadata_path = data_path + "/metadata" + "/HAM10000_metadata.csv"

    # Get split indices
    train_indices, val_indices, test_indices = split_dataset_indices(
        metadata_path=metadata_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    # Create datasets with appropriate transforms
    train_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=True),
        split_indices=train_indices
    )

    val_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=False),
        split_indices=val_indices
    )

    test_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=False),
        split_indices=test_indices
    )

    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=True  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
