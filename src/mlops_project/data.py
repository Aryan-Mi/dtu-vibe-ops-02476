from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Mapping of diagnosis codes to class indices
DX_TO_CLASS = {
    "nv": 0,  # Melanocytic nevi
    "mel": 1,  # Melanoma
    "bkl": 2,  # Benign keratosis
    "bcc": 3,  # Basal cell carcinoma
    "akiec": 4,  # Actinic keratoses
    "vasc": 5,  # Vascular lesions
    "df": 6,  # Dermatofibroma
}

CLASS_TO_DX = {v: k for k, v in DX_TO_CLASS.items()}


class CancerDataset(Dataset):
    """HAM10000 skin lesion dataset."""

    def __init__(
        self,
        data_path: str = "../../data/raw/HAM10000",
        metadata_file: str = "HAM10000_metadata.csv",
        transform: Callable | None = None,
        split_indices: list[int] | None = None,
    ) -> None:
        """
        Args:
            data_path: Path to the data folder containing images/ and metadata/
            metadata_file: Name of the metadata CSV file
            transform: Optional transform to apply to images
            split_indices: Optional list of indices for train/val/test split
        """
        self.data_path = Path(data_path)
        self.images_path = self.data_path / "images"
        self.metadata_path = self.data_path / "metadata" / metadata_file
        self.transform = transform

        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)

        # Filter by split indices if provided
        if split_indices is not None:
            self.metadata = self.metadata.iloc[split_indices].reset_index(drop=True)

        # Map diagnosis to class labels
        self.metadata["label"] = self.metadata["dx"].map(DX_TO_CLASS)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a given sample from the dataset."""
        row = self.metadata.iloc[index]

        # Load image
        image_id = row["image_id"]
        image_path = self.images_path / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Get label
        label = int(row["label"])

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

    def preprocess(self, output_folder: Path, target_size: int = 224) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create preprocessing transform (resize + save)
        preprocess_transform = transforms.Resize(
            (target_size, target_size), interpolation=InterpolationMode.BILINEAR, antialias=True
        )

        print(f"Preprocessing {len(self)} images to {target_size}x{target_size}...")

        for idx in range(len(self)):
            row = self.metadata.iloc[idx]
            image_id = row["image_id"]

            # Load and preprocess
            image_path = self.images_path / f"{image_id}.jpg"
            image = Image.open(image_path).convert("RGB")
            image = preprocess_transform(image)

            # Save preprocessed image
            output_path = output_folder / f"{image_id}.jpg"
            image.save(output_path, quality=95)

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self)} images")

        # Copy metadata
        output_metadata_path = output_folder / "metadata.csv"
        self.metadata.to_csv(output_metadata_path, index=False)
        print(f"Saved metadata to {output_metadata_path}")


def get_transforms(image_size: int = 224, augment: bool = True) -> transforms.Compose:
    """Get image transforms for training or validation.

    Args:
        image_size: Target size for images
        augment: Whether to apply data augmentation (for training)

    Returns:
        Composed transforms
    """
    if augment:
        # Training transforms with augmentation
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    # Validation/test transforms without augmentation
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess(data_path: str, output_folder: Path, target_size: int = 224) -> None:
    """Preprocess raw HAM10000 data.

    Args:
        data_path: Path to raw data folder
        output_folder: Path to save preprocessed data
        target_size: Target image size
    """
    print("Preprocessing data...")
    dataset = CancerDataset(data_path)
    dataset.preprocess(output_folder, target_size)


if __name__ == "__main__":
    typer.run(preprocess)
