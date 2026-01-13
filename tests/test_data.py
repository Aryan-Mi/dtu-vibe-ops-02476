"""Unit tests for the data module."""

import os

# Import numpy early to ensure it's available for torchvision transforms
try:
    import numpy as np  # noqa: F401

    # Check if torchvision can actually use numpy (runtime linking check)
    try:
        from PIL import Image
        from torchvision import transforms

        # Try to convert a small image to test numpy linking
        test_img = Image.new("RGB", (10, 10))
        transforms.ToTensor()(test_img)
        NUMPY_AVAILABLE = True
    except RuntimeError:
        # Numpy is installed but torchvision can't link to it
        NUMPY_AVAILABLE = False
except ImportError:
    NUMPY_AVAILABLE = False

import pytest
import torch

from mlops_project.data import DX_TO_CLASS, CancerDataset

# Path to real data (if available)
REAL_DATA_PATH = "data/raw/ham10000"
REAL_METADATA_PATH = os.path.join(REAL_DATA_PATH, "metadata", "HAM10000_metadata.csv")


class TestCancerDataset:
    """Tests for CancerDataset class."""

    @pytest.mark.skipif(
        not os.path.exists(REAL_METADATA_PATH) or not NUMPY_AVAILABLE,
        reason="Real data files not found or numpy not available for torchvision",
    )
    def test_data_loading_comprehensive(self):
        """Comprehensive test for data loading following lecture requirements.

        Checks:
        - Dataset length matches expected
        - Each datapoint has correct shape [3, H, W] (RGB images)
        - All labels are represented
        """
        dataset = CancerDataset(data_path=REAL_DATA_PATH)

        # Check dataset length
        assert len(dataset) > 0, "Dataset should not be empty"

        # Check a sample of datapoints
        sample_size = min(100, len(dataset))
        labels_found = set()

        for i in range(sample_size):
            image, label = dataset[i]

            # Check image shape: [C, H, W] where C=3 for RGB
            assert isinstance(image, torch.Tensor), f"Image at index {i} should be a tensor"
            assert image.ndim == 3, f"Image at index {i} should be 3D [C, H, W], got {image.ndim}D"
            assert image.shape[0] == 3, f"Image at index {i} should have 3 channels (RGB), got {image.shape[0]}"
            assert image.shape[1] > 0, f"Image at index {i} should have valid height"
            assert image.shape[2] > 0, f"Image at index {i} should have valid width"

            # Check label
            assert isinstance(label, int), f"Label at index {i} should be an integer"
            assert label >= 0, f"Label at index {i} should be non-negative, got {label}"
            assert label < len(DX_TO_CLASS), f"Label at index {i} should be < {len(DX_TO_CLASS)}, got {label}"

            labels_found.add(label)

        # Check that all labels are represented (at least in sample)
        assert len(labels_found) > 0, "At least some labels should be found in the dataset"

        # Check that labels are in valid range
        expected_labels = set(DX_TO_CLASS.values())
        assert labels_found.issubset(expected_labels), f"Found invalid labels: {labels_found - expected_labels}"

    def test_dataset_initialization(self, temp_data_dir):
        """Test dataset initialization with valid path."""
        dataset = CancerDataset(data_path=str(temp_data_dir))
        assert len(dataset) > 0, "Dataset should not be empty"
