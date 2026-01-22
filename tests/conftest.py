"""Shared pytest fixtures for unit tests."""

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from mlops_project.data import DX_TO_CLASS


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame for testing."""
    n_samples = 100
    diagnoses = list(DX_TO_CLASS.keys())

    return pd.DataFrame(
        {
            "image_id": [f"ISIC_{i:07d}" for i in range(n_samples)],
            "lesion_id": [f"lesion_{i % 20:03d}" for i in range(n_samples)],
            "dx": [diagnoses[i % len(diagnoses)] for i in range(n_samples)],
            "dx_type": ["histo"] * 50 + ["follow_up"] * 30 + ["consensus"] * 20,
            "age": np.random.randint(20, 80, n_samples),
            "sex": ["male"] * 55 + ["female"] * 45,
            "localization": ["back"] * 30 + ["chest"] * 25 + ["face"] * 20 + ["hand"] * 15 + ["foot"] * 10,
        }
    )


@pytest.fixture
def temp_data_dir(tmp_path, sample_metadata_df):
    """Create a temporary data directory with metadata and sample images."""
    # Create directory structure
    data_dir = tmp_path / "data"
    images_dir = data_dir / "images"
    metadata_dir = data_dir / "metadata"

    images_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    # Save metadata
    metadata_path = metadata_dir / "HAM10000_metadata.csv"
    sample_metadata_df.to_csv(metadata_path, index=False)

    # Create dummy images
    for image_id in sample_metadata_df["image_id"]:
        # Create a simple RGB image
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        img.save(images_dir / f"{image_id}.jpg")

    return data_dir
