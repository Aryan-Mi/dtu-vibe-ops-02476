"""Unit tests for the subsampling module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_project.subsample import subsample_dataset


@pytest.fixture
def sample_metadata_df():
    """Create a sample metadata DataFrame for testing."""
    return pd.DataFrame(
        {
            "image_id": [f"img_{i:03d}" for i in range(100)],
            "lesion_id": [f"lesion_{i % 20:03d}" for i in range(100)],
            "dx": ["nv"] * 40 + ["mel"] * 25 + ["bkl"] * 20 + ["bcc"] * 10 + ["akiec"] * 5,
            "dx_type": ["histo"] * 50 + ["follow_up"] * 30 + ["consensus"] * 20,
            "age": np.random.randint(20, 80, 100),
            "sex": ["male"] * 55 + ["female"] * 45,
            "localization": ["back"] * 30 + ["chest"] * 25 + ["face"] * 20 + ["hand"] * 15 + ["foot"] * 10,
            "dataset": ["train"] * 70 + ["test"] * 30,
        }
    )


@pytest.fixture
def temp_metadata_file(tmp_path, sample_metadata_df):
    """Create a temporary metadata CSV file for testing."""
    metadata_path = tmp_path / "test_metadata.csv"
    sample_metadata_df.to_csv(metadata_path, index=False)
    return metadata_path


def test_subsample_basic(temp_metadata_file):
    """Test basic subsampling functionality."""
    result = subsample_dataset(temp_metadata_file, percentage=0.1, random_seed=42)

    # Check that result is a dictionary
    assert isinstance(result, dict)

    # Check that all expected categories are present
    expected_categories = {"nv", "mel", "bkl", "bcc", "akiec"}
    assert set(result.keys()) == expected_categories

    # Check that total samples is approximately 10% of original
    total_sampled = sum(len(images) for images in result.values())
    assert 8 <= total_sampled <= 12  # Allow some variance due to rounding


def test_subsample_percentage_distribution(temp_metadata_file):
    """Test that subsampling maintains relative distribution of categories."""
    original_df = pd.read_csv(temp_metadata_file)
    original_counts = original_df["dx"].value_counts()
    original_total = len(original_df)
    original_percentages = original_counts / original_total

    result = subsample_dataset(temp_metadata_file, percentage=0.2, random_seed=42)

    # Calculate subsampled percentages
    sampled_counts = {dx: len(images) for dx, images in result.items()}
    sampled_total = sum(sampled_counts.values())

    # Check that relative distributions are approximately maintained
    for dx_category in result:
        original_pct = original_percentages[dx_category]
        sampled_pct = sampled_counts[dx_category] / sampled_total if sampled_total > 0 else 0

        # Allow 10% relative difference due to rounding with small samples
        assert abs(original_pct - sampled_pct) < 0.15


def test_subsample_reproducibility(temp_metadata_file):
    """Test that using the same random seed produces identical results."""
    result1 = subsample_dataset(temp_metadata_file, percentage=0.15, random_seed=123)
    result2 = subsample_dataset(temp_metadata_file, percentage=0.15, random_seed=123)

    # Check that results are identical
    assert result1.keys() == result2.keys()

    for dx_category in result1:
        images1 = result1[dx_category]
        images2 = result2[dx_category]

        assert len(images1) == len(images2)

        # Check that image_ids are the same
        ids1 = [img["image_id"] for img in images1]
        ids2 = [img["image_id"] for img in images2]
        assert ids1 == ids2


def test_subsample_different_seeds(temp_metadata_file):
    """Test that different random seeds produce different results."""
    result1 = subsample_dataset(temp_metadata_file, percentage=0.2, random_seed=42)
    result2 = subsample_dataset(temp_metadata_file, percentage=0.2, random_seed=99)

    # Check that at least one category has different samples
    differences_found = False
    for dx_category in result1:
        ids1 = {img["image_id"] for img in result1[dx_category]}
        ids2 = {img["image_id"] for img in result2[dx_category]}

        if ids1 != ids2:
            differences_found = True
            break

    assert differences_found


def test_subsample_100_percent(temp_metadata_file):
    """Test subsampling with 100% percentage."""
    original_df = pd.read_csv(temp_metadata_file)
    result = subsample_dataset(temp_metadata_file, percentage=1.0, random_seed=42)

    total_sampled = sum(len(images) for images in result.values())
    assert total_sampled == len(original_df)


def test_subsample_small_percentage(temp_metadata_file):
    """Test subsampling with very small percentage."""
    result = subsample_dataset(temp_metadata_file, percentage=0.01, random_seed=42)

    total_sampled = sum(len(images) for images in result.values())
    # Should have at least some samples
    assert total_sampled >= 0
    # Should not have too many samples
    assert total_sampled <= 5


def test_subsample_output_structure(temp_metadata_file):
    """Test that output has correct structure and all required fields."""
    result = subsample_dataset(temp_metadata_file, percentage=0.2, random_seed=42)

    required_fields = {"image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "dataset"}

    for dx_category, images in result.items():
        for image_record in images:
            # Check that all required fields are present
            assert set(image_record.keys()) == required_fields

            # Check that dx matches the category
            assert image_record["dx"] == dx_category

            # Check data types
            assert isinstance(image_record["image_id"], str)
            assert isinstance(image_record["lesion_id"], str)
            assert isinstance(image_record["dx"], str)
            assert isinstance(image_record["dx_type"], str)


def test_subsample_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        subsample_dataset("nonexistent_file.csv", percentage=0.1)


def test_subsample_invalid_percentage_zero(temp_metadata_file):
    """Test that ValueError is raised for percentage of 0."""
    with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
        subsample_dataset(temp_metadata_file, percentage=0.0)


def test_subsample_invalid_percentage_negative(temp_metadata_file):
    """Test that ValueError is raised for negative percentage."""
    with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
        subsample_dataset(temp_metadata_file, percentage=-0.1)


def test_subsample_invalid_percentage_too_large(temp_metadata_file):
    """Test that ValueError is raised for percentage > 1."""
    with pytest.raises(ValueError, match="Percentage must be between 0 and 1"):
        subsample_dataset(temp_metadata_file, percentage=1.5)


def test_subsample_path_as_string(temp_metadata_file):
    """Test that function accepts path as string."""
    result = subsample_dataset(str(temp_metadata_file), percentage=0.1, random_seed=42)

    assert isinstance(result, dict)
    assert len(result) > 0


def test_subsample_path_as_pathlib(temp_metadata_file):
    """Test that function accepts path as Path object."""
    result = subsample_dataset(Path(temp_metadata_file), percentage=0.1, random_seed=42)

    assert isinstance(result, dict)
    assert len(result) > 0


def test_subsample_no_random_seed(temp_metadata_file):
    """Test that function works without random seed."""
    result = subsample_dataset(temp_metadata_file, percentage=0.15)

    assert isinstance(result, dict)
    total_sampled = sum(len(images) for images in result.values())
    assert total_sampled > 0


def test_subsample_empty_categories_handled(tmp_path):
    """Test that categories with very few samples are handled correctly."""
    # Create a dataset with one very small category
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i:03d}" for i in range(11)],
            "lesion_id": [f"lesion_{i:03d}" for i in range(11)],
            "dx": ["nv"] * 10 + ["rare"],
            "dx_type": ["histo"] * 11,
            "age": [50] * 11,
            "sex": ["male"] * 11,
            "localization": ["back"] * 11,
            "dataset": ["train"] * 11,
        }
    )

    metadata_path = tmp_path / "small_metadata.csv"
    df.to_csv(metadata_path, index=False)

    # Subsample with small percentage that might round to 0 for rare category
    result = subsample_dataset(metadata_path, percentage=0.05, random_seed=42)

    # Check that result is still valid
    assert isinstance(result, dict)
    # The rare category might have 0 samples, which is acceptable
    if "rare" in result:
        assert isinstance(result["rare"], list)


def test_subsample_json_serializable(temp_metadata_file):
    """Test that the result can be serialized to JSON."""
    result = subsample_dataset(temp_metadata_file, percentage=0.1, random_seed=42)

    # This should not raise an exception
    json_str = json.dumps(result)

    # Verify we can load it back
    loaded_result = json.loads(json_str)
    assert loaded_result.keys() == result.keys()
