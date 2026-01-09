"""Subsampling module for HAM10000 dataset."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

app = typer.Typer()


def subsample_dataset(
    metadata_path: Path | str,
    percentage: float,
    random_seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Subsample HAM10000 dataset while maintaining relative distribution of dx categories.

    Args:
        metadata_path: Path to HAM10000_metadata CSV file.
        percentage: Fraction of data to subsample (0.0 to 1.0, e.g., 0.1 for 10%).
        random_seed: Optional integer for reproducible sampling.

    Returns:
        Dictionary with dx categories as keys, each containing a list of image records.
        Each record includes: image_id, lesion_id, dx, dx_type, age, sex, localization, dataset.

    Raises:
        FileNotFoundError: If metadata_path does not exist.
        ValueError: If percentage is not between 0 and 1.
    """
    if not isinstance(metadata_path, Path):
        metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)

    if not 0 < percentage <= 1:
        msg = f"Percentage must be between 0 and 1, got {percentage}"
        raise ValueError(msg)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Read CSV file
    df = pd.read_csv(metadata_path)

    # Calculate category distributions
    category_counts = df["dx"].value_counts()
    total_samples = len(df)
    category_percentages = category_counts / total_samples

    # Initialize result dictionary
    result: dict[str, list[dict[str, Any]]] = {}

    # Subsample each category maintaining relative distribution
    for dx_category, category_count in category_counts.items():
        # Calculate target count for this category
        target_count = int(total_samples * percentage * category_percentages[dx_category])

        # Ensure we don't sample more than available
        target_count = min(target_count, category_count)

        # Get all rows for this category
        category_df = df[df["dx"] == dx_category]

        # Sample rows
        if target_count > 0:
            sampled_df = category_df.sample(n=target_count, random_state=random_seed)

            # Convert to list of dictionaries with all features
            result[dx_category] = sampled_df[
                ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization", "dataset"]
            ].to_dict("records")
        else:
            result[dx_category] = []

    return result


@app.command()
def subsample_cli(
    metadata_path: Path = typer.Argument(..., help="Path to HAM10000_metadata CSV file"),
    percentage: float = typer.Argument(..., help="Fraction of data to subsample (0.0 to 1.0)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional JSON file to save results"),
    seed: int | None = typer.Option(None, "--seed", "-s", help="Random seed for reproducibility"),
) -> None:
    """CLI wrapper for subsample_dataset function."""
    try:
        result = subsample_dataset(metadata_path, percentage, random_seed=seed)

        # Print summary
        total_sampled = sum(len(images) for images in result.values())
        typer.echo(f"Subsampled {total_sampled} images ({percentage*100:.1f}% of dataset)")
        typer.echo("\nCategory distribution:")
        for dx_category, images in result.items():
            typer.echo(f"  {dx_category}: {len(images)} images")

        # Save to JSON if output path specified
        if output:
            with Path(output).open("w") as f:
                json.dump(result, f, indent=2)
            typer.echo(f"\nResults saved to: {output}")

    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    # CLI Usage Examples:

    # Basic usage (subsample 10% of dataset):
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.1

    # With random seed for reproducibility:
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.1 --seed 42

    # Save results to JSON file:
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.1 --output subsample_results.json

    # Combine options (seed + output):
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.2 --seed 123 --output results.json

    # Different percentages:
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.05   # 5%
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.25   # 25%
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.5    # 50%

    # Short form options:
    #   python -m src.mlops_project.subsample data/HAM10000_metadata 0.1 -s 42 -o output.json
    app()
