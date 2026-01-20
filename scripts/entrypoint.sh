#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "Training Container Entrypoint"
echo "=========================================="

# If data already exists locally
DATA_PATH="data/raw/ham10000"
METADATA_PATH="${DATA_PATH}/metadata/HAM10000_metadata.csv"

if [ -f "$METADATA_PATH" ]; then
    echo "Data already exists at $METADATA_PATH"
    echo "  Skipping DVC pull..."
else
    echo "Data not found at $METADATA_PATH"
    echo "  Pulling data from GCS using DVC..."
    
    # Pull data using DVC
    # DVC will use Application Default Credentials (ADC) from Vertex AI service account
    echo "  Pulling data.dvc (full dataset)..."
    if dvc pull data.dvc 2>&1; then
        echo "  Successfully pulled data.dvc from GCS"
    else
        echo "  Warning: Failed to pull data.dvc explicitly"
        echo "  Trying to pull all DVC-tracked files..."
        if ! dvc pull 2>&1; then
            echo "✗ ERROR: Failed to pull data from GCS"
            echo "  Make sure:"
            echo "  1. Vertex AI service account has access to gs://skin_cancer_data_set"
            echo "  2. DVC configuration files (.dvc/config, data.dvc) are present"
            echo "  3. The data.dvc file points to the correct location in GCS"
            echo "  4. data.dvc hash matches what's stored in GCS"
            exit 1
        fi
    fi

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Execute the training script with all passed arguments
exec uv run src/mlops_project/train.py "$@"
