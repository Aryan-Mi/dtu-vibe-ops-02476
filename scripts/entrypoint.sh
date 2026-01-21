#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "Training Container Entrypoint"
echo "=========================================="

# Initialize DVC without git (required for containerized environments)
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC (no git required)..."
    uv run dvc init --no-scm
fi

# Ensure DVC knows it doesn't need git
uv run dvc config core.no_scm true 2>/dev/null || true

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
    if uv run dvc pull data.dvc 2>&1; then
        echo "  Successfully pulled data.dvc from GCS"
    else
        echo "  Warning: Failed to pull data.dvc explicitly"
        echo "  Trying to pull all DVC-tracked files..."
        if ! uv run dvc pull 2>&1; then
            echo "✗ ERROR: Failed to pull data from GCS"
            echo "  Make sure:"
            echo "  1. Vertex AI service account has access to gs://skin_cancer_data_set"
            echo "  2. DVC configuration files (.dvc/config, data.dvc) are present"
            echo "  3. The data.dvc file points to the correct location in GCS"
            echo "  4. data.dvc hash matches what's stored in GCS"
            exit 1
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

# Execute the training script with all passed arguments
# Note: We don't use 'exec' here so we can run DVC operations after training
uv run src/mlops_project/train.py "$@"
TRAINING_EXIT_CODE=$?

# Check if training was successful
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    echo "=========================================="
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Training completed successfully"
echo "=========================================="
echo ""

# Track models with DVC and push to GCS
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "Tracking models with DVC..."
    
    # Ensure DVC is initialized and configured for no-scm mode
    if [ ! -d ".dvc" ]; then
        echo "  Initializing DVC..."
        uv run dvc init --no-scm
    fi
    
    # Ensure DVC knows it doesn't need git
    uv run dvc config core.no_scm true 2>/dev/null || true
    
    # Check if models.dvc exists, if not create it
    if [ ! -f "models.dvc" ]; then
        echo "  Creating models.dvc..."
        uv run dvc add models/
    else
        echo "  Updating models.dvc..."
        uv run dvc add models/ --force
    fi
    
    # Push models to the models remote
    echo "  Pushing models to GCS (gs://vibeops-models)..."
    if uv run dvc push models.dvc --remote gcs_models_remote 2>&1; then
        echo "  ✓ Successfully pushed models to GCS"
    else
        echo "  ⚠ Warning: Failed to push models to GCS"
        echo "  Models are saved locally but not pushed to remote"
        echo "  Make sure:"
        echo "  1. Vertex AI service account has write access to gs://vibeops-models"
        echo "  2. DVC remote 'gcs_models_remote' is configured correctly"
    fi
else
    echo "⚠ Warning: No models directory found or directory is empty"
    echo "  Skipping DVC tracking..."
fi

echo ""
echo "=========================================="
echo "Training pipeline completed"
echo "=========================================="
