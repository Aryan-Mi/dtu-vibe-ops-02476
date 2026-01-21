#!/bin/bash
# Quick test script - runs training locally without pushing to registry
# Useful for rapid iteration and testing DVC setup
#
# Usage:
#   ./scripts/quick_test.sh
#   ./scripts/quick_test.sh --model=efficientnet

set -e

# Default quick training args
TRAINING_ARGS=(
    "data.subsample_percentage=0.01"
    "training.max_epochs=2"
    "wandb.enabled=false"
)

# Add any additional args passed to script
TRAINING_ARGS+=("$@")

echo "=========================================="
echo "Quick Local Training Test"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Build Docker image locally"
echo "  2. Run quick training (1% data, 2 epochs)"
echo "  3. Test DVC model tracking"
echo ""
echo "Training args: ${TRAINING_ARGS[*]}"
echo ""

echo "[1/3] Building Docker image..."
docker build . \
    --file dockerfiles/train.dockerfile \
    --platform linux/amd64 \
    --tag train:test \
    --progress=plain

if [ $? -ne 0 ]; then
    echo "✗ Failed to build Docker image"
    exit 1
fi

echo "✓ Docker image built"
echo ""

# Create local directories
mkdir -p models data/raw data/processed logs outputs

# Run training
echo "[2/3] Running training..."
docker run --rm -it \
    --name quick-test-$(date +%s) \
    -v "$(pwd)/models:/models" \
    -v "$(pwd)/data:/data" \
    -v "$(pwd)/logs:/logs" \
    -v "$(pwd)/outputs:/outputs" \
    -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
    train:test \
    "${TRAINING_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi

echo ""
echo "[3/3] Verifying results..."
echo ""

if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "✓ Models directory contains files:"
    find models -type f | head -5
else
    echo "⚠ Models directory is empty"
fi

if [ -f "models.dvc" ]; then
    echo "✓ models.dvc file exists"
else
    echo "⚠ models.dvc file not found"
fi

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "  - models/          (trained models)"
echo "  - logs/            (training logs)"
echo "  - models.dvc       (DVC tracking file)"
echo ""
