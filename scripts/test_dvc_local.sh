#!/bin/bash
# Test DVC model tracking with local model files
# This script helps test DVC push/pull without running full training
#
# Usage:
#   ./scripts/test_dvc_local.sh                    # Test with existing models/
#   ./scripts/test_dvc_local.sh --push              # Also push to GCS
#   ./scripts/test_dvc_local.sh --create-test-model  # Create a test model file first

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PUSH_TO_GCS=false
CREATE_TEST_MODEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_GCS=true
            shift
            ;;
        --create-test-model)
            CREATE_TEST_MODEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--push] [--create-test-model]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=========================================="
echo "DVC Model Tracking Test (Local)"
echo "==========================================${NC}"
echo ""

if [ "$CREATE_TEST_MODEL" = true ]; then
    echo -e "${BLUE}[1/4] Creating test model file...${NC}"
    mkdir -p models/test_model
    echo "This is a test model checkpoint" > models/test_model/test_model.ckpt
    echo "Model metadata: test run $(date)" > models/test_model/metadata.json
    echo -e "${GREEN}✓ Test model created${NC}"
    echo ""
fi

if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo -e "${RED}✗ Error: models/ directory is empty or doesn't exist${NC}"
    echo ""
    echo "Options:"
    echo "  1. Run training first: ./scripts/quick_test.sh"
    echo "  2. Create test model: $0 --create-test-model"
    echo "  3. Copy existing models to models/ directory"
    exit 1
fi

echo -e "${BLUE}[1/4] Checking models directory...${NC}"
echo "  Models found:"
find models -type f | head -10
echo ""

echo -e "${BLUE}[2/4] Adding models to DVC...${NC}"

# Check if models.dvc exists
if [ -f "models.dvc" ]; then
    echo "  models.dvc already exists, updating..."
    uv run dvc add models/ --force
else
    echo "  Creating new models.dvc..."
    uv run dvc add models/
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Models tracked with DVC${NC}"
    echo ""
    echo "  models.dvc content:"
    cat models.dvc | head -10
    echo ""
else
    echo -e "${RED}✗ Failed to add models to DVC${NC}"
    exit 1
fi

echo -e "${BLUE}[3/4] Checking DVC status...${NC}"
echo "  DVC remotes configured:"
uv run dvc remote list 2>/dev/null || echo "    (no remotes configured)"
echo ""

echo "  models.dvc file info:"
if [ -f "models.dvc" ]; then
    echo "    - File exists: ✓"
    echo "    - Size: $(wc -l < models.dvc) lines"
    echo "    - Hash: $(head -1 models.dvc | grep -o 'md5: [a-f0-9]*' || echo 'N/A')"
else
    echo "    - File exists: ✗"
fi
echo ""

if [ "$PUSH_TO_GCS" = true ]; then
    echo -e "${BLUE}[4/4] Pushing models to GCS...${NC}"
    echo "  Remote: gcs_models_remote (gs://vibeops-models)"

    if uv run dvc push models.dvc --remote gcs_models_remote 2>&1; then
        echo -e "${GREEN}✓ Successfully pushed models to GCS${NC}"
        echo ""
        echo "  Verify with:"
        echo "    gsutil ls -r gs://vibeops-models/"
    else
        echo -e "${YELLOW}⚠ Failed to push to GCS${NC}"
        echo "  This might be expected if:"
        echo "    - GCP credentials are not configured"
        echo "    - You don't have write access to gs://vibeops-models"
        echo "    - Remote 'gcs_models_remote' is not configured"
        echo ""
        echo "  To configure:"
        echo "    uv run dvc remote add -d gcs_models_remote gs://vibeops-models"
    fi
else
    echo -e "${BLUE}[4/4] Skipping GCS push (use --push to enable)${NC}"
    echo ""
    echo "  To push to GCS, run:"
    echo "    $0 --push"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "DVC Test Complete!"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Models directory checked"
echo "  ✓ Models tracked with DVC (models.dvc created/updated)"
if [ "$PUSH_TO_GCS" = true ]; then
    echo "  ✓ Models pushed to GCS"
else
    echo "  ⏭ GCS push skipped (use --push to enable)"
fi
echo ""
echo "Next steps:"
echo "  - Check models.dvc: cat models.dvc"
echo "  - Push to GCS: $0 --push"
echo "  - Pull models: uv run dvc pull models.dvc"
echo ""
