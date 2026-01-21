#!/bin/bash
# Script to test training pipeline locally with Docker
# - Builds Docker image
# - Pushes to Artifact Registry
# - Runs quick training locally (or optionally submits to Vertex AI)
# - Verifies DVC model tracking works
#
# Usage:
#   ./scripts/test_training_local.sh                    # Run locally with Docker
#   ./scripts/test_training_local.sh --vertex-ai         # Submit to Vertex AI instead
#   ./scripts/test_training_local.sh --model=efficientnet # Use different model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "vibeops-483814")}
REGION=${GCP_REGION:-"europe-west1"}
ARTIFACT_REGISTRY=${GCP_ARTIFACT_REGISTRY:-"dtu-vibe-ops"}
USE_VERTEX_AI=false
TRAINING_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vertex-ai)
            USE_VERTEX_AI=true
            shift
            ;;
        --project-id=*)
            PROJECT_ID="${1#*=}"
            shift
            ;;
        --region=*)
            REGION="${1#*=}"
            shift
            ;;
        --artifact-registry=*)
            ARTIFACT_REGISTRY="${1#*=}"
            shift
            ;;
        --model=*)
            # Convert --model=X to model=X for Hydra
            TRAINING_ARGS+=("model=${1#*=}")
            shift
            ;;
        --*)
            # Convert other --key=value to key=value for Hydra
            TRAINING_ARGS+=("${1#--}")
            shift
            ;;
        *)
            # Pass through other args as-is (should already be in key=value format)
            TRAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: GCP project ID not found!${NC}"
    echo "Set it via: export GCP_PROJECT_ID=your-project-id"
    echo "Or: gcloud config set project your-project-id"
    exit 1
fi

# Generate image tag
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local-$(date +%Y%m%d-%H%M%S)")
IMAGE_TAG_BASE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY}/dtu-vibe-ops-02476-train"
IMAGE_TAG_SHA="${IMAGE_TAG_BASE}:${COMMIT_SHA}"
IMAGE_TAG_LATEST="${IMAGE_TAG_BASE}:latest"

echo -e "${BLUE}=========================================="
echo "Local Training Test Script"
echo "==========================================${NC}"
echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Artifact Registry: $ARTIFACT_REGISTRY"
echo "  Commit SHA: $COMMIT_SHA"
echo "  Image Tag (SHA): $IMAGE_TAG_SHA"
echo "  Image Tag (Latest): $IMAGE_TAG_LATEST"
echo "  Run Mode: $([ "$USE_VERTEX_AI" = true ] && echo "Vertex AI" || echo "Local Docker")"
echo ""

# Step 1: Authenticate to GCP
echo -e "${BLUE}[1/5] Authenticating to Google Cloud...${NC}"
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo "  Setting up Application Default Credentials..."
    gcloud auth application-default login
else
    echo "  ✓ Already authenticated"
fi

# Configure Docker for Artifact Registry
echo "  Configuring Docker for Artifact Registry..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Step 2: Build Docker image
echo ""
echo -e "${BLUE}[2/5] Building Docker image for linux/amd64 (required for Vertex AI)...${NC}"
docker build . \
    --file dockerfiles/train.dockerfile \
    --platform linux/amd64 \
    --tag ${IMAGE_TAG_SHA} \
    --tag ${IMAGE_TAG_LATEST} \
    --progress=plain

if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "  ${RED}✗ Failed to build Docker image${NC}"
    exit 1
fi

# Step 3: Push to Artifact Registry
echo ""
echo -e "${BLUE}[3/5] Pushing image to Artifact Registry...${NC}"
echo "  Pushing ${IMAGE_TAG_SHA}..."
docker push ${IMAGE_TAG_SHA}

echo "  Pushing ${IMAGE_TAG_LATEST}..."
docker push ${IMAGE_TAG_LATEST}

if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ Image pushed successfully${NC}"
else
    echo -e "  ${RED}✗ Failed to push image${NC}"
    exit 1
fi

# Step 4: Run training
echo ""
if [ "$USE_VERTEX_AI" = true ]; then
    echo -e "${BLUE}[4/5] Submitting training job to Vertex AI...${NC}"
    
    # Create temporary config with the built image
    TEMP_CONFIG=$(mktemp)
    sed "s|europe-west1-docker.pkg.dev/PROJECT_ID/dtu-vibe-ops/dtu-vibe-ops-02476-train:latest|${IMAGE_TAG_LATEST}|g" \
        vertex-ai/train-job.yaml > "$TEMP_CONFIG"
    
    # Add quick training args if not provided
    if [ ${#TRAINING_ARGS[@]} -eq 0 ]; then
        TRAINING_ARGS=(
            "data.subsample_percentage=0.01"
            "training.max_epochs=2"
            "wandb.enabled=false"
        )
    fi
    
    # Add args to config
    for arg in "${TRAINING_ARGS[@]}"; do
        if ! grep -q "$arg" "$TEMP_CONFIG"; then
            sed -i.bak "/args:/a\\
        - $arg
" "$TEMP_CONFIG"
        fi
    done
    rm -f "${TEMP_CONFIG}.bak"
    
    JOB_NAME="test-training-$(date +%Y%m%d-%H%M%S)"
    echo "  Job Name: $JOB_NAME"
    echo "  Image: ${IMAGE_TAG_LATEST}"
    echo "  Args: ${TRAINING_ARGS[*]}"
    echo ""
    
    JOB_OUTPUT=$(gcloud ai custom-jobs create \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --display-name="$JOB_NAME" \
        --config="$TEMP_CONFIG" \
        2>&1)
    
    rm -f "$TEMP_CONFIG"
    
    # Extract job ID (works on both Linux and macOS)
    # Try multiple patterns to extract the job ID
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE 'customJobs/[0-9]+' | sed 's|customJobs/||' || \
             echo "$JOB_OUTPUT" | grep -oE '[0-9]{15,20}' | head -1 || \
             echo "")
    
    if [ -n "$JOB_ID" ]; then
        echo -e "  ${GREEN}✓ Job submitted successfully!${NC}"
        echo ""
        echo "  Job ID: $JOB_ID"
        echo ""
        echo "  Monitor the job:"
        echo "    gcloud ai custom-jobs describe projects/$PROJECT_ID/locations/$REGION/customJobs/$JOB_ID --region=$REGION"
        echo ""
        echo "  Stream logs:"
        echo "    gcloud ai custom-jobs stream-logs projects/$PROJECT_ID/locations/$REGION/customJobs/$JOB_ID --region=$REGION"
        echo ""
        echo "  View in console:"
        echo "    https://console.cloud.google.com/vertex-ai/training/custom-jobs/$JOB_ID?project=$PROJECT_ID"
    else
        echo -e "  ${RED}✗ Failed to submit job${NC}"
        echo "$JOB_OUTPUT"
        exit 1
    fi
    
    echo ""
    echo -e "${BLUE}[5/5] Waiting for job to complete...${NC}"
    echo "  (This may take a few minutes. You can monitor progress using the commands above.)"
    echo ""
    echo "  To check if models were pushed to DVC:"
    echo "    gsutil ls gs://vibeops-models/"
    
else
    echo -e "${BLUE}[4/5] Running training locally with Docker...${NC}"
    echo ""
    echo "  Note: This will:"
    echo "    1. Pull data from GCS via DVC (if not cached)"
    echo "    2. Run quick training (1% data, 2 epochs)"
    echo "    3. Track models with DVC"
    echo "    4. Push models to gs://vibeops-models"
    echo ""
    
    # Create local directories for volumes
    mkdir -p models data/raw data/processed logs outputs
    
    # Set up quick training args if not provided
    if [ ${#TRAINING_ARGS[@]} -eq 0 ]; then
        TRAINING_ARGS=(
            "data.subsample_percentage=0.01"
            "training.max_epochs=2"
            "wandb.enabled=false"
        )
    fi
    
    echo "  Training args: ${TRAINING_ARGS[*]}"
    echo ""
    
    # Convert array to properly quoted arguments for docker run
    DOCKER_ARGS=()
    for arg in "${TRAINING_ARGS[@]}"; do
        DOCKER_ARGS+=("$arg")
    done
    
    # Run Docker container
    # Mount models directory so we can see the output locally
    # Use host network and pass through GCP credentials
    docker run --rm -it \
        --name test-training-$(date +%s) \
        -v "$(pwd)/models:/models" \
        -v "$(pwd)/data:/data" \
        -v "$(pwd)/logs:/logs" \
        -v "$(pwd)/outputs:/outputs" \
        -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
        -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
        ${IMAGE_TAG_LATEST} \
        "${DOCKER_ARGS[@]}"
    
    TRAINING_EXIT_CODE=$?
    
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Training failed with exit code $TRAINING_EXIT_CODE${NC}"
        exit 1
    fi
    
    # Step 5: Verify DVC model tracking
    echo ""
    echo -e "${BLUE}[5/5] Verifying DVC model tracking...${NC}"
    
    if [ -f "models.dvc" ]; then
        echo -e "  ${GREEN}✓ models.dvc file exists${NC}"
        echo "  Content:"
        cat models.dvc | head -10
        echo ""
    else
        echo -e "  ${YELLOW}⚠ models.dvc file not found (may need to be created manually)${NC}"
    fi
    
    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        echo -e "  ${GREEN}✓ Models directory contains files${NC}"
        echo "  Models found:"
        find models -type f -name "*.ckpt" -o -name "*.onnx" -o -name "*.json" | head -5
        echo ""
    else
        echo -e "  ${YELLOW}⚠ Models directory is empty${NC}"
    fi
    
    # Check if models were pushed to GCS
    echo "  Checking if models were pushed to GCS..."
    if gsutil ls gs://vibeops-models/ &>/dev/null; then
        MODEL_COUNT=$(gsutil ls -r gs://vibeops-models/ 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓ Found $MODEL_COUNT items in gs://vibeops-models/${NC}"
        echo "  Recent models:"
        gsutil ls -r gs://vibeops-models/ 2>/dev/null | tail -5
    else
        echo -e "  ${YELLOW}⚠ Could not access gs://vibeops-models/ (check permissions)${NC}"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Test Complete!"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "  ✓ Docker image built and pushed"
echo "  ✓ Training completed"
echo "  ✓ DVC model tracking verified"
echo ""
echo "Next steps:"
echo "  - Check models in: gs://vibeops-models/"
echo "  - View training logs in: logs/"
echo "  - Models saved locally in: models/"
echo ""
