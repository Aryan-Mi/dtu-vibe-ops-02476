#!/bin/bash
# Script to submit a Vertex AI Custom Job for training
#
# Usage:
#   ./scripts/submit_training_job.sh
#   ./scripts/submit_training_job.sh --model=efficientnet --data.subsample_percentage=0.1
#
# Environment variables:
#   GCP_PROJECT_ID: GCP project ID (defaults to gcloud config value)
#   GCP_REGION: GCP region (defaults to europe-west1)
#   GCS_MODELS_BUCKET: GCS bucket for models (defaults to vibeops-models)

set -e

# Get project ID from environment or gcloud config
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "vibeops-483814")}
REGION=${GCP_REGION:-"europe-west1"}
GCS_BUCKET=${GCS_MODELS_BUCKET:-"vibeops-models"}
JOB_NAME="dtu-vibe-ops-training-$(date +%Y%m%d-%H%M%S)"

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: GCP project ID not found!"
    echo "Set it via: export GCP_PROJECT_ID=your-project-id"
    echo "Or: gcloud config set project your-project-id"
    exit 1
fi

echo "=========================================="
echo "Submitting Vertex AI Training Job"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Job Name: $JOB_NAME"
echo "GCS Models Bucket: $GCS_BUCKET"
echo ""

# Check if config file exists
CONFIG_FILE="vertex-ai/train-job.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create a temporary config file with PROJECT_ID replaced
TEMP_CONFIG=$(mktemp)
sed "s/PROJECT_ID/$PROJECT_ID/g" "$CONFIG_FILE" > "$TEMP_CONFIG"

# Add GCS bucket to env vars in temp config if not already set
if ! grep -q "GCS_MODELS_BUCKET" "$TEMP_CONFIG"; then
    # Add GCS_MODELS_BUCKET env var before args section
    sed -i.bak "/args:/i\\
      env:\\
        - name: GCS_MODELS_BUCKET\\
          value: \"$GCS_BUCKET\"
" "$TEMP_CONFIG"
    rm -f "${TEMP_CONFIG}.bak"
fi

# Add any additional args passed to the script
ARGS=()
if [ $# -gt 0 ]; then
    # Add args to temp config
    for arg in "$@"; do
        ARGS+=("$arg")
    done

    # Append args to the args section in temp config
    if grep -q "args:" "$TEMP_CONFIG"; then
        # Remove empty args: line if no args exist, or append to existing args
        if grep -A 1 "args:" "$TEMP_CONFIG" | grep -q "^\s*-\s*model="; then
            # Args already exist, append
            for arg in "${ARGS[@]}"; do
                sed -i.bak "/args:/a\\
        - $arg
" "$TEMP_CONFIG"
            done
        else
            # No args yet, add them
            for arg in "${ARGS[@]}"; do
                sed -i.bak "/args:/a\\
        - $arg
" "$TEMP_CONFIG"
            done
        fi
        rm -f "${TEMP_CONFIG}.bak"
    fi
fi

echo "Submitting job..."
echo ""

# Submit the job
JOB_OUTPUT=$(gcloud ai custom-jobs create \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --display-name="$JOB_NAME" \
    --config="$TEMP_CONFIG" \
    2>&1)

# Clean up temp file
rm -f "$TEMP_CONFIG"

# Extract job ID from output
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'customJobs/\K[0-9]+' || echo "")

if [ -n "$JOB_ID" ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Job ID: $JOB_ID"
    echo ""
    echo "Monitor the job:"
    echo "  gcloud ai custom-jobs describe projects/$PROJECT_ID/locations/$REGION/customJobs/$JOB_ID --region=$REGION"
    echo ""
    echo "Stream logs:"
    echo "  gcloud ai custom-jobs stream-logs projects/$PROJECT_ID/locations/$REGION/customJobs/$JOB_ID --region=$REGION"
    echo ""
    echo "View in console:"
    echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs/$JOB_ID?project=$PROJECT_ID"
else
    echo ""
    echo "Job submission output:"
    echo "$JOB_OUTPUT"
    echo ""
    echo "⚠ Could not extract job ID. Check the output above for errors."
fi
