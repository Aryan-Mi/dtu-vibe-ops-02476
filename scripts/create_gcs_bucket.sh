#!/bin/bash
# Script to create GCS bucket for model storage
#
# Usage:
#   ./scripts/create_gcs_bucket.sh
#   ./scripts/create_gcs_bucket.sh --bucket-name=my-models-bucket --region=europe-west1

set -e

BUCKET_NAME=${GCS_MODELS_BUCKET:-"vibeops-models"}
REGION=${GCP_REGION:-"europe-west1"}
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "")}


while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket-name=*)
            BUCKET_NAME="${1#*=}"
            shift
            ;;
        --region=*)
            REGION="${1#*=}"
            shift
            ;;
        --project-id=*)
            PROJECT_ID="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--bucket-name=BUCKET] [--region=REGION] [--project-id=PROJECT_ID]"
            exit 1
            ;;
    esac
done

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: GCP project ID not found!"
    echo "Set it via: export GCP_PROJECT_ID=your-project-id"
    echo "Or: gcloud config set project your-project-id"
    exit 1
fi

echo "=========================================="
echo "Creating GCS Bucket for Model Storage"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Bucket Name: $BUCKET_NAME"
echo "Region: $REGION"
echo ""

# Check if bucket already exists
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo "Bucket gs://$BUCKET_NAME already exists"
    echo "Skipping creation..."
else
    echo "Creating bucket gs://$BUCKET_NAME..."
    gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$BUCKET_NAME"
    echo "Bucket created successfully"
fi

# Versioning
echo ""
echo "Enabling versioning on bucket..."
gsutil versioning set on "gs://$BUCKET_NAME"
echo "Versioning enabled"

# Set lifecycle policy
echo ""
echo "Setting lifecycle policy..."
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 30,
          "numNewerVersions": 5
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json "gs://$BUCKET_NAME"
rm -f /tmp/lifecycle.json
echo "Lifecycle policy set (keep 5 newest versions, delete after 90 days)"

echo ""
echo "=========================================="
echo "Bucket Setup Complete"
echo "=========================================="
echo "Bucket: gs://$BUCKET_NAME"
echo ""
echo "You can now use this bucket for model storage by setting:"
echo "  export GCS_MODELS_BUCKET=$BUCKET_NAME"
echo "  Or set paths.gcs_bucket in config.yaml"
echo ""
