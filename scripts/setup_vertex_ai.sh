#!/bin/bash
set -e

PROJECT_ID="vibeops-483814"
PROJECT_NUMBER="267598825904"
REGION="europe-west1"
DATA_BUCKET="skin_cancer_data_set"
MODELS_BUCKET="vibeops-models"

echo "Setting up Vertex AI permissions..."

# Set project
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Enable APIs
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable storage.googleapis.com --project=$PROJECT_ID

# Grant Vertex AI service account permissions
VAI_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Read access to data bucket
gsutil iam ch serviceAccount:$VAI_SA:roles/storage.objectViewer gs://$DATA_BUCKET

# Write access to models bucket
./scripts/create_gcs_bucket.sh
gsutil iam ch serviceAccount:$VAI_SA:roles/storage.objectAdmin gs://$MODELS_BUCKET

echo "✓ Setup complete!"
