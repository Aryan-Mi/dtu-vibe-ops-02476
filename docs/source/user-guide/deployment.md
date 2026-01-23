# Deployment

## Docker Build

```bash
# Build API image
docker build -f dockerfiles/api.dockerfile -t skin-lesion-api .

# Run locally
docker run -p 8000:8000 skin-lesion-api
```

## Google Cloud Run

### Push to Container Registry

```bash
# Tag image
docker tag skin-lesion-api gcr.io/PROJECT_ID/skin-lesion-api

# Push
docker push gcr.io/PROJECT_ID/skin-lesion-api
```

### Deploy

```bash
gcloud run deploy skin-lesion-api \
  --image gcr.io/PROJECT_ID/skin-lesion-api \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

## CI/CD

Deployment is automated via GitHub Actions:

1. Push to `main` branch triggers build
2. Tests run automatically
3. Docker image is built and pushed
4. Cloud Run deployment is updated

See `.github/workflows/` for workflow definitions.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to ONNX model file |
| `PORT` | Server port (default: 8000) |
| `LOG_LEVEL` | Logging level (default: INFO) |
