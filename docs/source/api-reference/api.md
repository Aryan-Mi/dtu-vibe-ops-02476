# API Endpoints Reference

API documentation for the FastAPI inference server.

## API Module

::: mlops_project.api

## Response Models

### PredictionResponse

::: mlops_project.api.PredictionResponse

## Endpoints

### GET /

Root endpoint returning welcome message.

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

### POST /predict

Classify a skin lesion image.

**Request:** `multipart/form-data` with image file

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

**Response:**
```json
{
  "prediction": "malignant",
  "confidence": 0.847,
  "probabilities": {"benign": 0.153, "malignant": 0.847}
}
```
