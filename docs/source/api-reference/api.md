# API Endpoints Reference

Complete documentation for the FastAPI inference server endpoints.

## API Server Module

::: src.mlops_project.api

## Endpoints

### Health and Status

#### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1642612345.67,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is unhealthy

#### GET /health/detailed

Comprehensive health check with system metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1642612345.67,
  "version": "1.0.0",
  "system": {
    "cpu_usage": 25.3,
    "memory_usage": 67.8,
    "gpu_available": true,
    "gpu_memory_used": 2.1
  },
  "model": {
    "loaded": true,
    "name": "efficientnet",
    "variant": "b3",
    "input_size": [300, 300]
  }
}
```

### Model Information

#### GET /model/info

Get information about the loaded model.

**Response:**
```json
{
  "name": "efficientnet",
  "variant": "b3",
  "version": "1.0.0",
  "input_size": [300, 300],
  "num_classes": 2,
  "parameters": 12000000,
  "architecture": "EfficientNet-B3 with custom classification head",
  "training": {
    "dataset": "HAM10000",
    "epochs": 30,
    "accuracy": 0.887
  }
}
```

### Prediction

#### POST /predict

Classify a skin lesion image.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Form data with image file

**Parameters:**
- `file` (required): Image file (JPEG, PNG)
- `confidence_threshold` (optional): Minimum confidence for prediction (default: 0.5)
- `return_probabilities` (optional): Include class probabilities (default: true)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@skin_lesion.jpg" \
     -F "confidence_threshold=0.7"
```

**Python Example:**
```python
import requests

with open("skin_lesion.jpg", "rb") as f:
    files = {"file": f}
    data = {"confidence_threshold": 0.7}
    response = requests.post("http://localhost:8000/predict", files=files, data=data)

result = response.json()
print(f"Prediction: {result['prediction']}")
```

**Response:**
```json
{
  "prediction": "malignant",
  "confidence": 0.847,
  "class_probabilities": {
    "benign": 0.153,
    "malignant": 0.847
  },
  "metadata": {
    "image_size": [300, 300],
    "preprocessing_time_ms": 12.4,
    "inference_time_ms": 45.2,
    "total_time_ms": 57.6
  },
  "model_info": {
    "name": "efficientnet",
    "variant": "b3",
    "version": "1.0.0"
  }
}
```

**Error Responses:**

*Invalid Image Format (400):*
```json
{
  "error": "Invalid image format",
  "detail": "Supported formats: JPEG, PNG",
  "code": "INVALID_FORMAT"
}
```

*File Too Large (413):*
```json
{
  "error": "File too large",
  "detail": "Maximum file size: 10MB",
  "code": "FILE_TOO_LARGE"
}
```

*Model Error (500):*
```json
{
  "error": "Model inference failed",
  "detail": "Internal model error occurred",
  "code": "MODEL_ERROR"
}
```

#### POST /predict/batch

Process multiple images in a single request.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** Multiple image files

**Parameters:**
- `files` (required): Array of image files
- `confidence_threshold` (optional): Minimum confidence threshold

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": "benign",
      "confidence": 0.923,
      "class_probabilities": {
        "benign": 0.923,
        "malignant": 0.077
      }
    },
    {
      "filename": "image2.jpg", 
      "prediction": "malignant",
      "confidence": 0.781,
      "class_probabilities": {
        "benign": 0.219,
        "malignant": 0.781
      }
    }
  ],
  "summary": {
    "total_images": 2,
    "malignant_count": 1,
    "benign_count": 1,
    "average_confidence": 0.852,
    "processing_time_ms": 98.4
  }
}
```

## Authentication

### API Key Authentication (Optional)

If authentication is enabled:

**Header:**
```
Authorization: Bearer your-api-key
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Authorization: Bearer your-api-key" \
     -F "file=@skin_lesion.jpg"
```

## Rate Limiting

Default rate limits:
- **100 requests per minute** per IP address
- **1000 requests per hour** per IP address
- **10MB maximum** file size per request

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642612400
```

## Error Handling

### Standard Error Format

All errors follow this format:
```json
{
  "error": "Brief error description",
  "detail": "Detailed error message", 
  "code": "ERROR_CODE",
  "timestamp": "2024-01-23T10:30:45Z"
}
```

### Common Error Codes

| Code | Description | Status |
|------|-------------|---------|
| `INVALID_FORMAT` | Unsupported image format | 400 |
| `FILE_TOO_LARGE` | File exceeds size limit | 413 |
| `INVALID_IMAGE` | Corrupted or invalid image | 400 |
| `MODEL_ERROR` | Model inference failure | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `UNAUTHORIZED` | Invalid or missing API key | 401 |
| `SERVER_ERROR` | Internal server error | 500 |

## Performance Metrics

### Response Times

| Endpoint | Avg Response Time | 95th Percentile |
|----------|------------------|-----------------|
| `/health` | 2ms | 5ms |
| `/model/info` | 3ms | 8ms |
| `/predict` (single) | 45ms | 80ms |
| `/predict/batch` (5 images) | 180ms | 320ms |

### Throughput

- **Single prediction**: ~22 requests/second
- **Batch prediction**: ~28 images/second
- **Concurrent requests**: Up to 50 with proper scaling

*Benchmarks on Intel Xeon CPU with NVIDIA V100 GPU*

## Client Libraries

### Python Client

```python
import requests
from typing import Optional, Dict, Any

class SkinLesionAPI:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def predict(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"confidence_threshold": confidence_threshold}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                data=data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()

# Usage
client = SkinLesionAPI("http://localhost:8000")
result = client.predict("skin_lesion.jpg")
print(f"Prediction: {result['prediction']}")
```

### JavaScript Client

```javascript
class SkinLesionAPI {
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {};
    }
    
    async predict(file, confidenceThreshold = 0.5) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence_threshold', confidenceThreshold);
        
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: this.headers,
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`, {
            headers: this.headers
        });
        return response.json();
    }
}

// Usage
const client = new SkinLesionAPI('http://localhost:8000');
const fileInput = document.getElementById('file-input');
const result = await client.predict(fileInput.files[0]);
console.log('Prediction:', result.prediction);
```

## API Versioning

The API uses URL path versioning:
- `/v1/predict` - Current stable version
- `/v2/predict` - Future version with additional features

Version headers are included in responses:
```
API-Version: 1.0.0
Supported-Versions: 1.0.0, 1.1.0
```