# Running Inference

## Start the API Server

```bash
uv run python -m mlops_project.api
```

Server runs at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@skin_lesion.jpg"
```

### Response Format
```json
{
  "prediction": "malignant",
  "confidence": 0.847,
  "probabilities": {
    "benign": 0.153,
    "malignant": 0.847
  }
}
```

## Python Client

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Docker

```bash
docker build -f dockerfiles/api.dockerfile -t skin-lesion-api .
docker run -p 8000:8000 skin-lesion-api
```
