# Deployment Guide

Complete guide for deploying the skin lesion classification pipeline to production environments.

## Local Deployment

### Docker Deployment

#### Build and Run Container

```bash
# Build Docker image
docker build -t skin-lesion-api .

# Run container locally
docker run -p 8000:8000 skin-lesion-api

# Run with custom model
docker run -p 8000:8000 -v $(pwd)/models:/app/models \
    skin-lesion-api --model-path /app/models/efficientnet/best.ckpt
```

#### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/efficientnet/best.ckpt
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

```bash
# Start with docker-compose
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3
```

## Cloud Deployment

### Google Cloud Run

#### Deploy with gcloud CLI

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/skin-lesion-api

# Deploy to Cloud Run
gcloud run deploy skin-lesion-api \
    --image gcr.io/PROJECT_ID/skin-lesion-api \
    --platform managed \
    --region europe-west1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 100 \
    --max-instances 10
```

#### Cloud Build Configuration

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/skin-lesion-api', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/skin-lesion-api']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'skin-lesion-api'
      - '--image'
      - 'gcr.io/$PROJECT_ID/skin-lesion-api'
      - '--region'
      - 'europe-west1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'

images:
  - 'gcr.io/$PROJECT_ID/skin-lesion-api'
```

### Azure Container Instances

```bash
# Create resource group
az group create --name skin-lesion-rg --location westeurope

# Deploy container
az container create \
    --resource-group skin-lesion-rg \
    --name skin-lesion-api \
    --image your-registry/skin-lesion-api:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --environment-variables MODEL_PATH=/app/models/best.ckpt
```

### AWS ECS/Fargate

```json
{
  "family": "skin-lesion-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "skin-lesion-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/skin-lesion-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/efficientnet/best.ckpt"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/skin-lesion-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Production Optimizations

### Performance Tuning

#### ONNX Runtime Configuration

```python
# Optimized ONNX session
import onnxruntime as ort

# Configure providers for performance
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
    }),
    'CPUExecutionProvider'
]

# Session options for optimization
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

# Create optimized session
ort_session = ort.InferenceSession(
    "models/efficientnet_optimized.onnx",
    sess_options=session_options,
    providers=providers
)
```

#### FastAPI Performance Settings

```python
# app_config.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI(
    title="Skin Lesion Classification API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Adjust based on CPU cores
        loop="uvloop",  # High-performance event loop
        http="httptools",  # Fast HTTP parser
        access_log=False,  # Disable for performance
        log_level="warning"
    )
```

### Load Balancing

#### Nginx Configuration

```nginx
# nginx.conf
upstream skin_lesion_api {
    least_conn;
    server api1:8000 weight=3 max_fails=3 fail_timeout=30s;
    server api2:8000 weight=3 max_fails=3 fail_timeout=30s;
    server api3:8000 weight=2 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    # Gzip compression
    gzip on;
    gzip_types
        text/plain
        text/css
        text/js
        application/json
        application/javascript;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://skin_lesion_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # File upload size
        client_max_body_size 10M;
    }
    
    location /health {
        proxy_pass http://skin_lesion_api/health;
        access_log off;
    }
}
```

## Monitoring and Observability

### Application Metrics

```python
# monitoring.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'prediction'])
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
error_counter = Counter('prediction_errors_total', 'Prediction errors', ['error_type'])

@prediction_latency.time()
def timed_prediction(image):
    try:
        result = model.predict(image)
        prediction_counter.labels(
            model='efficientnet',
            prediction=result['prediction']
        ).inc()
        return result
    except Exception as e:
        error_counter.labels(error_type=type(e).__name__).inc()
        raise

# Start metrics server
start_http_server(8001)
```

### Health Checks

```python
# health.py
from fastapi import FastAPI, HTTPException
import psutil
import torch

app = FastAPI()

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Check system resources
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    health_status["checks"]["system"] = {
        "memory_usage": memory.percent,
        "cpu_usage": cpu_percent,
        "memory_available_gb": memory.available / (1024**3)
    }
    
    # Check model availability
    try:
        if hasattr(app.state, 'model'):
            health_status["checks"]["model"] = {"status": "loaded"}
        else:
            health_status["checks"]["model"] = {"status": "not_loaded"}
    except Exception as e:
        health_status["checks"]["model"] = {"status": "error", "error": str(e)}
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        health_status["checks"]["gpu"] = {
            "available": True,
            "memory_used_gb": gpu_memory
        }
    
    # Determine overall health
    if memory.percent > 90 or cpu_percent > 95:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@app.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe"""
    try:
        # Quick model inference test
        dummy_input = torch.randn(1, 3, 224, 224)
        _ = app.state.model(dummy_input)
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(e)})
```

## Security Considerations

### API Security

```python
# security.py
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        if payload.get("exp", 0) < datetime.utcnow().timestamp():
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(file: UploadFile, token: dict = Depends(verify_token)):
    # Protected prediction endpoint
    pass
```

### Input Validation

```python
# validation.py
from PIL import Image
import io
from fastapi import HTTPException

def validate_image(file_content: bytes) -> Image.Image:
    """Validate and sanitize uploaded images"""
    try:
        # Check file size (max 10MB)
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Validate image format
        image = Image.open(io.BytesIO(file_content))
        
        # Check image dimensions
        if image.width > 2048 or image.height > 2048:
            raise HTTPException(status_code=400, detail="Image dimensions too large")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
```

## Continuous Deployment

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}
      
      - name: Build and Deploy
        run: |
          gcloud builds submit --config cloudbuild.yaml
          
      - name: Run Health Check
        run: |
          curl -f https://your-api-url.com/health || exit 1
```

---

*Continue to the [API Reference](../api-reference/models.md) for detailed technical documentation.*