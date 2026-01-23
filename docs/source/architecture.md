# System Architecture

Overview of the MLOps Skin Lesion Classification Pipeline architecture.

## High-Level Architecture

![System Architecture](assets/images/system-architecture.png)

## Model Architecture

We use **EfficientNet** with transfer learning for skin lesion classification:

- **Backbone**: EfficientNet-B0/B3 pretrained on ImageNet
- **Classification Head**: Custom fully-connected layers for binary classification
- **Training**: Fine-tuning with PyTorch Lightning

## Training Pipeline

![Training Pipeline](assets/images/training-pipeline.png)

## Data Pipeline

```mermaid
graph LR
    A[HAM10000 Dataset] --> B[DVC]
    B --> C[Google Cloud Storage]
    C --> D[Local Cache]
    D --> E[DataLoader]
    E --> F[Model Training]
```

## Inference Pipeline

```mermaid
graph LR
    A[Client Request] --> B[FastAPI]
    B --> C[Image Preprocessing]
    C --> D[ONNX Runtime]
    D --> E[Postprocessing]
    E --> F[JSON Response]
```

## Deployment Architecture

```mermaid
graph TD
    A[GitHub] --> B[GitHub Actions CI/CD]
    B --> C[Docker Build]
    C --> D[Container Registry]
    D --> E[Google Cloud Run]
    E --> F[Load Balancer]
    F --> G[Users]
```

## Technology Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | PyTorch Lightning | Training orchestration |
| Configuration | Hydra | Experiment configs |
| Data Versioning | DVC | Dataset management |
| Experiment Tracking | Weights & Biases | Metrics logging |
| API Server | FastAPI | Inference endpoint |
| Model Runtime | ONNX Runtime | Fast inference |
| Containerization | Docker | Reproducible deployment |
| Cloud Platform | Google Cloud Run | Serverless hosting |
| CI/CD | GitHub Actions | Automated testing |

## Directory Structure

```
dtu-vibe-ops-02476/
├── src/mlops_project/     # Source code
│   ├── model.py           # EfficientNet model
│   ├── data.py            # Dataset handling
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation
│   └── api.py             # FastAPI server
├── configs/               # Hydra configurations
├── data/                  # Dataset (DVC managed)
├── models/                # Trained models
├── tests/                 # Test suite
├── dockerfiles/           # Container definitions
└── docs/                  # Documentation
```
