# MLOps Skin Lesion Classification

Welcome to the **MLOps Skin Lesion Classification Pipeline** - a production-ready ML system by **DTU Group 36** for the MLOps course (02476).

## Overview

This pipeline classifies dermatoscopic images of skin lesions to estimate **malignant vs. benign diagnosis**, demonstrating MLOps best practices from data handling to cloud deployment.

### Key Features

- **Three-Tier Architecture**: Baseline CNN → ResNet → EfficientNet
- **Modern ML Stack**: PyTorch Lightning, Hydra, Weights & Biases
- **Complete MLOps Pipeline**: DVC versioning, CI/CD, containerized deployment
- **Production Ready**: FastAPI server, Google Cloud deployment

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Aryan-Mi/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476
uv sync

# Train a model
uv run python -m mlops_project.train model=efficientnet

# Start inference server
uv run python -m mlops_project.api
```

[Installation guide →](installation.md) | [Tutorial →](getting-started.md)

## Dataset

- **Source**: HAM10000 (Harvard Dataverse)
- **Images**: 10,015 dermatoscopic images
- **Task**: Binary classification (malignant vs. benign)

## Technology Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch, PyTorch Lightning |
| Configuration | Hydra |
| Data/Model Versioning | DVC, Google Cloud Storage |
| Experiment Tracking | Weights & Biases |
| API | FastAPI, ONNX Runtime |
| Deployment | Docker, Google Cloud Run |
| Dev Tools | uv, Ruff, pytest, GitHub Actions |

## Team - DTU Group 36

- Aryan Mirzazadeh
- Mohamad Marwan Summakieh
- Trinity Sara McConnachie Evans
- Vladyslav Horbatenko
- Yuen Yi Hui

## Links

- [GitHub Repository](https://github.com/Aryan-Mi/dtu-vibe-ops-02476)
- [Architecture](architecture.md)
- [API Reference](api-reference/models.md)
