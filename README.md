# MLOps Skin Lesion Classification

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://aryan-mi.github.io/dtu-vibe-ops-02476/)
[![Tests](https://github.com/Aryan-Mi/dtu-vibe-ops-02476/actions/workflows/tests.yaml/badge.svg)](https://github.com/Aryan-Mi/dtu-vibe-ops-02476/actions/workflows/tests.yaml)

**DTU Course: MLOps 02476 (Winter 2026) - Group 36**

## Project Goal

Build an end-to-end MLOps pipeline that classifies dermatoscopic images of skin lesions to estimate **malignant vs. benign** diagnosis.

## Documentation

📚 **[Full Documentation](https://aryan-mi.github.io/dtu-vibe-ops-02476/)**

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Aryan-Mi/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476
uv sync

# Pull data
uv run dvc pull

# Train model
uv run python -m mlops_project.train model=efficientnet

# Start API server
uv run python -m mlops_project.api
```

## Dataset

**HAM10000** (Harvard Dataverse) - 10,015 dermatoscopic images mapped to binary classification (malignant vs. benign).

## Model

**EfficientNet** with transfer learning from ImageNet pretrained weights, fine-tuned on HAM10000 dataset.

## Technology Stack

| Category | Tools |
|----------|-------|
| ML Framework | PyTorch, PyTorch Lightning |
| Configuration | Hydra |
| Data Versioning | DVC, Google Cloud Storage |
| Experiment Tracking | Weights & Biases |
| API | FastAPI, ONNX Runtime |
| Deployment | Docker, Google Cloud Run |
| Dev Tools | uv, Ruff, pytest, GitHub Actions |

## Team - Group 36

- Aryan Mirzazadeh
- Mohamad Marwan Summakieh
- Trinity Sara McConnachie Evans
- Vladyslav Horbatenko
- Yuen Yi Hui

## Links

- [Documentation](https://aryan-mi.github.io/dtu-vibe-ops-02476/)
- [GitHub Repository](https://github.com/Aryan-Mi/dtu-vibe-ops-02476)
