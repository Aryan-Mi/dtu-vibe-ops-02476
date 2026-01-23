# Getting Started

Train your first skin lesion classification model in 5 minutes.

## 1. Setup

```bash
git clone https://github.com/Aryan-Mi/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476
uv sync
```

## 2. Get Data

```bash
uv run dvc pull
```

## 3. Train a Model

```bash
# Train EfficientNet (recommended)
uv run python -m mlops_project.train model=efficientnet

# Or train with smaller subset for testing
uv run python -m mlops_project.train model=efficientnet data.subsample_percentage=0.1
```

## 4. Run Inference

```bash
# Start the API server
uv run python -m mlops_project.api

# Test with curl
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

## Configuration Options

Training is configured via Hydra. Override any parameter:

```bash
# Change model
uv run python -m mlops_project.train model=resnet
uv run python -m mlops_project.train model=baseline

# Change hyperparameters
uv run python -m mlops_project.train model.learning_rate=0.001 trainer.max_epochs=50

# Enable W&B logging
uv run python -m mlops_project.train wandb.enabled=true
```

## Next Steps

- [Training Guide](user-guide/training.md) - Advanced training options
- [Architecture](architecture.md) - System design
- [API Reference](api-reference/models.md) - Code documentation
