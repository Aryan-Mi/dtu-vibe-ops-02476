# Training Guide

## Basic Training

```bash
uv run python -m mlops_project.train model=efficientnet
```

## Model Options

| Model | Command | Description |
|-------|---------|-------------|
| EfficientNet | `model=efficientnet` | Best accuracy, transfer learning |
| ResNet | `model=resnet` | Good balance of speed/accuracy |
| BaselineCNN | `model=baseline` | Fast, simple baseline |

## Hyperparameters

```bash
# Learning rate
uv run python -m mlops_project.train model.learning_rate=0.0001

# Batch size
uv run python -m mlops_project.train data.batch_size=64

# Epochs
uv run python -m mlops_project.train trainer.max_epochs=50

# Early stopping
uv run python -m mlops_project.train trainer.patience=10
```

## Data Subsampling

For faster iteration during development:

```bash
# Use 10% of data
uv run python -m mlops_project.train data.subsample_percentage=0.1

# Use 1% for quick tests
uv run python -m mlops_project.train data.subsample_percentage=0.01
```

## Experiment Tracking

Enable Weights & Biases logging:

```bash
uv run python -m mlops_project.train wandb.enabled=true wandb.project=skin-lesion
```

## GPU Training

```bash
# Single GPU
uv run python -m mlops_project.train trainer.accelerator=gpu

# Multiple GPUs
uv run python -m mlops_project.train trainer.accelerator=gpu trainer.devices=2
```

## Output

Training produces:
- `models/` - Model checkpoints
- `outputs/` - Hydra logs and configs
- W&B dashboard (if enabled)
