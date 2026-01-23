# Training API Reference

API documentation for training and evaluation.

## Training Functions

### train

::: mlops_project.train.train

### train_model

::: mlops_project.train.train_model

### create_model

::: mlops_project.train.create_model

## Evaluation Functions

### evaluate

::: mlops_project.evaluate.evaluate

## Model Export

### export_model_to_onnx

::: mlops_project.train.export_model_to_onnx

## Usage Examples

```python
from mlops_project.train import train_model, create_model
from mlops_project.evaluate import evaluate

# Train with Hydra config
# uv run python -m mlops_project.train model=efficientnet

# Evaluate a trained model
results = evaluate(
    model_path="models/best_model.ckpt",
    data_dir="data/"
)
```
