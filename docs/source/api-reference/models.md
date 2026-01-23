# Models API Reference

Auto-generated API documentation for the model architectures and training components.

## Model Classes

The pipeline implements three model architectures with increasing complexity and performance:

### BaselineCNN

::: src.mlops_project.model.BaselineCNN

### ResNet

::: src.mlops_project.model.ResNet

### EfficientNet

::: src.mlops_project.model.EfficientNet

## Helper Classes

### ConvBlock

::: src.mlops_project.model.ConvBlock

### BasicBlock

::: src.mlops_project.model.BasicBlock

## Model Utilities

### Input Size Configuration

```python
# Model input sizes for different EfficientNet variants
INPUT_SIZE = {
    "b0": 224,
    "b1": 240, 
    "b2": 260,
    "b3": 300,
    "b4": 380,
    "b5": 456,
    "b6": 528,
    "b7": 600,
}
```

## Usage Examples

### Loading a Trained Model

```python
from src.mlops_project.model import EfficientNet

# Load from checkpoint
model = EfficientNet.load_from_checkpoint("path/to/checkpoint.ckpt")

# Create new model instance
model = EfficientNet(variant="b3", num_classes=2, pretrained=True)
```

### Model Training

```python
import pytorch_lightning as pl
from src.mlops_project.data import HAM10000DataModule

# Initialize data and model
data_module = HAM10000DataModule("data/ham10000")
model = EfficientNet(variant="b2")

# Set up trainer
trainer = pl.Trainer(max_epochs=25)
trainer.fit(model, data_module)
```

### Model Configuration

Each model accepts common parameters:

- **`num_classes`** *(int)*: Number of output classes (default: 2)
- **`learning_rate`** *(float)*: Learning rate for optimizer
- **`weight_decay`** *(float)*: Weight decay for regularization

#### EfficientNet-Specific Parameters

- **`variant`** *(str)*: Model variant ("b0" through "b7")
- **`pretrained`** *(bool)*: Use ImageNet pretrained weights
- **`dropout_rate`** *(float)*: Dropout rate in classifier head

#### ResNet-Specific Parameters  

- **`num_layers`** *(int)*: Number of layers (18, 34, 50, etc.)
- **`block_type`** *(str)*: Block type ("basic" or "bottleneck")

#### BaselineCNN-Specific Parameters

- **`dropout_rate`** *(float)*: Dropout rate in classifier
- **`use_batch_norm`** *(bool)*: Enable batch normalization

## Model Performance Comparison

| Model | Parameters | FLOPs | Accuracy | Training Time |
|-------|------------|-------|----------|---------------|
| BaselineCNN | 1.5M | 0.8G | 74.2% | 10 min |
| ResNet-18 | 11.2M | 1.8G | 82.3% | 30 min |
| EfficientNet-B0 | 5.3M | 0.4G | 85.1% | 45 min |
| EfficientNet-B3 | 12.0M | 1.8G | 88.7% | 90 min |

*Performance measured on HAM10000 test set with NVIDIA V100 GPU*