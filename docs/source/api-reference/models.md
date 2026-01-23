# Models API Reference

API documentation for the model architectures.

## Model Classes

### BaselineCNN

::: mlops_project.model.BaselineCNN

### ResNet

::: mlops_project.model.ResNet

### EfficientNet

::: mlops_project.model.EfficientNet

## Helper Classes

### ConvBlock

::: mlops_project.model.ConvBlock

### ResidualBlock

::: mlops_project.model.ResidualBlock

## Usage Examples

```python
from mlops_project.model import EfficientNet, ResNet, BaselineCNN

# Create model instances
model = EfficientNet(variant="b3", num_classes=2, pretrained=True)
model = ResNet(num_classes=2)
model = BaselineCNN(num_classes=2)

# Load from checkpoint
model = EfficientNet.load_from_checkpoint("path/to/checkpoint.ckpt")
```
