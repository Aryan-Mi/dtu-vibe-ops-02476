# Testing

## Run Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_model.py -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── test_model.py      # Model architecture tests
├── test_data.py       # Data loading tests
├── test_train.py      # Training pipeline tests
└── test_api.py        # API endpoint tests
```

## Writing Tests

```python
import pytest
from mlops_project.model import EfficientNet

def test_model_forward():
    model = EfficientNet(num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 2)
```

## CI Integration

Tests run automatically on:
- Every push to any branch
- Every pull request
- Scheduled nightly builds

See `.github/workflows/tests.yaml` for configuration.
