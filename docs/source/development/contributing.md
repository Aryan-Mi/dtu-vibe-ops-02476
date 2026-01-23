# Contributing to MLOps Skin Lesion Classification

Welcome! This guide helps you contribute to the MLOps Skin Lesion Classification pipeline. We appreciate contributions of all kinds - from bug reports to new features.

## Quick Start for Contributors

### 1. Set Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476

# Set up development environment
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run pytest tests/ -v
```

### 2. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run quality checks
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run tests
uv run pytest tests/ --cov=src

# Commit changes (pre-commit hooks will run automatically)
git add .
git commit -m "feat: add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

## Development Guidelines

### Code Style and Standards

We use **Ruff** for code formatting and linting with strict configuration:

```bash
# Format code
uv run ruff format src/ tests/

# Check for linting issues
uv run ruff check src/ tests/

# Fix auto-fixable issues
uv run ruff check --fix src/ tests/
```

#### Code Style Rules

- **Line length**: 120 characters maximum
- **Import organization**: Automatic with Ruff's isort integration
- **Docstrings**: Google-style format required for all public functions
- **Type hints**: Required for all function signatures
- **Variable naming**: snake_case for variables, PascalCase for classes

#### Example Code Style

```python
from typing import List, Optional, Tuple
import torch
import pytorch_lightning as pl
from pathlib import Path

class SkinLesionClassifier(pl.LightningModule):
    """Skin lesion classification model using PyTorch Lightning.
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        
    Examples:
        >>> model = SkinLesionClassifier(num_classes=2, learning_rate=0.001)
        >>> model.configure_optimizers()
    """
    
    def __init__(
        self, 
        num_classes: int = 2, 
        learning_rate: float = 0.001,
        weight_decay: float = 0.01
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Build model architecture
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the model architecture."""
        # Implementation details
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Implementation
        pass
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step implementation."""
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
```

### Testing Requirements

#### Test Coverage

We maintain **>90% test coverage**. All new code must include tests:

```bash
# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

#### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_data.py            # Data loading tests
├── test_models.py          # Model architecture tests
├── test_training.py        # Training pipeline tests
├── test_api.py             # API endpoint tests
├── test_integration.py     # End-to-end tests
└── fixtures/               # Test data and fixtures
    ├── sample_images/
    └── mock_data/
```

#### Writing Tests

```python
import pytest
import torch
from unittest.mock import MagicMock, patch
from src.mlops_project.model import EfficientNet

class TestEfficientNet:
    """Test suite for EfficientNet model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return EfficientNet(variant="b0", num_classes=2, pretrained=False)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.num_classes == 2
        assert model.variant == "b0"
        assert not model.pretrained
    
    def test_forward_pass_shape(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
    
    @pytest.mark.parametrize("variant,expected_size", [
        ("b0", 224),
        ("b1", 240),
        ("b2", 260),
    ])
    def test_input_sizes(self, variant, expected_size):
        """Test different variants have correct input sizes."""
        from src.mlops_project.model import INPUT_SIZE
        assert INPUT_SIZE[variant] == expected_size
    
    def test_training_step(self, model):
        """Test training step works correctly."""
        batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
        
        # Mock trainer
        model.trainer = MagicMock()
        
        loss = model.training_step(batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

# Integration test example
class TestTrainingPipeline:
    """Integration tests for training pipeline."""
    
    def test_end_to_end_training(self, tmp_path):
        """Test complete training pipeline."""
        # Create minimal test data
        test_config = {
            "model": {"name": "baseline_cnn"},
            "training": {"max_epochs": 1, "fast_dev_run": True},
            "data": {"subsample_percentage": 0.001}
        }
        
        # Run training
        with patch('src.mlops_project.train.main') as mock_train:
            mock_train.return_value = None
            # Test passes if no exception is raised
            
    @pytest.mark.slow
    def test_model_convergence(self):
        """Test that model can overfit small dataset (slow test)."""
        # This test verifies the model can learn
        # Run with: pytest -m slow
        pass
```

### Documentation Standards

#### Docstring Format

Use **Google-style docstrings** for all public functions:

```python
def train_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Train a PyTorch model.
    
    Trains the provided model using the given data loader and optimizer
    for the specified number of epochs.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader providing training batches
        optimizer: Optimizer for updating model parameters
        epochs: Number of training epochs (default: 10)
        device: Device to run training on (default: auto-detect)
        
    Returns:
        Dictionary containing training metrics:
        - 'final_loss': Final training loss value
        - 'avg_loss': Average loss across all epochs
        
    Raises:
        ValueError: If epochs is less than 1
        RuntimeError: If training fails due to device/memory issues
        
    Examples:
        >>> model = torch.nn.Linear(10, 1)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> metrics = train_model(model, data_loader, optimizer, epochs=5)
        >>> print(f"Final loss: {metrics['final_loss']:.4f}")
        
    Note:
        This function modifies the model in-place. Use model.eval()
        after training for inference.
    """
    if epochs < 1:
        raise ValueError("Epochs must be at least 1")
        
    # Implementation...
    return {"final_loss": 0.1, "avg_loss": 0.2}
```

#### Code Comments

```python
class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        # Store configuration for later use
        self.config = config
        
        # Initialize image transforms based on config
        self.transforms = self._build_transforms()
        
        # Cache for preprocessed images (memory optimization)
        self._image_cache: Dict[str, torch.Tensor] = {}
    
    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of images for model input."""
        
        # Convert numpy arrays to PIL Images for consistent processing
        pil_images = [Image.fromarray(img) for img in images]
        
        # Apply transforms (resize, normalize, etc.)
        processed = []
        for img in pil_images:
            # Apply cached transforms to avoid recomputation
            tensor = self.transforms(img)
            processed.append(tensor)
        
        # Stack into batch tensor (N, C, H, W)
        return torch.stack(processed, dim=0)
```

## Contributing Types

### 1. Bug Reports

When reporting bugs, include:

- **Environment details**: OS, Python version, uv version
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected vs actual behavior**
- **Error messages**: Full stack traces
- **Configuration**: Model/training configs if relevant

**Bug Report Template:**

```markdown
## Bug Description
Brief description of the issue

## Environment
- OS: macOS 14.2
- Python: 3.12.1
- uv: 0.4.30
- GPU: NVIDIA RTX 3080

## Steps to Reproduce
1. Run `uv run src/mlops_project/train.py model=efficientnet`
2. Error occurs at epoch 5

## Expected Behavior
Training should complete successfully

## Actual Behavior
Training fails with CUDA out of memory error

## Error Output
```
RuntimeError: CUDA out of memory. Tried to allocate 2.73 GiB
```

## Additional Context
Using default configuration with batch_size=32
```

### 2. Feature Requests

For new features:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Implementation ideas**: Technical details if you have them

### 3. Code Contributions

#### Pull Request Process

1. **Fork** the repository
2. **Create feature branch** from main
3. **Implement changes** following coding standards
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run quality checks** and tests
7. **Submit pull request** with clear description

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No breaking changes (or marked as breaking)
```

## Development Tools

### Useful Commands

```bash
# Development server with hot reload
uv run uvicorn src.mlops_project.api:app --reload

# Run specific test file
uv run pytest tests/test_models.py -v

# Run tests with specific markers
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m integration  # Only integration tests

# Profile code performance
uv run python -m cProfile -s cumulative src/mlops_project/train.py

# Generate API documentation
uv run mkdocs serve

# Build documentation
uv run mkdocs build
```

### IDE Setup

#### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
        "source.fixAll.ruff": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/.venv": true,
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

#### PyCharm Configuration

1. **Interpreter**: Set to `.venv/bin/python`
2. **Code Style**: Import our `.editorconfig`
3. **Test Runner**: Configure pytest as default
4. **Linting**: Enable Ruff integration

### Debugging

#### Local Debugging

```python
# Add debugging breakpoints
import pdb; pdb.set_trace()

# Or use rich for better debugging
from rich import inspect
inspect(model, methods=True)

# Debug training with minimal data
uv run src/mlops_project/train.py \
    training.fast_dev_run=true \
    data.subsample_percentage=0.001
```

#### Remote Debugging (VS Code)

```python
# For debugging training on remote servers
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()  # Pause until debugger attaches
```

## Release Process

### Version Management

We use **semantic versioning** (semver):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

```bash
# Update version in pyproject.toml
# Create release branch
git checkout -b release/v1.2.0

# Update CHANGELOG.md
# Commit changes
git commit -m "chore: prepare release v1.2.0"

# Tag release
git tag -a v1.2.0 -m "Release v1.2.0"

# Push to GitHub
git push origin release/v1.2.0 --tags
```

### Pre-release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Performance regression tests pass
- [ ] Security scan completed

## Community Guidelines

### Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember we're all learning
- **Ask questions**: No question is too small

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: Check existing docs first
- **Code Review**: Learn from pull request feedback

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors section
- Special thanks in major releases

---

*Thank you for contributing to the MLOps Skin Lesion Classification project! Your contributions help improve medical AI applications.*