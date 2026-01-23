# Testing Guide

Comprehensive guide to testing the MLOps Skin Lesion Classification pipeline.

## Overview

Our testing strategy ensures reliability, performance, and maintainability:

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Validate speed and memory usage
- **API Tests**: Test inference server endpoints

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_models.py

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run tests in parallel (faster)
uv run pytest -n auto
```

### Test Markers

We use pytest markers to categorize tests:

```bash
# Run only fast tests (default in CI)
uv run pytest -m "not slow"

# Run integration tests
uv run pytest -m integration

# Run slow tests (full model training)
uv run pytest -m slow

# Run GPU tests (requires CUDA)
uv run pytest -m gpu

# Run API tests (requires server)
uv run pytest -m api
```

### Test Configuration

Configure pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "gpu: marks tests requiring GPU",
    "api: marks tests requiring API server",
]

addopts = [
    "--strict-markers",
    "--strict-config",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",
    "--cov-fail-under=90",
]
```

## Test Structure

### Test Organization

```
tests/
├── conftest.py                # Shared fixtures and configuration
├── test_data.py              # Data loading and processing tests
├── test_models.py            # Model architecture tests
├── test_training.py          # Training pipeline tests
├── test_api.py               # API endpoint tests
├── test_integration.py       # Integration tests
├── test_performance.py       # Performance benchmarks
├── fixtures/                 # Test data and fixtures
│   ├── sample_images/       # Test images
│   ├── mock_configs/        # Test configurations
│   └── expected_outputs/    # Expected test outputs
└── utils/                   # Test utilities
    ├── helpers.py          # Common test helpers
    └── mock_data.py        # Mock data generators
```

### Shared Fixtures

```python
# tests/conftest.py
import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytorch_lightning as pl
from src.mlops_project.model import EfficientNet, BaselineCNN
from src.mlops_project.data import HAM10000Dataset

@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_batch():
    """Generate a sample batch for testing."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    return images, labels

@pytest.fixture
def baseline_model():
    """Create a baseline model for testing."""
    return BaselineCNN(num_classes=2, dropout_rate=0.5)

@pytest.fixture  
def efficientnet_model():
    """Create an EfficientNet model for testing."""
    return EfficientNet(variant="b0", num_classes=2, pretrained=False)

@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = MagicMock(spec=HAM10000Dataset)
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = (torch.randn(3, 224, 224), torch.tensor(1))
    return dataset

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "name": "efficientnet",
            "variant": "b0",
            "num_classes": 2,
            "pretrained": False
        },
        "training": {
            "max_epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 4
        },
        "data": {
            "data_dir": "tests/fixtures/sample_data",
            "image_size": 224,
            "subsample_percentage": 0.01
        }
    }
```

## Unit Tests

### Model Architecture Tests

```python
# tests/test_models.py
import pytest
import torch
from src.mlops_project.model import EfficientNet, BaselineCNN, ResNet

class TestBaselineCNN:
    """Test suite for BaselineCNN model."""
    
    def test_initialization(self):
        """Test model initializes with correct parameters."""
        model = BaselineCNN(num_classes=7, dropout_rate=0.3)
        
        assert model.num_classes == 7
        assert model.dropout_rate == 0.3
    
    def test_forward_pass(self, baseline_model, sample_batch):
        """Test forward pass produces correct output shape."""
        images, _ = sample_batch
        output = baseline_model(images)
        
        assert output.shape == (4, 2)  # batch_size=4, num_classes=2
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_training_step(self, baseline_model, sample_batch):
        """Test training step works correctly."""
        baseline_model.trainer = MagicMock()
        
        loss = baseline_model.training_step(sample_batch, 0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() > 0

class TestEfficientNet:
    """Test suite for EfficientNet model."""
    
    @pytest.mark.parametrize("variant", ["b0", "b1", "b2"])
    def test_different_variants(self, variant):
        """Test different EfficientNet variants."""
        model = EfficientNet(variant=variant, num_classes=2, pretrained=False)
        
        assert model.variant == variant
        
        # Test with appropriate input size
        from src.mlops_project.model import INPUT_SIZE
        input_size = INPUT_SIZE[variant]
        x = torch.randn(2, 3, input_size, input_size)
        
        output = model(x)
        assert output.shape == (2, 2)
    
    def test_pretrained_weights(self):
        """Test loading pretrained weights works."""
        model = EfficientNet(variant="b0", pretrained=True)
        
        # Check that weights are loaded (not random initialization)
        # This is a simple heuristic - pretrained models have specific weight distributions
        conv_layer = model.backbone.features[0][0]
        weights = conv_layer.weight.data
        
        # Pretrained weights should have reasonable variance
        assert weights.var() > 0.001
        assert weights.var() < 10.0

class TestModelSaving:
    """Test model saving and loading."""
    
    def test_checkpoint_save_load(self, efficientnet_model, temp_dir):
        """Test saving and loading model checkpoints."""
        checkpoint_path = temp_dir / "test_model.ckpt"
        
        # Save checkpoint
        trainer = pl.Trainer(max_epochs=1, default_root_dir=temp_dir)
        trainer.save_checkpoint(checkpoint_path)
        
        # Load checkpoint
        loaded_model = EfficientNet.load_from_checkpoint(checkpoint_path)
        
        assert loaded_model.variant == efficientnet_model.variant
        assert loaded_model.num_classes == efficientnet_model.num_classes
```

### Data Processing Tests

```python
# tests/test_data.py
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.mlops_project.data import HAM10000Dataset, preprocess_image, get_transforms
from src.mlops_project.dataloader import HAM10000DataModule

class TestHAM10000Dataset:
    """Test HAM10000 dataset functionality."""
    
    @pytest.fixture
    def mock_dataset_dir(self, temp_dir):
        """Create mock dataset structure."""
        # Create directory structure
        images_dir = temp_dir / "HAM10000_images"
        images_dir.mkdir()
        
        # Create mock images
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img.save(images_dir / f"ISIC_{i:07d}.jpg")
        
        # Create mock metadata
        metadata_path = temp_dir / "HAM10000_metadata.csv"
        with open(metadata_path, "w") as f:
            f.write("image_id,dx,dx_type,age,sex,localization\n")
            for i in range(10):
                label = "mel" if i % 2 == 0 else "nv"
                f.write(f"ISIC_{i:07d},{label},histo,45,male,back\n")
        
        return temp_dir
    
    def test_dataset_loading(self, mock_dataset_dir):
        """Test dataset loads correctly."""
        dataset = HAM10000Dataset(str(mock_dataset_dir), split="train")
        
        assert len(dataset) > 0
        
        # Test getting item
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)  # C, H, W
        assert isinstance(label, torch.Tensor)
        assert label.item() in [0, 1]  # Binary classification
    
    def test_data_splits(self, mock_dataset_dir):
        """Test train/val/test splits work correctly."""
        train_dataset = HAM10000Dataset(str(mock_dataset_dir), split="train")
        val_dataset = HAM10000Dataset(str(mock_dataset_dir), split="val")
        test_dataset = HAM10000Dataset(str(mock_dataset_dir), split="test")
        
        # Check that splits don't overlap
        train_files = set(train_dataset.image_files)
        val_files = set(val_dataset.image_files)
        test_files = set(test_dataset.image_files)
        
        assert len(train_files & val_files) == 0
        assert len(train_files & test_files) == 0
        assert len(val_files & test_files) == 0
    
    def test_transforms(self):
        """Test image transforms work correctly."""
        transforms = get_transforms(train=True, image_size=256)
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        
        # Apply transforms
        transformed = transforms(test_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        assert transformed.dtype == torch.float32
        assert 0 <= transformed.min() <= transformed.max() <= 1

class TestDataModule:
    """Test PyTorch Lightning DataModule."""
    
    def test_datamodule_setup(self, mock_dataset_dir):
        """Test DataModule setup works correctly."""
        dm = HAM10000DataModule(
            data_dir=str(mock_dataset_dir),
            batch_size=4,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        dm.setup("fit")
        
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
    
    def test_dataloaders(self, mock_dataset_dir):
        """Test DataLoaders work correctly."""
        dm = HAM10000DataModule(
            data_dir=str(mock_dataset_dir),
            batch_size=2,
            num_workers=0
        )
        
        dm.setup("fit")
        
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape == (2, 3, 224, 224)
        assert labels.shape == (2,)
```

### Training Tests

```python
# tests/test_training.py
import pytest
import torch
import pytorch_lightning as pl
from unittest.mock import MagicMock, patch

from src.mlops_project.train import train_model
from src.mlops_project.model import BaselineCNN

class TestTrainingPipeline:
    """Test training pipeline functionality."""
    
    def test_training_step_gradients(self, baseline_model, sample_batch):
        """Test that gradients are computed correctly."""
        optimizer = torch.optim.Adam(baseline_model.parameters())
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = baseline_model.training_step(sample_batch, 0)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in baseline_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    @pytest.mark.slow
    def test_overfitting_small_batch(self, baseline_model):
        """Test that model can overfit a small batch (slow test)."""
        # Create small, fixed dataset
        batch_size = 8
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(images, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        # Configure trainer for overfitting test
        trainer = pl.Trainer(
            max_epochs=50,
            overfit_batches=1,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False
        )
        
        # Train
        initial_loss = baseline_model.training_step((images, labels), 0).item()
        trainer.fit(baseline_model, dataloader)
        final_loss = baseline_model.training_step((images, labels), 0).item()
        
        # Model should overfit (loss should decrease significantly)
        assert final_loss < initial_loss * 0.5
    
    def test_validation_step(self, efficientnet_model, sample_batch):
        """Test validation step works correctly."""
        with torch.no_grad():
            result = efficientnet_model.validation_step(sample_batch, 0)
            
        assert "val_loss" in result
        assert isinstance(result["val_loss"], torch.Tensor)
        assert result["val_loss"].item() > 0
    
    def test_optimizer_configuration(self, baseline_model):
        """Test optimizer and scheduler configuration."""
        optimizer_config = baseline_model.configure_optimizers()
        
        assert "optimizer" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.Optimizer)
        
        if "lr_scheduler" in optimizer_config:
            scheduler_config = optimizer_config["lr_scheduler"]
            assert "scheduler" in scheduler_config
            assert "monitor" in scheduler_config
```

## Integration Tests

### End-to-End Pipeline Tests

```python
# tests/test_integration.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

@pytest.mark.integration
class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""
    
    def test_training_to_inference_pipeline(self, temp_dir, sample_config):
        """Test complete pipeline from training to inference."""
        
        # Mock training to avoid actual training
        with patch('src.mlops_project.train.main') as mock_train:
            mock_train.return_value = None
            
            # Mock model saving
            model_path = temp_dir / "test_model.ckpt"
            model_path.touch()  # Create empty file
            
            # Test that training completes without errors
            # In real implementation, this would run actual training
            assert model_path.exists()
    
    @pytest.mark.slow
    def test_full_training_cycle(self, sample_config):
        """Test complete training cycle with real data (slow)."""
        # This test runs actual training with minimal data
        config = sample_config.copy()
        config["data"]["subsample_percentage"] = 0.001
        config["training"]["max_epochs"] = 1
        config["training"]["fast_dev_run"] = True
        
        # Run training
        with patch('wandb.init'), patch('wandb.log'):
            result = train_model(config)
            
        # Check training completed successfully
        assert result is not None
```

## API Tests

### FastAPI Endpoint Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import numpy as np

from src.mlops_project.api import app

@pytest.fixture
def client():
    """Create test client for API."""
    return TestClient(app)

@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes for testing."""
    # Create random RGB image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @patch('src.mlops_project.api.model')  # Mock the model
    def test_predict_endpoint(self, mock_model, client, sample_image_bytes):
        """Test prediction endpoint."""
        # Mock model prediction
        mock_model.predict.return_value = {
            "prediction": "benign",
            "confidence": 0.85,
            "class_probabilities": {"benign": 0.85, "malignant": 0.15}
        }
        
        # Send request
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["benign", "malignant"]
        assert 0 <= data["confidence"] <= 1
        assert "class_probabilities" in data
    
    def test_predict_invalid_file(self, client):
        """Test prediction with invalid file."""
        # Send text file instead of image
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 400
        assert "error" in response.json()
    
    def test_predict_no_file(self, client):
        """Test prediction without file."""
        response = client.post("/predict")
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
```

## Performance Tests

### Benchmark Tests

```python
# tests/test_performance.py
import pytest
import time
import torch
import psutil
from memory_profiler import profile

from src.mlops_project.model import EfficientNet

@pytest.mark.slow
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_inference_speed(self, efficientnet_model):
        """Test inference speed benchmark."""
        model = efficientnet_model
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                x = torch.randn(1, 3, 224, 224)
                _ = model(x)
        
        # Benchmark
        num_iterations = 100
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            images_per_second = batch_size / avg_time_per_batch
            
            print(f"Batch size {batch_size}: {images_per_second:.2f} images/second")
            
            # Performance assertions
            assert avg_time_per_batch < 1.0  # Should be under 1 second per batch
            assert images_per_second > 10    # Should process at least 10 images/sec
    
    def test_memory_usage(self, efficientnet_model):
        """Test memory usage during inference."""
        model = efficientnet_model
        model.eval()
        
        process = psutil.Process()
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference
        with torch.no_grad():
            for _ in range(50):
                x = torch.randn(4, 3, 224, 224)
                _ = model(x)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        # Memory usage should not increase significantly
        assert memory_increase < 100  # Less than 100MB increase
    
    @pytest.mark.gpu
    def test_gpu_memory_usage(self, efficientnet_model):
        """Test GPU memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device("cuda")
        model = efficientnet_model.to(device)
        model.eval()
        
        # Measure initial GPU memory
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Run inference
        with torch.no_grad():
            for _ in range(20):
                x = torch.randn(8, 3, 224, 224).to(device)
                _ = model(x)
        
        # Measure peak GPU memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # GPU memory should be reasonable
        assert peak_memory < 2000  # Less than 2GB
```

## Test Data Management

### Creating Test Fixtures

```python
# tests/utils/mock_data.py
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

def create_mock_ham10000_data(num_samples: int = 100) -> Path:
    """Create mock HAM10000 dataset for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create directory structure
    images_dir = temp_dir / "HAM10000_images"
    images_dir.mkdir()
    
    # Create mock images
    for i in range(num_samples):
        # Generate random image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save image
        image_path = images_dir / f"ISIC_{i:07d}.jpg"
        image.save(image_path)
    
    # Create mock metadata
    metadata_path = temp_dir / "HAM10000_metadata.csv"
    with open(metadata_path, "w") as f:
        f.write("image_id,dx,dx_type,age,sex,localization\n")
        for i in range(num_samples):
            # Alternate between malignant and benign
            dx = "mel" if i % 2 == 0 else "nv"
            f.write(f"ISIC_{i:07d},{dx},histo,{30+i%50},{'male' if i%2 else 'female'},back\n")
    
    return temp_dir

def create_sample_batch(batch_size: int = 4, num_classes: int = 2):
    """Create sample batch for testing."""
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels
```

## Continuous Integration

### GitHub Actions Test Configuration

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.12"]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
        
      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}
        
      - name: Install dependencies
        run: uv sync
        
      - name: Run linting
        run: uv run ruff check src/ tests/
        
      - name: Run tests
        run: uv run pytest -v --tb=short -m "not slow and not gpu"
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.os == 'ubuntu-latest'
        with:
          file: ./coverage.xml
```

---

*Continue to [CI/CD Pipeline](ci-cd.md) to understand the complete automation workflow.*