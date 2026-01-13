"""Unit tests for the model module."""

import pytest
import torch

from mlops_project.model import BaselineCNN, EfficientNet, ResNet


class TestBaselineCNN:
    """Tests for BaselineCNN model."""

    @pytest.mark.parametrize(("num_classes", "input_dim"), [(2, 224), (7, 256)])
    def test_model_input_output_shapes(self, num_classes, input_dim):
        """Test that given input shape X produces output shape Y.

        This is the required model test from the lecture.
        """
        model = BaselineCNN(num_classes=num_classes, input_dim=input_dim)

        # Input shape: [batch_size, channels, height, width]
        batch_size = 4
        x = torch.randn(batch_size, 3, input_dim, input_dim)

        # Forward pass
        output = model(x)

        # Check output shape: [batch_size, num_classes]
        assert output.shape == (
            batch_size,
            num_classes,
        ), f"Expected output shape ({batch_size}, {num_classes}), got {output.shape}"

    def test_input_validation_wrong_dimensions(self):
        """Test that model raises ValueError for wrong input dimensions."""
        model = BaselineCNN(num_classes=7, input_dim=224)

        # Test 3D input (should be 4D)
        x_3d = torch.randn(3, 224, 224)
        with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
            _ = model(x_3d)


class TestResNet:
    """Tests for ResNet model."""

    def test_model_input_output_shapes(self):
        """Test that given input shape produces correct output shape."""
        model = ResNet(num_classes=7)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, 7), f"Expected output shape (4, 7), got {output.shape}"


class TestEfficientNet:
    """Tests for EfficientNet model."""

    def test_model_input_output_shapes(self):
        """Test that given input shape produces correct output shape."""
        model = EfficientNet(model_size="b0", num_classes=7, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, 7), f"Expected output shape (4, 7), got {output.shape}"

    def test_input_size_warning(self):
        """Test that EfficientNet warns for incorrect input size."""
        model = EfficientNet(model_size="b0", num_classes=7, pretrained=False)

        # b0 expects 224x224, test with wrong size
        x_wrong_size = torch.randn(4, 3, 256, 256)
        with pytest.warns(UserWarning, match="Input size.*does not match expected size"):
            _ = model(x_wrong_size)
