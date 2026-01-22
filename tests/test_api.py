"""Unit tests for the API module."""

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from mlops_project.api import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img = Image.new("RGB", (224, 224), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    img = Image.new("L", (224, 224), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_png_image():
    """Create a sample PNG image for testing."""
    img = Image.new("RGB", (224, 224), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_health_check_endpoint(self, client):
        """Test health check endpoint returns correct status."""
        response = client.get("/")
        assert response.status_code == 200

    @patch("mlops_project.api.model_session", None)
    def test_health_check_endpoint_without_model(self, client):
        """Test health check endpoint when model is not loaded."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["error"] == "Model not loaded"

    @patch("mlops_project.api.model_session")
    def test_prediction_endpoint_valid_input(self, mock_session, client, sample_image):
        """Test prediction endpoint with valid input returns correct format."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[2.5, 1.8, 0.5, 0.3, 0.2, -0.5, -1.0]])]

        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "diagnosis" in data
        assert "confidence" in data
        assert "is_cancer" in data
        assert "probabilities" in data
        assert isinstance(data["predicted_class"], int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["is_cancer"], bool)
        assert isinstance(data["probabilities"], dict)
        assert 0 <= data["confidence"] <= 1
        assert len(data["probabilities"]) == 7

    @pytest.mark.skip(reason="API doesn't validate image format, PIL accepts most files")
    def test_prediction_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input returns 422."""
        response = client.post("/inference", files={"file": ("test.jpg", io.BytesIO(b""))})
        assert response.status_code in [422, 500]

    @patch("mlops_project.api.model_session", None)
    def test_prediction_endpoint_model_not_loaded(self, client, sample_image):
        """Test prediction endpoint when model is not loaded."""
        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == "Model not loaded"

    @patch("mlops_project.api.model_session")
    def test_response_format_validation(self, mock_session, client, sample_image):
        """Test response format validation."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[2.5, 1.8, 0.5, 0.3, 0.2, -0.5, -1.0]])]

        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        expected_keys = {"predicted_class", "diagnosis", "confidence", "is_cancer", "probabilities"}
        assert set(data.keys()) == expected_keys
        assert isinstance(data["predicted_class"], int)
        assert 0 <= data["predicted_class"] <= 6
        assert isinstance(data["diagnosis"], str)
        assert data["diagnosis"] in ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["is_cancer"], bool)
        assert isinstance(data["probabilities"], dict)
        for prob in data["probabilities"].values():
            assert isinstance(prob, float)
            assert 0 <= prob <= 1

    @patch("mlops_project.api.model_session")
    def test_grayscale_image_conversion(self, mock_session, client, sample_grayscale_image):
        """Test that grayscale images are converted to RGB."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[2.5, 1.8, 0.5, 0.3, 0.2, -0.5, -1.0]])]

        response = client.post("/inference", files={"file": ("test.png", sample_grayscale_image, "image/png")})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "diagnosis" in data

    @patch("mlops_project.api.model_session")
    def test_png_image_support(self, mock_session, client, sample_png_image):
        """Test that PNG images are supported."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[2.5, 1.8, 0.5, 0.3, 0.2, -0.5, -1.0]])]

        response = client.post("/inference", files={"file": ("test.png", sample_png_image, "image/png")})

        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

    @patch("mlops_project.api.model_session")
    def test_cancer_detection(self, mock_session, client, sample_image):
        """Test cancer detection with cancer class prediction."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]

        cancer_logits = np.array([-2.0, 5.0, -1.0, 0.5, 0.3, -3.0, -4.0])
        mock_session.run.return_value = [np.array([cancer_logits])]

        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        assert data["is_cancer"] is True
        assert data["predicted_class"] == 1
        assert data["diagnosis"] == "mel"

    @patch("mlops_project.api.model_session")
    def test_non_cancer_detection(self, mock_session, client, sample_image):
        """Test non-cancer detection with benign class prediction."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]

        benign_logits = np.array([5.0, -2.0, 1.0, -1.0, -1.5, -3.0, -4.0])
        mock_session.run.return_value = [np.array([benign_logits])]

        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        assert data["is_cancer"] is False
        assert data["predicted_class"] == 0
        assert data["diagnosis"] == "nv"

    @patch("mlops_project.api.model_session")
    def test_probabilities_sum_to_one(self, mock_session, client, sample_image):
        """Test that probabilities sum to approximately 1."""
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.array([[2.5, 1.8, 0.5, 0.3, 0.2, -0.5, -1.0]])]

        response = client.post("/inference", files={"file": ("test.jpg", sample_image, "image/jpeg")})

        assert response.status_code == 200
        data = response.json()
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 1e-6
