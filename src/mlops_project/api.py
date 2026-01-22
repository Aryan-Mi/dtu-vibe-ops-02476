# bump
from contextlib import asynccontextmanager
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, File, HTTPException, UploadFile
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel

from mlops_project.data import CLASS_TO_DX, get_transforms
from mlops_project.model import INPUT_SIZE

if TYPE_CHECKING:
    from torch import Tensor

# Configuration constants
IMAGE_SIZE = 224
CANCER_IDX = [1, 3, 4]
CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")
MODELS_BASE_DIR = Path(__file__).parent.parent.parent / "models"

# Global variables for model and runtime configuration
model_session: ort.InferenceSession | None = None
runtime_config = None
MODEL_PATH: Path | None = None


# Load Hydra config
def load_config():
    """Load Hydra config without changing working directory."""
    try:
        # Clean up any existing Hydra instance
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="config")
    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        raise RuntimeError(error_msg) from e


def _check_models_exist(models_dir: Path) -> bool:
    """Check if models already exist locally."""
    if models_dir.exists():
        onnx_files = list(models_dir.rglob("*.onnx"))
        if onnx_files:
            print(f"Models already exist locally ({len(onnx_files)} ONNX files found)")
            return True
    return False


def _initialize_dvc(dvc_dir: Path, work_dir: Path) -> None:
    """Initialize DVC if needed (for containerized environments without git)."""
    if dvc_dir.exists():
        return

    print("Initializing DVC (no git required)...")
    try:
        init_result = subprocess.run(
            ["uv", "run", "dvc", "init", "--no-scm"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if init_result.returncode != 0:
            subprocess.run(
                ["dvc", "init", "--no-scm"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
    except FileNotFoundError:
        pass  # DVC might not be available, will fail later


def _configure_dvc_no_scm(work_dir: Path) -> None:
    """Configure DVC to not require git."""
    try:
        subprocess.run(
            ["uv", "run", "dvc", "config", "core.no_scm", "true"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["dvc", "config", "core.no_scm", "true"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pass


def _run_dvc_pull(work_dir: Path) -> bool:
    """Run DVC pull command."""
    print("Models not found locally. Pulling from DVC...")
    try:
        dvc_cmd = ["uv", "run", "dvc", "pull", "models.dvc", "--remote", "gcs_models_remote"]
        result = subprocess.run(
            dvc_cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            result = subprocess.run(
                ["dvc", "pull", "models.dvc", "--remote", "gcs_models_remote"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode == 0:
            print("[OK] Successfully pulled models from DVC")
            return True

        print(f"[WARNING] DVC pull failed: {result.stderr}")
        return False
    except FileNotFoundError:
        print("[WARNING] DVC not found in PATH. Skipping DVC pull.")
        return False
    except (subprocess.SubprocessError, OSError) as e:
        error_msg = f"Error pulling from DVC: {e}"
        print(f"[WARNING] {error_msg}")
        return False


def pull_models_dvc() -> bool:
    """Pull models from DVC remote if models directory is empty or missing."""
    models_dir = MODELS_BASE_DIR
    models_dvc_file = models_dir.parent / "models.dvc"
    dvc_dir = models_dir.parent / ".dvc"
    work_dir = models_dir.parent

    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if models already exist
    if _check_models_exist(models_dir):
        return True

    # Check if models.dvc exists
    if not models_dvc_file.exists():
        print("[WARNING] models.dvc file not found. Cannot pull models from DVC.")
        return False

    # Initialize and configure DVC
    _initialize_dvc(dvc_dir, work_dir)
    _configure_dvc_no_scm(work_dir)

    # Pull models from DVC
    return _run_dvc_pull(work_dir)


def find_latest_model(model_name: str = "EfficientNet") -> Path | None:
    """
    Find the latest ONNX model file for a given model name.

    Args:
        model_name: Name of the model (e.g., "EfficientNet", "ResNet")

    Returns:
        Path to the latest model file, or None if not found
    """
    # Ensure models directory exists
    if not MODELS_BASE_DIR.exists():
        return None

    model_dir = MODELS_BASE_DIR / model_name

    if not model_dir.exists():
        # Try to find any model directory
        try:
            model_dirs = [d for d in MODELS_BASE_DIR.iterdir() if d.is_dir()]
            if model_dirs:
                model_dir = model_dirs[0]
                print(f"Model directory '{model_name}' not found, using '{model_dir.name}'")
            else:
                return None
        except (OSError, FileNotFoundError):
            return None

    # Find all ONNX files matching the pattern: {model_name}-epoch=XX-val_loss=X.XXXX.onnx
    onnx_files = list(model_dir.glob(f"{model_name}-epoch=*.onnx"))

    if not onnx_files:
        # Fallback: find any ONNX file in the directory
        onnx_files = list(model_dir.glob("*.onnx"))

    if not onnx_files:
        return None

    # Sort by filename (which includes epoch and val_loss)
    # This will naturally sort by epoch and val_loss
    onnx_files.sort(key=lambda x: x.name, reverse=True)

    latest_model = onnx_files[0]
    print(f"Found {len(onnx_files)} model(s), using latest: {latest_model.name}")

    return latest_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    global model_session, runtime_config, IMAGE_SIZE, MODEL_PATH

    try:
        runtime_config = load_config()
        IMAGE_SIZE = runtime_config.data.image_size

        # Auto-adjust image size for EfficientNet model variants (same as training)
        if runtime_config.model.name == "EfficientNet" and hasattr(runtime_config.model, "model_size"):
            model_size = runtime_config.model.model_size
            if model_size in INPUT_SIZE:
                IMAGE_SIZE = INPUT_SIZE[model_size]
                print(f"Auto-adjusted image_size to {IMAGE_SIZE} for EfficientNet {model_size}")
            else:
                print(f"Warning: Unknown EfficientNet model size '{model_size}', using config value {IMAGE_SIZE}")

        # Try to pull models from DVC if needed
        pull_models_dvc()

        # Find the latest model dynamically
        model_name = os.getenv("MODEL_NAME", "EfficientNet")
        MODEL_PATH = find_latest_model(model_name)

        if MODEL_PATH is None:
            error_msg = (
                "No ONNX model found for " + str(model_name) + ". "
                "Make sure models are pulled from DVC or available locally."
            )
            raise RuntimeError(error_msg)

        if not MODEL_PATH.exists():
            error_msg = "Model file not found: " + str(MODEL_PATH)
            raise FileNotFoundError(error_msg)

        print(f"Loading ONNX model from {MODEL_PATH}...")
        model_session = ort.InferenceSession(str(MODEL_PATH))
        print("Model loaded successfully!")

    except Exception as e:
        error_msg = f"Failed to initialize model: {e}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg) from e

    yield

    print("Shutting down...")
    if model_session is not None:
        del model_session
    if runtime_config is not None:
        del runtime_config


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_class: int
    diagnosis: str
    confidence: float
    is_cancer: bool
    probabilities: dict[str, float]


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """Root endpoint."""
    if model_session is None:
        return {"message": "Skin Lesion Classification API", "status": "error", "error": "Model not loaded"}
    return {"message": "Skin Lesion Classification API", "status": "ready"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True, "model_path": str(MODEL_PATH) if MODEL_PATH else None}


@app.post("/inference", response_model=PredictionResponse)
async def perform_inference(file: UploadFile = File(...)):
    """
    Perform inference on an uploaded image.

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        Dictionary with predicted class, diagnosis name, and probabilities

    Raises:
        HTTPException: If model is not loaded or inference fails
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Load and preprocess image
        i_image = Image.open(file.file)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        transform = get_transforms(image_size=IMAGE_SIZE, augment=False)
        image_tensor = cast("Tensor", transform(i_image))

        # Prepare input data
        input_name = model_session.get_inputs()[0].name
        input_data = {input_name: image_tensor.unsqueeze(0).detach().cpu().numpy()}

        # Run inference
        outputs = model_session.run(None, input_data)

        # Process outputs
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        predicted_class = int(np.argmax(probabilities))

        if predicted_class not in CLASS_TO_DX:
            error_msg = "Invalid predicted class: " + str(predicted_class)
            raise ValueError(error_msg)

        predicted_dx = CLASS_TO_DX[predicted_class]
        confidence = float(probabilities[predicted_class])
        is_cancer = predicted_class in CANCER_IDX

        return PredictionResponse(
            predicted_class=predicted_class,
            diagnosis=predicted_dx,
            confidence=confidence,
            is_cancer=is_cancer,
            probabilities={CLASS_TO_DX[i]: float(prob) for i, prob in enumerate(probabilities)},
        )

    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg) from e
