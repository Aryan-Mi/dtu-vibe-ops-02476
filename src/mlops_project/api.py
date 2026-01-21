from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from hydra import compose, initialize_config_dir
from PIL import Image
from pydantic import BaseModel

from mlops_project.data import CLASS_TO_DX, get_transforms

if TYPE_CHECKING:
    from torch import Tensor

MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "EfficientNet" / "EfficientNet-epoch=04-val_loss=0.4910.onnx"
)  # noqa: E501
IMAGE_SIZE = 224

CANCER_IDX = [1, 3, 4]

model_session = None
runtime_config = None

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")


# load_config function
def load_config():
    """Load Hydra config without changing working directory."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    global model_session, runtime_config, IMAGE_SIZE

    runtime_config = load_config()
    IMAGE_SIZE = runtime_config.data.image_size

    print(f"Loading ONNX model from {MODEL_PATH}...")
    model_session = ort.InferenceSession(str(MODEL_PATH))
    print("Model loaded successfully!")

    yield

    print("Shutting down...")
    del model_session, runtime_config


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
        return {"error": "Model not loaded"}
    return {"message": "Skin Lesion Classification API", "status": "ready"}


@app.post("/inference", response_model=PredictionResponse)
async def perform_inference(file: UploadFile = File(...)):
    """
    Perform inference on an uploaded image.

    Args:
        file: Image file (JPG, PNG, etc.)

    Returns:
        Dictionary with predicted class, diagnosis name, and probabilities
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    i_image = Image.open(file.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    transform = get_transforms(image_size=IMAGE_SIZE, augment=False)
    image_tensor = cast("Tensor", transform(i_image))

    input_data = {model_session.get_inputs()[0].name: image_tensor.unsqueeze(0).detach().cpu().numpy()}

    outputs = model_session.run(None, input_data)

    logits = outputs[0][0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    predicted_class = int(np.argmax(probabilities))
    predicted_dx = CLASS_TO_DX[predicted_class]
    confidence = float(probabilities[predicted_class])
    is_cancer = predicted_class in CANCER_IDX

    return {
        "predicted_class": predicted_class,
        "diagnosis": predicted_dx,
        "confidence": confidence,
        "is_cancer": is_cancer,
        "probabilities": {CLASS_TO_DX[i]: float(prob) for i, prob in enumerate(probabilities)},
    }
