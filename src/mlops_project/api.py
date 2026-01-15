from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from hydra import compose, initialize_config_dir
from PIL import Image
from torchvision import transforms
from data import get_transforms, CLASS_TO_DX
# 1. Initialization
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "EfficientNet" / "EfficientNet-epoch=04-val_loss=0.4910.onnx" # Hardcoded for now
IMAGE_SIZE = 224  # Default image size, will be updated from config

# Binary mapping: 0 = non-cancer, 1 = cancer
CANCER_IDX = [1, 3, 4]  # mel, bcc, akiec

model_session = None
runtime_config = None

# 2. Hydra config loading helper (called inside lifespan)
CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")
def load_config():
    """Load Hydra config without changing working directory."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config")

# 3. Lifespan for FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown."""
    # 1. Startup: Load model and config
    global model_session, runtime_config, IMAGE_SIZE

    runtime_config = load_config()
    IMAGE_SIZE = runtime_config.data.image_size

    print(f"Loading ONNX model from {MODEL_PATH}...")
    model_session = ort.InferenceSession(str(MODEL_PATH))
    print("Model loaded successfully!")

    yield

    # 2. Shutdown: Cleanup
    print("Shutting down...")
    del model_session, runtime_config

app = FastAPI(lifespan=lifespan)

# 4. APIs
@app.get("/")
def read_root():
    """Root endpoint."""
    if model_session is None:
        return {"error": "Model not loaded"}
    else:
        return {"message": "Skin Lesion Classification API", "status": "ready"}

@app.post("/inference")
async def perform_inference(file: UploadFile = File(...)):
    """
    Perform inference on an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Dictionary with predicted class, diagnosis name, and probabilities
    """
    if model_session is None:
        return {"error": "Model not loaded"}
    
    # 1. Read & open image
    i_image = Image.open(file.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    
    # 2. Apply preprocessing transforms
    transform = get_transforms(image_size=IMAGE_SIZE, augment=False)
    image_tensor = transform(i_image)
    
    # 3. Add batch dimension and convert to numpy
    input_data = {model_session.get_inputs()[0].name: image_tensor.unsqueeze(0).numpy()}
    
    # 4. Run inference
    outputs = model_session.run(None, input_data) # Basically calling forward - outputs logits
    
    # 5. Get predictions
    logits = outputs[0][0]  # 5.1 Remove batch dimension
    probabilities = np.exp(logits) / np.sum(np.exp(logits))  # 5.2 Softmax
    predicted_class = int(np.argmax(probabilities))
    predicted_dx = CLASS_TO_DX[predicted_class]
    confidence = float(probabilities[predicted_class])
    
    return {
        "predicted_class": predicted_class,
        "diagnosis": predicted_dx,
        "confidence": confidence,
        "probabilities": {CLASS_TO_DX[i]: float(prob) for i, prob in enumerate(probabilities)}
    }