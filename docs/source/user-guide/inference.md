# Running Inference

Learn how to use your trained models for prediction, both locally and through the API server.

## Local Inference

### Single Image Prediction

```python
import torch
from src.mlops_project.model import EfficientNet
from src.mlops_project.data import preprocess_image
from PIL import Image

# Load trained model
model = EfficientNet.load_from_checkpoint("models/efficientnet/best.ckpt")
model.eval()

# Preprocess image
image = Image.open("path/to/skin_lesion.jpg")
input_tensor = preprocess_image(image)

# Make prediction
with torch.no_grad():
    logits = model(input_tensor.unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1)
    
prediction = "malignant" if probabilities[0][1] > 0.5 else "benign"
confidence = float(probabilities.max())

print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
```

### Batch Inference

```python
from src.mlops_project.data import HAM10000DataModule
import torch

# Load test data
data_module = HAM10000DataModule("data/ham10000", batch_size=32)
data_module.setup("test")
test_loader = data_module.test_dataloader()

# Load model
model = EfficientNet.load_from_checkpoint("models/efficientnet/best.ckpt")
model.eval()

# Run batch inference
predictions = []
confidences = []

for batch in test_loader:
    images, labels = batch
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        
        batch_predictions = (probs[:, 1] > 0.5).int()
        batch_confidences = probs.max(dim=1)[0]
        
        predictions.extend(batch_predictions.tolist())
        confidences.extend(batch_confidences.tolist())

print(f"Processed {len(predictions)} images")
```

## API Server Inference

### Starting the API Server

```bash
# Start FastAPI server with default model
uv run src/mlops_project/api.py

# Start with specific model
uv run src/mlops_project/api.py --model-path models/efficientnet/best.ckpt

# Start with custom configuration
uv run src/mlops_project/api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --model-path models/efficientnet/best.ckpt
```

### API Usage Examples

#### cURL Commands

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/image.jpg" \
     | jq

# Health check
curl "http://localhost:8000/health"

# Model information
curl "http://localhost:8000/model/info" | jq
```

#### Python Client

```python
import requests
from pathlib import Path

# Upload and predict
def predict_image(image_path: str, api_url: str = "http://localhost:8000"):
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{api_url}/predict", files=files)
    
    return response.json()

# Example usage
result = predict_image("skin_lesion.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

#### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predictImage(imagePath) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));
    
    try {
        const response = await axios.post(
            'http://localhost:8000/predict',
            formData,
            { headers: formData.getHeaders() }
        );
        return response.data;
    } catch (error) {
        console.error('Prediction failed:', error.message);
    }
}

// Usage
predictImage('skin_lesion.jpg').then(result => {
    console.log(`Prediction: ${result.prediction}`);
    console.log(`Confidence: ${result.confidence.toFixed(3)}`);
});
```

### API Response Format

```json
{
  "prediction": "malignant",
  "confidence": 0.847,
  "class_probabilities": {
    "benign": 0.153,
    "malignant": 0.847
  },
  "model_info": {
    "name": "efficientnet",
    "variant": "b3",
    "version": "1.0.0",
    "input_size": [300, 300]
  },
  "processing_time_ms": 45.2
}
```

## ONNX Inference (Production)

### Export Model to ONNX

```python
import torch
from src.mlops_project.model import EfficientNet

# Load PyTorch model
model = EfficientNet.load_from_checkpoint("models/efficientnet/best.ckpt")
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 300, 300)
torch.onnx.export(
    model,
    dummy_input,
    "models/efficientnet_b3.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
ort_session = ort.InferenceSession("models/efficientnet_b3.onnx")

# Preprocess image
image = Image.open("skin_lesion.jpg").resize((300, 300))
input_array = np.array(image).astype(np.float32) / 255.0
input_array = np.transpose(input_array, (2, 0, 1))
input_array = np.expand_dims(input_array, axis=0)

# Run inference
outputs = ort_session.run(None, {'input': input_array})
probabilities = np.softmax(outputs[0][0])

prediction = "malignant" if probabilities[1] > 0.5 else "benign"
confidence = float(np.max(probabilities))

print(f"ONNX Prediction: {prediction} (confidence: {confidence:.3f})")
```

---

*Next: Learn about [Deployment](deployment.md) strategies for production environments.*