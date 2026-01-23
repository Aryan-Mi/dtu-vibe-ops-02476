# API Usage Examples

Comprehensive examples for integrating with the skin lesion classification API server.

## Getting Started

### Start the API Server

```bash
# Start with default model
uv run src/mlops_project/api.py

# Start with specific model and configuration
uv run src/mlops_project/api.py \
    --model-path models/efficientnet/best.ckpt \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4
```

Server will be available at `http://localhost:8000` with automatic API documentation at `http://localhost:8000/docs`.

## Basic Usage Examples

### 1. Simple Python Client

```python
import requests
from pathlib import Path
import json

def predict_skin_lesion(image_path: str, api_url: str = "http://localhost:8000"):
    """Classify a single skin lesion image"""
    
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(f"{api_url}/predict", files=files)
        
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Example usage
result = predict_skin_lesion("path/to/skin_lesion.jpg")
if result:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Malignant probability: {result['class_probabilities']['malignant']:.3f}")
```

### 2. Async Python Client

```python
import aiohttp
import asyncio
from pathlib import Path

class AsyncSkinLesionAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def predict(self, image_path: str) -> dict:
        """Async prediction for better performance"""
        
        async with aiohttp.ClientSession() as session:
            with open(image_path, 'rb') as image_file:
                data = aiohttp.FormData()
                data.add_field('file', image_file, filename=Path(image_path).name)
                
                async with session.post(f"{self.base_url}/predict", data=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error = await response.json()
                        raise Exception(f"API Error: {error}")
    
    async def health_check(self) -> dict:
        """Check API health status"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()

# Example async usage
async def main():
    api = AsyncSkinLesionAPI()
    
    # Check if API is healthy
    health = await api.health_check()
    print(f"API Status: {health['status']}")
    
    # Make predictions
    tasks = [
        api.predict("image1.jpg"),
        api.predict("image2.jpg"),
        api.predict("image3.jpg")
    ]
    
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"Image {i}: {result['prediction']} (confidence: {result['confidence']:.3f})")

# Run async example
# asyncio.run(main())
```

### 3. Batch Processing Script

```python
import requests
from pathlib import Path
import pandas as pd
from typing import List, Dict
import time

def process_image_batch(image_dir: str, output_file: str = "predictions.csv"):
    """Process multiple images and save results to CSV"""
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    results = []
    
    print(f"Processing {len(image_files)} images...")
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {image_path.name} ({i+1}/{len(image_files)})")
            
            # Make prediction
            with open(image_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://localhost:8000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store result
                results.append({
                    "filename": image_path.name,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "malignant_probability": result["class_probabilities"]["malignant"],
                    "benign_probability": result["class_probabilities"]["benign"],
                    "processing_time_ms": result["metadata"]["total_time_ms"]
                })
            else:
                print(f"Error processing {image_path.name}: {response.status_code}")
                results.append({
                    "filename": image_path.name,
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "malignant_probability": 0.0,
                    "benign_probability": 0.0,
                    "processing_time_ms": 0.0
                })
        
        except Exception as e:
            print(f"Exception processing {image_path.name}: {e}")
            
        # Add small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to {output_file}")
    print(f"Summary:")
    print(f"  Total images: {len(results)}")
    print(f"  Malignant predictions: {len(df[df['prediction'] == 'malignant'])}")
    print(f"  Benign predictions: {len(df[df['prediction'] == 'benign'])}")
    print(f"  Average confidence: {df['confidence'].mean():.3f}")
    print(f"  Average processing time: {df['processing_time_ms'].mean():.1f}ms")

# Usage
# process_image_batch("path/to/images/", "results.csv")
```

## Advanced Usage Patterns

### 4. Production Client Class

```python
import requests
import time
from typing import Optional, Dict, Any
from pathlib import Path
import logging

class SkinLesionAPIClient:
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Set up headers
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method, 
                    url, 
                    headers=self.headers,
                    timeout=self.timeout,
                    **kwargs
                )
                
                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s")
                time.sleep(wait_time)
    
    def predict(
        self, 
        image_path: str, 
        confidence_threshold: float = 0.5,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """Predict skin lesion classification"""
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            data = {
                "confidence_threshold": confidence_threshold,
                "return_probabilities": return_probabilities
            }
            
            response = self._make_request("POST", "/predict", files=files, data=data)
            return response.json()
    
    def predict_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        """Predict multiple images in a single request"""
        
        files = []
        for image_path in image_paths:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            files.append(("files", open(image_path, "rb")))
        
        try:
            response = self._make_request("POST", "/predict/batch", files=files)
            return response.json()
        finally:
            # Close all file handles
            for _, file_handle in files:
                file_handle.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self._make_request("GET", "/health")
        return response.json()
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """Get detailed health information"""
        response = self._make_request("GET", "/health/detailed")
        return response.json()
    
    def model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self._make_request("GET", "/model/info")
        return response.json()

# Example usage with production client
client = SkinLesionAPIClient(
    base_url="https://your-api-domain.com",
    api_key="your-api-key",
    timeout=30,
    max_retries=3
)

# Check API status
health = client.health_check()
print(f"API Status: {health['status']}")

# Get model information
model_info = client.model_info()
print(f"Model: {model_info['name']} {model_info['variant']}")

# Make prediction
result = client.predict("skin_lesion.jpg", confidence_threshold=0.7)
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
```

### 5. Web Application Integration

#### Flask Example

```python
from flask import Flask, request, jsonify, render_template
import requests
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

SKIN_LESION_API_URL = "http://localhost:8000"

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Send to skin lesion API
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{SKIN_LESION_API_URL}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Format response for web app
                return jsonify({
                    'success': True,
                    'prediction': result['prediction'],
                    'confidence': round(result['confidence'], 3),
                    'probabilities': {
                        'malignant': round(result['class_probabilities']['malignant'], 3),
                        'benign': round(result['class_probabilities']['benign'], 3)
                    },
                    'processing_time': result['metadata']['total_time_ms']
                })
            else:
                return jsonify({'error': 'API request failed'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
```

#### React/JavaScript Example

```javascript
import React, { useState } from 'react';
import axios from 'axios';

const SkinLesionClassifier = () => {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setPrediction(null);
        setError(null);
    };

    const analyzeLesion = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('confidence_threshold', '0.7');

        try {
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 30000,
            });

            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    const renderResult = () => {
        if (!prediction) return null;

        const isMalignant = prediction.prediction === 'malignant';
        const confidence = (prediction.confidence * 100).toFixed(1);

        return (
            <div className={`result ${isMalignant ? 'malignant' : 'benign'}`}>
                <h3>Analysis Result</h3>
                <div className="prediction">
                    <strong>Prediction:</strong> {prediction.prediction.toUpperCase()}
                </div>
                <div className="confidence">
                    <strong>Confidence:</strong> {confidence}%
                </div>
                <div className="probabilities">
                    <div>Malignant: {(prediction.class_probabilities.malignant * 100).toFixed(1)}%</div>
                    <div>Benign: {(prediction.class_probabilities.benign * 100).toFixed(1)}%</div>
                </div>
                <div className="model-info">
                    <small>
                        Model: {prediction.model_info.name} {prediction.model_info.variant}
                        | Processing time: {prediction.metadata.total_time_ms}ms
                    </small>
                </div>
                {isMalignant && (
                    <div className="warning">
                        ⚠️ This result suggests the lesion may be malignant. 
                        Please consult a dermatologist for professional evaluation.
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="skin-lesion-classifier">
            <h2>Skin Lesion Classification</h2>
            
            <div className="upload-section">
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    disabled={loading}
                />
                
                <button 
                    onClick={analyzeLesion} 
                    disabled={!file || loading}
                    className="analyze-button"
                >
                    {loading ? 'Analyzing...' : 'Analyze Lesion'}
                </button>
            </div>

            {error && (
                <div className="error">
                    Error: {error}
                </div>
            )}

            {renderResult()}

            <div className="disclaimer">
                <small>
                    ⚠️ This tool is for educational purposes only and should not be used 
                    for medical diagnosis. Always consult healthcare professionals for 
                    medical advice.
                </small>
            </div>
        </div>
    );
};

export default SkinLesionClassifier;
```

## Error Handling Examples

### 6. Robust Error Handling

```python
import requests
from typing import Optional, Dict, Union
import json

class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: int, error_code: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)

def predict_with_error_handling(
    image_path: str, 
    api_url: str = "http://localhost:8000",
    confidence_threshold: float = 0.5
) -> Union[Dict, None]:
    """
    Make prediction with comprehensive error handling
    """
    
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            data = {"confidence_threshold": confidence_threshold}
            
            response = requests.post(
                f"{api_url}/predict", 
                files=files, 
                data=data,
                timeout=30
            )
        
        # Handle different HTTP status codes
        if response.status_code == 200:
            return response.json()
        
        elif response.status_code == 400:
            error_data = response.json()
            if error_data.get('code') == 'INVALID_FORMAT':
                print("❌ Invalid image format. Please use JPEG or PNG.")
            elif error_data.get('code') == 'INVALID_IMAGE':
                print("❌ Image appears to be corrupted or invalid.")
            else:
                print(f"❌ Bad request: {error_data.get('detail', 'Unknown error')}")
                
        elif response.status_code == 413:
            print("❌ Image file too large. Maximum size is 10MB.")
            
        elif response.status_code == 429:
            print("❌ Rate limit exceeded. Please wait before making another request.")
            
        elif response.status_code == 500:
            error_data = response.json()
            print(f"❌ Server error: {error_data.get('detail', 'Internal server error')}")
            
        elif response.status_code == 503:
            print("❌ Service unavailable. The API server may be down.")
            
        else:
            print(f"❌ Unexpected error: HTTP {response.status_code}")
        
        return None
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Check if API server is running.")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server may be overloaded.")
        
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        
    except json.JSONDecodeError:
        print("❌ Invalid response format from API.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return None

# Usage with error handling
result = predict_with_error_handling("skin_lesion.jpg", confidence_threshold=0.8)

if result:
    print(f"✅ Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']:.3f}")
else:
    print("❌ Prediction failed. Please check the error messages above.")
```

## Performance Optimization

### 7. Connection Pooling and Sessions

```python
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict

class OptimizedAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000", pool_size: int = 10):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def predict_single(self, image_path: str) -> Dict:
        """Single prediction using session"""
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = self.session.post(f"{self.base_url}/predict", files=files)
            response.raise_for_status()
            return response.json()
    
    def predict_concurrent(self, image_paths: List[str], max_workers: int = 5) -> List[Dict]:
        """Concurrent predictions for better throughput"""
        
        def predict_image(image_path: str):
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                result['success'] = True
                return result
            except Exception as e:
                return {
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                }
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(predict_image, path): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
        
        return results
    
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()

# Example: Process 100 images concurrently
client = OptimizedAPIClient()

image_paths = [f"image_{i:03d}.jpg" for i in range(100)]

start_time = time.time()
results = client.predict_concurrent(image_paths, max_workers=8)
end_time = time.time()

successful_predictions = [r for r in results if r['success']]
failed_predictions = [r for r in results if not r['success']]

print(f"Processed {len(image_paths)} images in {end_time - start_time:.2f} seconds")
print(f"Successful: {len(successful_predictions)}")
print(f"Failed: {len(failed_predictions)}")
print(f"Throughput: {len(image_paths) / (end_time - start_time):.2f} images/second")
```

---

*Continue to [Contributing](../development/contributing.md) to learn about contributing to the project.*