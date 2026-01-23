# MLOps Skin Lesion Classification Pipeline

<div class="hero">
  <h1>🔬 Advanced MLOps Pipeline for Medical AI</h1>
  <p>End-to-end skin lesion classification using modern ML practices, PyTorch Lightning, and production-ready deployment</p>
</div>

## Overview

Welcome to the **MLOps Skin Lesion Classification Pipeline** - a comprehensive, production-ready machine learning system designed by **DTU Group 36** for the MLOps course (02476). This project demonstrates best practices in machine learning operations through a real-world medical AI application.

Our pipeline classifies dermatoscopic images of skin lesions to estimate **malignant vs. benign diagnosis**, showcasing a complete MLOps workflow from data handling to cloud deployment.

### 🎯 Project Goals

- **Medical Impact**: Build a reliable skin lesion classification system
- **MLOps Excellence**: Demonstrate industry-standard ML engineering practices  
- **Reproducibility**: Ensure consistent, traceable model development
- **Scalability**: Design for production deployment and monitoring

### ✨ Key Features

<div class="grid">
  <div class="card">
    <h3>🏗️ Three-Tier Architecture</h3>
    <p>Progressive model complexity from baseline CNN to transfer learning with EfficientNet</p>
  </div>
  
  <div class="card">
    <h3>⚡ Modern ML Stack</h3>
    <p>PyTorch Lightning, Hydra configuration, Weights & Biases tracking</p>
  </div>
  
  <div class="card">
    <h3>🔄 Complete MLOps Pipeline</h3>
    <p>DVC data versioning, automated testing, containerized deployment</p>
  </div>
  
  <div class="card">
    <h3>🚀 Production Ready</h3>
    <p>FastAPI inference server, Google Cloud deployment, monitoring</p>
  </div>
  
  <div class="card">
    <h3>🧪 Rigorous Testing</h3>
    <p>Cross-platform CI/CD, comprehensive unit tests, model validation</p>
  </div>
  
  <div class="card">
    <h3>📊 Experiment Tracking</h3>
    <p>W&B integration, Hydra sweeps, reproducible configurations</p>
  </div>
</div>

## 🩺 Medical Application

### The Challenge
Skin cancer is one of the most common forms of cancer worldwide. Early detection significantly improves patient outcomes, but requires specialized expertise that isn't always available.

### Our Solution
We've built an AI-powered diagnostic aid using the **HAM10000 dataset** - a comprehensive collection of dermatoscopic images representing seven types of skin lesions. Our system:

- Processes high-resolution dermatoscopic images
- Provides binary classification (malignant vs. benign)
- Offers confidence scores and interpretability
- Maintains medical-grade reliability standards

### Dataset Details
- **Source**: HAM10000 (Harvard Dataverse)
- **Images**: 10,015 dermatoscopic images
- **Classes**: 7 lesion types → Binary classification
- **Quality**: Medical-grade, expert-validated annotations

## 🏛️ Architecture Overview

Our system implements a sophisticated three-tier model architecture with comprehensive MLOps infrastructure:

<div class="feature-badge">Baseline CNN</div>
<div class="feature-badge">Residual Networks</div>
<div class="feature-badge">Transfer Learning</div>
<div class="feature-badge">FastAPI Deployment</div>
<div class="feature-badge">Cloud Native</div>

### Model Tiers

1. **Baseline CNN**: Simple convolutional architecture for baseline performance
2. **ResNet-style**: Residual connections for improved gradient flow
3. **EfficientNet**: State-of-the-art transfer learning with pretrained weights

[Learn more about our architecture →](architecture.md)

## 🚀 Quick Start

Get up and running in minutes with our modern Python tooling:

```bash
# Clone the repository
git clone https://github.com/DTU-Group-36/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476

# Set up environment with uv (recommended)
uv sync

# Train a model
uv run src/mlops_project/train.py model=efficientnet

# Start inference server
uv run src/mlops_project/api.py
```

[Detailed installation guide →](installation.md) | [Complete tutorial →](getting-started.md)

## 📚 Documentation Structure

<div class="grid">
  <div class="card">
    <h3>🏁 [Getting Started](getting-started.md)</h3>
    <p>Quick tutorial to train your first model</p>
  </div>
  
  <div class="card">
    <h3>⚙️ [Installation](installation.md)</h3>
    <p>Environment setup with uv and dependencies</p>
  </div>
  
  <div class="card">
    <h3>🏗️ [System Architecture](architecture.md)</h3>
    <p>Detailed system design and component overview</p>
  </div>
  
  <div class="card">
    <h3>📖 [User Guides](user-guide/training.md)</h3>
    <p>Training, evaluation, inference, and deployment</p>
  </div>
  
  <div class="card">
    <h3>💡 [Examples](examples/basic-training.md)</h3>
    <p>Practical code examples and tutorials</p>
  </div>
  
  <div class="card">
    <h3>🔧 [API Reference](api-reference/models.md)</h3>
    <p>Complete API documentation with examples</p>
  </div>
</div>

## 🛠️ Technology Stack

### Core ML Framework
- **PyTorch 2.2.2**: Deep learning framework
- **PyTorch Lightning 2.2.1**: Training orchestration
- **TorchVision 0.17.2**: Computer vision utilities

### MLOps & Infrastructure  
- **Hydra**: Configuration management
- **DVC**: Data and model versioning
- **Weights & Biases**: Experiment tracking
- **FastAPI**: Production inference server
- **Docker**: Containerization
- **Google Cloud**: Deployment platform

### Development Tools
- **uv**: Fast Python package management
- **Ruff**: Code formatting and linting  
- **pytest**: Testing framework
- **GitHub Actions**: CI/CD automation
- **MkDocs Material**: Documentation

## 👥 Team - DTU Group 36

- **Aryan Mirzazadeh**
- **Mohamad Marwan Summakieh**  
- **Trinity Sara McConnachie Evans**
- **Vladyslav Horbatenko**
- **Yuen Yi Hui**

## 🎓 Academic Context

This project was developed for the **MLOps course (02476)** at the Danish Technical University (DTU) during Winter 2026. It serves as a comprehensive demonstration of modern machine learning engineering practices applied to a real-world medical AI problem.

### Learning Objectives Achieved
- ✅ Reproducible ML pipelines
- ✅ Version control for data and models  
- ✅ Automated testing and deployment
- ✅ Experiment tracking and monitoring
- ✅ Production-ready system design
- ✅ Documentation and knowledge sharing

## 🤝 Contributing

We welcome contributions! See our [contributing guide](development/contributing.md) for details on:

- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Code style guidelines

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/DTU-Group-36/dtu-vibe-ops-02476/blob/main/LICENSE) file for details.

## 🔗 Quick Links

- **[GitHub Repository](https://github.com/DTU-Group-36/dtu-vibe-ops-02476)**
- **[Issue Tracker](https://github.com/DTU-Group-36/dtu-vibe-ops-02476/issues)**
- **[Deployment Dashboard](https://console.cloud.google.com/)**
- **[W&B Project](https://wandb.ai/)**

---

*Ready to dive in? Start with our [installation guide](installation.md) or jump straight to [training your first model](getting-started.md)!*