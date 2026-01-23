# Installation Guide

This guide will help you set up the MLOps Skin Lesion Classification Pipeline on your local machine. We use **uv** for fast and reliable Python dependency management.

## Prerequisites

### System Requirements

- **Python 3.12** (required)
- **Git** for version control
- **4GB+ RAM** (8GB+ recommended for training)
- **2GB+ free disk space** (more for datasets)

### Operating System Support

<div class="grid">
  <div class="card">
    <h3>🐧 Linux</h3>
    <p>Ubuntu 20.04+, CentOS 7+, or any modern distribution</p>
  </div>
  
  <div class="card">
    <h3>🍎 macOS</h3>
    <p>macOS 11+ (Big Sur) with Intel or Apple Silicon</p>
  </div>
  
  <div class="card">
    <h3>🪟 Windows</h3>
    <p>Windows 10+ or Windows Subsystem for Linux (WSL2)</p>
  </div>
</div>

## Installing uv

**uv** is a modern Python package manager that's significantly faster than pip and provides better dependency resolution.

### Quick Installation

=== "Linux/macOS"

    ```bash
    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add to your shell profile (if not done automatically)
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

=== "Windows"

    ```powershell
    # Install using PowerShell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Or using Scoop
    scoop install uv
    
    # Or using Chocolatey
    choco install uv
    ```

=== "Alternative Methods"

    ```bash
    # Using pip (if you have Python already)
    pip install uv
    
    # Using Homebrew (macOS)
    brew install uv
    
    # Using Conda
    conda install -c conda-forge uv
    ```

### Verify Installation

```bash
uv --version
```

You should see output like: `uv 0.4.30`

!!! tip "Why uv?"
    - **10-100x faster** than pip for package installation
    - **Better dependency resolution** with conflict detection  
    - **Automatic virtual environment** management
    - **Lock files** for reproducible environments
    - **Compatible** with existing pip/requirements.txt workflows

## Project Setup

### 1. Clone the Repository

```bash
# Clone the project repository
git clone https://github.com/DTU-Group-36/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476
```

### 2. Set Up Python Environment

uv will automatically create and manage a virtual environment for you:

```bash
# Install all dependencies (including dev dependencies)
uv sync

# Or install only production dependencies
uv sync --no-dev
```

This command will:
- ✅ Create a virtual environment with Python 3.12
- ✅ Install all dependencies from `pyproject.toml`
- ✅ Generate a `uv.lock` file for reproducibility
- ✅ Set up PyTorch with CPU support by default

### 3. Verify Installation

```bash
# Run a quick test to verify everything works
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
```

Expected output:
```
PyTorch: 2.2.2
Lightning: 2.2.1
```

## GPU Support (Optional)

For faster training on NVIDIA GPUs:

### CUDA Installation

1. **Install NVIDIA drivers** for your GPU
2. **Install CUDA Toolkit 12.1+** from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### PyTorch GPU Setup

```bash
# Remove CPU-only PyTorch
uv remove torch torchvision

# Install GPU-enabled PyTorch  
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU Support

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Development Setup

For contributing to the project:

### 1. Install Development Dependencies

```bash
# Development dependencies are included by default with 'uv sync'
uv sync
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks for code quality
uv run pre-commit install

# Test the hooks
uv run pre-commit run --all-files
```

### 3. Verify Development Environment

```bash
# Run the test suite
uv run python -m pytest tests/ -v

# Check code formatting
uv run ruff check src/ tests/

# Generate coverage report
uv run python -m pytest tests/ --cov=src --cov-report=html
```

## Data Setup

### Download Dataset

The HAM10000 dataset is managed with DVC (Data Version Control):

```bash
# Configure DVC (if needed for Google Cloud Storage)
uv run dvc remote modify myremote url gs://your-bucket-name

# Pull the dataset
uv run dvc pull
```

!!! note "Data Size"
    The HAM10000 dataset is approximately **1.5GB** compressed. Ensure you have sufficient disk space and a stable internet connection.

### Verify Data Setup

```bash
# Check data structure
ls -la data/
ls -la data/ham10000/

# Quick data validation
uv run python -c "
from src.mlops_project.data import HAM10000Dataset
dataset = HAM10000Dataset('data/ham10000')
print(f'Dataset size: {len(dataset)} images')
"
```

## Environment Variables

Create a `.env` file for configuration (optional):

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your preferred settings
nano .env
```

Example `.env` contents:
```bash
# Weights & Biases (optional)
WANDB_ENTITY=your-wandb-username
WANDB_PROJECT=skin-lesion-classification

# Google Cloud (for deployment)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GCP_PROJECT_ID=your-project-id

# Model settings
DEFAULT_MODEL=efficientnet
DEFAULT_BATCH_SIZE=32
```

## Common Issues and Solutions

### Issue: Python Version Mismatch

```bash
# Error: This project requires Python 3.12
# Solution: Install Python 3.12 or use pyenv
pyenv install 3.12
pyenv local 3.12
uv sync
```

### Issue: CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8
```

### Issue: Permission Errors (Linux/macOS)

```bash
# Don't use sudo with uv
# Instead, ensure proper permissions:
chown -R $USER:$USER ~/.local/share/uv
```

### Issue: Corporate Firewall/Proxy

```bash
# Configure proxy for uv
export HTTPS_PROXY=http://proxy.company.com:8080
export HTTP_PROXY=http://proxy.company.com:8080

# Or use alternative index
uv add torch --index-url https://pypi.org/simple/
```

## Performance Optimization

### Memory Settings

For large model training:

```bash
# Set environment variables for better memory usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Disk Space Optimization

```bash
# Clean uv cache to free disk space
uv cache clean

# Use subsample for development
uv run src/mlops_project/train.py data.subsample_percentage=0.01
```

## Next Steps

Once installation is complete:

1. **[Quick Start Tutorial](getting-started.md)** - Train your first model
2. **[System Architecture](architecture.md)** - Understand the pipeline
3. **[Training Guide](user-guide/training.md)** - Detailed training instructions
4. **[API Usage](examples/api-usage.md)** - Set up inference server

## Troubleshooting

If you encounter issues:

1. **Check the [FAQ section](#common-issues-and-solutions)** above
2. **Review logs** in the `logs/` directory
3. **Search [existing issues](https://github.com/DTU-Group-36/dtu-vibe-ops-02476/issues)**
4. **Create a new issue** with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the problem

---

*Ready to start training? Continue to the [Getting Started Guide](getting-started.md)!*