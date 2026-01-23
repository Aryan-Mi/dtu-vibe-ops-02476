# Installation Guide

## Prerequisites

- **Python 3.12**
- **Git**
- **4GB+ RAM** (8GB+ recommended)

## Installing uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

## Project Setup

```bash
# Clone repository
git clone https://github.com/Aryan-Mi/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476

# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## GPU Support (Optional)

```bash
# Install GPU-enabled PyTorch
uv remove torch torchvision
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Data Setup

```bash
# Pull dataset with DVC
uv run dvc pull

# Verify
ls data/
```

## Development Setup

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest tests/ -v

# Check code style
uv run ruff check src/
```

## Common Issues

### Python Version Mismatch
```bash
pyenv install 3.12
pyenv local 3.12
uv sync
```

### CUDA Version Mismatch
```bash
nvidia-smi  # Check CUDA version
uv add torch --index-url https://download.pytorch.org/whl/cu118
```

## Next Steps

- [Quick Start Tutorial](getting-started.md)
- [Training Guide](user-guide/training.md)
