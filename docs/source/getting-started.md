# Getting Started

Welcome to the MLOps Skin Lesion Classification Pipeline! This guide will walk you through training your first model in just a few minutes.

## Quick Start Checklist

Before we begin, make sure you have:

- ✅ **Installed uv** and set up the environment ([Installation Guide](installation.md))
- ✅ **Cloned the repository** and run `uv sync`
- ✅ **Downloaded the dataset** (optional for this tutorial)

## Your First Model in 3 Steps

### Step 1: Train a Baseline Model

Let's start with a quick training run using a subset of data:

```bash
# Train a baseline CNN with minimal data for testing
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.01 \
    training.max_epochs=2 \
    data.batch_size=8
```

This command will:
- 🏃‍♂️ **Train quickly** using only 1% of the dataset
- 📊 **Log metrics** to the console and local files
- 💾 **Save checkpoints** in the `models/` directory
- 📈 **Track progress** with built-in logging

**Expected Output:**
```
[INFO] Starting training with baseline_cnn...
[INFO] Using 100 training samples (1% subsample)
Epoch 1/2: 100%|████████| 13/13 [00:15<00:00, loss=0.693]
Epoch 2/2: 100%|████████| 13/13 [00:14<00:00, loss=0.512]
[INFO] Training completed! Best model saved.
```

### Step 2: Evaluate Your Model

Run evaluation to see how well your model performed:

```bash
# Evaluate the trained model
uv run src/mlops_project/evaluate.py \
    --model-path models/baseline_cnn/latest.ckpt \
    --data-path data/ham10000 \
    --batch-size 16
```

**Expected Output:**
```
📊 Evaluation Results:
├── Accuracy: 0.72
├── F1 Score: 0.68
├── Precision: 0.71
├── Recall: 0.65
└── AUC-ROC: 0.78

📈 Confusion Matrix saved to: reports/confusion_matrix.png
```

### Step 3: Run Inference

Start the API server and make predictions:

```bash
# Start the FastAPI inference server
uv run src/mlops_project/api.py
```

In another terminal, test the API:

```bash
# Test with a sample image
curl -X POST "http://localhost:8000/predict" \
     -F "file=@data/ham10000/HAM10000_images/ISIC_0024306.jpg" \
     | jq
```

**Expected Response:**
```json
{
  "prediction": "benign",
  "confidence": 0.87,
  "class_probabilities": {
    "malignant": 0.13,
    "benign": 0.87
  },
  "model_info": {
    "name": "baseline_cnn",
    "version": "1.0.0"
  }
}
```

🎉 **Congratulations!** You've successfully trained, evaluated, and deployed your first skin lesion classification model!

## Understanding the Configuration System

Our pipeline uses [Hydra](https://hydra.cc/) for flexible configuration management. Let's explore the key concepts:

### Configuration Structure

```
configs/
├── config.yaml          # Main configuration with defaults
├── model/               # Model architecture configurations
│   ├── baseline_cnn.yaml
│   ├── resnet.yaml
│   └── efficientnet.yaml
└── sweeps.yaml          # Hyperparameter sweep configurations
```

### Key Configuration Options

#### Model Selection
```bash
# Train different model architectures
uv run src/mlops_project/train.py model=baseline_cnn    # Simple CNN
uv run src/mlops_project/train.py model=resnet          # Residual network
uv run src/mlops_project/train.py model=efficientnet    # Transfer learning
```

#### Training Parameters
```bash
# Adjust training settings
uv run src/mlops_project/train.py \
    training.max_epochs=50 \
    training.learning_rate=0.001 \
    data.batch_size=32 \
    data.image_size=224
```

#### Data Configuration
```bash
# Control data usage
uv run src/mlops_project/train.py \
    data.subsample_percentage=0.1 \     # Use 10% of data
    data.augment=true \                 # Enable data augmentation
    data.num_workers=4                  # Parallel data loading
```

### Example: Full Training Run

For a complete training run with the best model:

```bash
# Train EfficientNet with full dataset
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    training.max_epochs=25 \
    training.learning_rate=0.0001 \
    data.batch_size=16 \
    data.image_size=300 \
    wandb.enabled=true
```

This will:
- 🚀 **Use EfficientNet-B3** (state-of-the-art transfer learning)
- 📊 **Train for 25 epochs** with the full dataset
- 🔄 **Log to Weights & Biases** for experiment tracking
- 🎯 **Automatically adjust** image size for the model variant

## Experiment Tracking with Weights & Biases

### Set Up W&B (Optional)

```bash
# Install and configure Weights & Biases
uv add wandb
uv run wandb login
```

### Train with Experiment Tracking

```bash
# Enable W&B logging
uv run src/mlops_project/train.py \
    wandb.enabled=true \
    wandb.project="skin-lesion-classification" \
    wandb.tags=["baseline","quick-test"]
```

Visit [wandb.ai](https://wandb.ai) to see real-time training metrics, model comparisons, and experiment logs.

### Key Metrics Tracked

- **Training/Validation Loss**: Model learning progress
- **Accuracy & F1 Score**: Classification performance  
- **Learning Rate**: Optimization schedule
- **GPU Utilization**: Resource usage
- **Model Artifacts**: Checkpoints and predictions

## Common Workflows

### 1. Quick Development Testing

```bash
# Fast iteration for development
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.005 \
    training.max_epochs=1 \
    data.batch_size=4
```

### 2. Hyperparameter Exploration

```bash
# Test different learning rates
for lr in 0.001 0.0001 0.00001; do
    uv run src/mlops_project/train.py \
        training.learning_rate=$lr \
        wandb.tags=["lr-sweep"] \
        wandb.name="lr_${lr}"
done
```

### 3. Model Comparison

```bash
# Train all three model types
for model in baseline_cnn resnet efficientnet; do
    uv run src/mlops_project/train.py \
        model=$model \
        wandb.tags=["model-comparison"] \
        wandb.name="comparison_${model}"
done
```

### 4. Production Training

```bash
# Full production training run
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b4 \
    training.max_epochs=50 \
    training.early_stopping_patience=10 \
    data.batch_size=32 \
    wandb.enabled=true \
    wandb.tags=["production","final-model"]
```

## Understanding the Output

### Training Logs

During training, you'll see output like:

```
Epoch 10/25:  40%|████      | 456/1142 [02:15<03:23,  3.37it/s, loss=0.512, v_num=123]
Train Loss: 0.512 | Val Loss: 0.487 | Val Acc: 0.742 | Val F1: 0.701
```

This shows:
- **Progress**: 40% through epoch 10 of 25
- **Speed**: Processing 3.37 batches per second  
- **Performance**: Training loss, validation metrics
- **Version**: Experiment number (v_num=123)

### Model Checkpoints

Models are automatically saved in:
```
models/
├── baseline_cnn/
│   ├── version_0/
│   │   ├── checkpoints/
│   │   │   ├── epoch=024-val_loss=0.45.ckpt  # Best checkpoint
│   │   │   └── last.ckpt                     # Latest checkpoint
│   │   └── hparams.yaml                      # Hyperparameters
```

### Experiment Artifacts

Additional outputs include:
- **`logs/`**: Training logs and metrics
- **`reports/`**: Evaluation plots and figures  
- **`wandb/`**: W&B local cache (if enabled)

## Next Steps

Now that you've trained your first model, explore more advanced features:

### 📚 **Deep Dive Guides**
- **[Training Guide](user-guide/training.md)**: Advanced training techniques
- **[Evaluation Guide](user-guide/evaluation.md)**: Comprehensive model evaluation  
- **[Deployment Guide](user-guide/deployment.md)**: Production deployment

### 🛠️ **Technical References**
- **[System Architecture](architecture.md)**: Understanding the pipeline design
- **[API Reference](api-reference/models.md)**: Complete API documentation
- **[Configuration Examples](examples/custom-config.md)**: Advanced Hydra usage

### 💡 **Practical Examples**
- **[Basic Training Examples](examples/basic-training.md)**: More training scenarios
- **[API Usage Examples](examples/api-usage.md)**: Integration patterns

## Troubleshooting

### Common Issues

**🔴 OutOfMemoryError**
```bash
# Reduce batch size
uv run src/mlops_project/train.py data.batch_size=8

# Use gradient accumulation
uv run src/mlops_project/train.py training.accumulate_grad_batches=4
```

**🔴 Data Not Found**  
```bash
# Use subsampling if dataset isn't available
uv run src/mlops_project/train.py data.subsample_percentage=0.01

# Or download the dataset
uv run dvc pull
```

**🔴 CUDA Out of Memory**
```bash
# Switch to CPU training
uv run src/mlops_project/train.py training.accelerator=cpu

# Or reduce model size
uv run src/mlops_project/train.py model=baseline_cnn
```

### Getting Help

- 📖 **Documentation**: Browse the [full documentation](index.md)
- 🐛 **Issues**: Report bugs on [GitHub Issues](https://github.com/DTU-Group-36/dtu-vibe-ops-02476/issues)
- 💬 **Discussions**: Join [GitHub Discussions](https://github.com/DTU-Group-36/dtu-vibe-ops-02476/discussions)

---

*Excellent work! You're now ready to explore the full capabilities of our MLOps pipeline. Happy experimenting! 🚀*