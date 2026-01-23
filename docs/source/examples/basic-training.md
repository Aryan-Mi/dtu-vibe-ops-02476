# Basic Training Examples

Practical examples for training skin lesion classification models with different configurations and use cases.

## Quick Start Examples

### 1. Minimal Training Example

Perfect for testing the setup and getting familiar with the pipeline:

```bash
# Train for 2 epochs with 1% of data
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.01 \
    training.max_epochs=2 \
    data.batch_size=8
```

**Expected Output:**
```
🔬 Starting training with BaselineCNN...
📊 Using 100 training samples (1% subsample)
📈 Training progress:
Epoch 1/2: 100%|████████| 13/13 [00:15<00:00, loss=0.693]
Epoch 2/2: 100%|████████| 13/13 [00:14<00:00, loss=0.512]
✅ Training completed! Model saved to: models/baseline_cnn/version_0/
```

### 2. GPU Training with EfficientNet

Train the best performing model with GPU acceleration:

```bash
# Full EfficientNet-B2 training
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b2 \
    training.max_epochs=25 \
    data.image_size=260 \
    data.batch_size=32 \
    training.accelerator=gpu
```

### 3. CPU Training for Development

When GPU isn't available or for debugging:

```bash
# CPU training with smaller model
uv run src/mlops_project/train.py \
    model=resnet \
    training.max_epochs=10 \
    data.batch_size=16 \
    training.accelerator=cpu \
    data.num_workers=4
```

## Model Architecture Examples

### Baseline CNN Training

Simple convolutional network for quick experimentation:

```bash
# Baseline with data augmentation
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.augment=true \
    data.horizontal_flip=0.5 \
    data.rotation_degrees=15 \
    training.learning_rate=0.001 \
    training.max_epochs=20
```

**Model Configuration:**
```yaml
# configs/model/baseline_cnn.yaml
name: baseline_cnn
num_classes: 2
dropout_rate: 0.5
learning_rate: 0.001
weight_decay: 0.01

architecture:
  conv_channels: [32, 64, 128, 256]
  classifier_hidden: 128
  use_batch_norm: true
```

### ResNet Training

Residual network with skip connections:

```bash
# ResNet-18 equivalent training
uv run src/mlops_project/train.py \
    model=resnet \
    model.num_layers=18 \
    training.learning_rate=0.0001 \
    training.weight_decay=0.01 \
    data.batch_size=32 \
    training.max_epochs=30
```

### EfficientNet Variants

Train different EfficientNet variants based on your computational resources:

#### EfficientNet-B0 (Fastest)
```bash
# Lightweight option for quick training
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b0 \
    data.image_size=224 \
    data.batch_size=64 \
    training.max_epochs=30
```

#### EfficientNet-B3 (Balanced)
```bash
# Good balance of performance and speed
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    data.image_size=300 \
    data.batch_size=32 \
    training.max_epochs=25 \
    training.precision=16  # Mixed precision for speed
```

#### EfficientNet-B5 (Best Performance)
```bash
# Maximum accuracy (requires more GPU memory)
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b5 \
    data.image_size=456 \
    data.batch_size=16 \
    training.max_epochs=50 \
    training.accumulate_grad_batches=2
```

## Data Configuration Examples

### Subsampling for Development

Useful for quick iteration and testing:

```bash
# Train on 5% of data for development
uv run src/mlops_project/train.py \
    data.subsample_percentage=0.05 \
    data.stratified_sampling=true \
    training.max_epochs=10 \
    wandb.tags=["development","subset"]
```

### Data Augmentation Configurations

#### Minimal Augmentation
```bash
# Basic augmentation for stable training
uv run src/mlops_project/train.py \
    data.augment=true \
    data.horizontal_flip=0.5 \
    data.normalize=true
```

#### Aggressive Augmentation
```bash
# Strong augmentation for better generalization
uv run src/mlops_project/train.py \
    data.augment=true \
    data.horizontal_flip=0.5 \
    data.vertical_flip=0.3 \
    data.rotation_degrees=30 \
    data.color_jitter=0.4 \
    data.random_crop=0.8
```

#### Medical-Specific Augmentation
```bash
# Augmentation suitable for dermatoscopic images
uv run src/mlops_project/train.py \
    data.augment=true \
    data.horizontal_flip=0.5 \
    data.vertical_flip=0.5 \
    data.rotation_degrees=45 \
    data.brightness_factor=0.2 \
    data.contrast_factor=0.2
```

## Training Optimization Examples

### Learning Rate Optimization

#### Learning Rate Scheduling
```bash
# Cosine annealing for smooth learning rate decay
uv run src/mlops_project/train.py \
    training.learning_rate=0.001 \
    training.lr_scheduler=cosine \
    training.max_epochs=50 \
    training.min_lr=0.00001
```

#### Reduce on Plateau
```bash
# Adaptive learning rate based on validation loss
uv run src/mlops_project/train.py \
    training.learning_rate=0.001 \
    training.lr_scheduler=reduce_on_plateau \
    training.lr_patience=5 \
    training.lr_factor=0.5
```

### Memory Optimization

#### Gradient Accumulation
```bash
# Simulate larger batch size with limited memory
uv run src/mlops_project/train.py \
    data.batch_size=16 \
    training.accumulate_grad_batches=4 \  # Effective batch size = 64
    training.gradient_clip_val=1.0
```

#### Mixed Precision Training
```bash
# Faster training with reduced memory usage
uv run src/mlops_project/train.py \
    training.precision=16 \
    training.amp_backend=native \
    data.batch_size=48  # Can use larger batch size
```

## Transfer Learning Examples

### Fine-tuning Pretrained Models

#### Conservative Fine-tuning
```bash
# Lower learning rate for pretrained backbone
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.pretrained=true \
    training.learning_rate=0.0001 \
    training.backbone_lr_factor=0.1 \
    training.freeze_backbone_epochs=5
```

#### Progressive Unfreezing
```bash
# Gradually unfreeze model layers
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.pretrained=true \
    training.progressive_unfreezing=true \
    training.unfreeze_schedule="[5,10,15,20]" \
    training.max_epochs=30
```

### From-Scratch Training

```bash
# Train without pretrained weights
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.pretrained=false \
    training.learning_rate=0.01 \
    training.warmup_epochs=5 \
    training.max_epochs=100
```

## Multi-GPU Training Examples

### Distributed Training

```bash
# Train on multiple GPUs with DDP
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    training.accelerator=gpu \
    training.devices=2 \
    training.strategy=ddp \
    data.batch_size=32  # Per GPU batch size
```

### Model Parallel Training

```bash
# For very large models
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b7 \
    training.strategy=model_parallel \
    training.accelerator=gpu \
    training.devices=4
```

## Experiment Tracking Examples

### Weights & Biases Integration

#### Basic W&B Logging
```bash
# Enable experiment tracking
uv run src/mlops_project/train.py \
    wandb.enabled=true \
    wandb.project="skin-lesion-classification" \
    wandb.name="efficientnet-b3-baseline" \
    wandb.tags=["efficientnet","baseline"]
```

#### Advanced W&B Features
```bash
# Log model artifacts and watch gradients
uv run src/mlops_project/train.py \
    wandb.enabled=true \
    wandb.log_model=true \
    wandb.watch_model=gradients \
    wandb.log_code=true \
    wandb.notes="Testing with enhanced augmentation"
```

### Local Experiment Organization

```bash
# Organized local experiments
uv run src/mlops_project/train.py \
    model=efficientnet \
    training.experiment_name="efficientnet_comparison" \
    training.version_name="b3_augmented" \
    training.save_dir="experiments/efficientnet_study"
```

## Production Training Examples

### Full Production Pipeline

```bash
# Complete production training setup
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b4 \
    model.pretrained=true \
    data.image_size=380 \
    data.batch_size=24 \
    data.augment=true \
    training.max_epochs=75 \
    training.learning_rate=0.0001 \
    training.lr_scheduler=cosine \
    training.early_stopping_patience=15 \
    training.precision=16 \
    wandb.enabled=true \
    wandb.project="production" \
    wandb.name="efficientnet-b4-final" \
    wandb.tags=["production","final"]
```

### Reproducible Training

```bash
# Ensure reproducible results
uv run src/mlops_project/train.py \
    model=efficientnet \
    training.seed=42 \
    training.deterministic=true \
    training.benchmark=false \
    data.num_workers=1  # Deterministic data loading
```

## Debugging Examples

### Fast Debug Run

```bash
# Quick debug with minimal data
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    training.fast_dev_run=true \
    training.limit_train_batches=5 \
    training.limit_val_batches=3
```

### Overfitting Test

```bash
# Test if model can overfit small dataset
uv run src/mlops_project/train.py \
    model=efficientnet \
    training.overfit_batches=10 \
    training.max_epochs=50 \
    training.learning_rate=0.01
```

### Profiling Run

```bash
# Profile training performance
uv run src/mlops_project/train.py \
    model=efficientnet \
    training.profiler=advanced \
    training.max_epochs=1 \
    data.subsample_percentage=0.1 \
    training.log_every_n_steps=1
```

## Expected Training Times

### Estimated Training Durations

| Model | GPU | Epochs | Full Dataset | 10% Subset |
|-------|-----|--------|--------------|------------|
| BaselineCNN | GTX 1080 | 20 | 25 min | 3 min |
| ResNet-18 | GTX 1080 | 25 | 65 min | 8 min |
| EfficientNet-B0 | V100 | 30 | 45 min | 5 min |
| EfficientNet-B3 | V100 | 25 | 90 min | 12 min |
| EfficientNet-B5 | V100 | 50 | 4 hours | 30 min |

### Performance Monitoring

Monitor your training with:

```bash
# Check GPU usage
nvidia-smi

# Monitor system resources  
htop

# View training logs
tail -f logs/train_*.log
```

---

*Continue to [Custom Configuration](custom-config.md) to learn about advanced Hydra configuration patterns.*