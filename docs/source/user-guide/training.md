# Training Models

This guide covers everything you need to know about training skin lesion classification models, from basic usage to advanced techniques and optimization strategies.

## Overview

Our training system is built on **PyTorch Lightning** with **Hydra** configuration management, providing a flexible and reproducible training experience. You can train three different model architectures with various configurations.

## Quick Training Examples

### Basic Training Commands

```bash
# Train baseline CNN (fastest, good for testing)
uv run src/mlops_project/train.py model=baseline_cnn

# Train ResNet (balanced performance)  
uv run src/mlops_project/train.py model=resnet

# Train EfficientNet (best performance)
uv run src/mlops_project/train.py model=efficientnet
```

### Common Training Scenarios

#### Development Testing
```bash
# Quick test with minimal data
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.01 \
    training.max_epochs=2 \
    data.batch_size=8
```

#### Full Production Training
```bash
# Production training with EfficientNet-B3
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    training.max_epochs=50 \
    training.learning_rate=0.0001 \
    data.batch_size=16 \
    data.image_size=300 \
    wandb.enabled=true
```

## Configuration Deep Dive

### Model Selection and Variants

#### Baseline CNN Configuration
```yaml
# configs/model/baseline_cnn.yaml
name: baseline_cnn
num_classes: 2
dropout_rate: 0.5
learning_rate: 0.001
weight_decay: 0.01

architecture:
  conv_blocks: 4
  channels: [32, 64, 128, 256]
  classifier_hidden: 128
```

#### EfficientNet Variants
```bash
# Available EfficientNet variants with optimal settings
uv run src/mlops_project/train.py model=efficientnet model.variant=b0 data.image_size=224  # 5.3M params
uv run src/mlops_project/train.py model=efficientnet model.variant=b1 data.image_size=240  # 7.8M params  
uv run src/mlops_project/train.py model=efficientnet model.variant=b2 data.image_size=260  # 9.1M params
uv run src/mlops_project/train.py model=efficientnet model.variant=b3 data.image_size=300  # 12M params
uv run src/mlops_project/train.py model=efficientnet model.variant=b4 data.image_size=380  # 19M params
```

### Training Parameters

#### Learning Rate and Optimization
```bash
# Learning rate scheduling
uv run src/mlops_project/train.py \
    training.learning_rate=0.001 \
    training.lr_scheduler=cosine \
    training.weight_decay=0.01 \
    training.optimizer=adamw
```

#### Batch Size and Memory Management
```bash
# Optimize for available memory
uv run src/mlops_project/train.py \
    data.batch_size=32 \                      # Adjust based on GPU memory
    training.accumulate_grad_batches=2 \      # Effective batch size = 64
    data.num_workers=4                        # Parallel data loading
```

#### Early Stopping and Checkpointing
```bash
# Configure early stopping
uv run src/mlops_project/train.py \
    training.early_stopping_patience=10 \
    training.monitor_metric=val_loss \
    training.save_top_k=3
```

### Data Configuration

#### Data Augmentation
```bash
# Enable comprehensive data augmentation
uv run src/mlops_project/train.py \
    data.augment=true \
    data.horizontal_flip=0.5 \
    data.vertical_flip=0.2 \
    data.rotation_degrees=15 \
    data.color_jitter=0.3
```

#### Dataset Subsampling
```bash
# Use subset for faster iteration
uv run src/mlops_project/train.py \
    data.subsample_percentage=0.1 \           # Use 10% of data
    data.stratified_sampling=true            # Maintain class balance
```

## Training Workflows

### 1. Hyperparameter Search

#### Manual Grid Search
```bash
# Search over learning rates
for lr in 0.1 0.01 0.001 0.0001; do
    uv run src/mlops_project/train.py \
        training.learning_rate=$lr \
        wandb.tags=["lr-search"] \
        wandb.name="lr_${lr}"
done

# Search over batch sizes
for batch_size in 16 32 64; do
    uv run src/mlops_project/train.py \
        data.batch_size=$batch_size \
        wandb.tags=["batch-search"] \
        wandb.name="batch_${batch_size}"
done
```

#### Weights & Biases Sweeps
```bash
# Start hyperparameter sweep (requires wandb setup)
uv run wandb sweep configs/sweeps.yaml
uv run wandb agent <sweep-id>
```

Example sweep configuration:
```yaml
# configs/sweeps.yaml
program: src/mlops_project/train.py
method: random
metric:
  name: val_f1_score
  goal: maximize
parameters:
  training.learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.01
  data.batch_size:
    values: [16, 32, 64]
  model.variant:
    values: ["b0", "b1", "b2"]
```

### 2. Progressive Training Strategy

```bash
# Stage 1: Quick baseline with small model
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.1 \
    training.max_epochs=10

# Stage 2: ResNet with more data  
uv run src/mlops_project/train.py \
    model=resnet \
    data.subsample_percentage=0.5 \
    training.max_epochs=25

# Stage 3: Full EfficientNet training
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    training.max_epochs=50 \
    wandb.enabled=true
```

### 3. Transfer Learning and Fine-tuning

#### Custom Transfer Learning
```bash
# Fine-tune pretrained model with lower learning rate
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.pretrained=true \
    training.learning_rate=0.0001 \
    training.freeze_backbone_epochs=5 \     # Freeze backbone for 5 epochs
    training.unfreeze_lr_factor=0.1         # Lower LR for backbone
```

#### Domain Adaptation
```bash
# Progressive unfreezing strategy
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b2 \
    training.progressive_unfreezing=true \
    training.unfreeze_schedule="[5,10,15]"  # Unfreeze layers at epochs 5,10,15
```

## Advanced Training Techniques

### Mixed Precision Training

```bash
# Enable automatic mixed precision for faster training
uv run src/mlops_project/train.py \
    training.precision=16 \                 # Use FP16
    training.amp_backend=native             # Native PyTorch AMP
```

### Multi-GPU Training

```bash
# Distributed training on multiple GPUs
uv run src/mlops_project/train.py \
    training.accelerator=gpu \
    training.devices=2 \                    # Use 2 GPUs
    training.strategy=ddp                   # Distributed Data Parallel
```

### Gradient Accumulation

```bash
# Simulate larger batch sizes with gradient accumulation
uv run src/mlops_project/train.py \
    data.batch_size=16 \
    training.accumulate_grad_batches=4 \    # Effective batch size = 64
    training.gradient_clip_val=1.0          # Gradient clipping
```

## Monitoring and Debugging

### Local Monitoring

#### Training Progress
```bash
# Monitor training with rich progress bars
uv run src/mlops_project/train.py \
    training.progress_bar=rich \
    training.log_every_n_steps=10
```

#### Model Profiling
```bash
# Profile model performance
uv run src/mlops_project/train.py \
    training.profiler=simple \              # or 'advanced', 'pytorch'
    training.max_epochs=1 \
    data.subsample_percentage=0.01
```

### Weights & Biases Integration

#### Basic W&B Setup
```bash
# Initialize W&B (first time only)
uv run wandb login

# Train with W&B logging
uv run src/mlops_project/train.py \
    wandb.enabled=true \
    wandb.project="skin-lesion-classification" \
    wandb.entity="your-username" \
    wandb.name="efficientnet-b3-full" \
    wandb.tags=["production","efficientnet"]
```

#### Advanced W&B Features
```bash
# Log additional artifacts
uv run src/mlops_project/train.py \
    wandb.enabled=true \
    wandb.log_model=true \                  # Upload model checkpoints
    wandb.log_code=true \                   # Log source code
    wandb.watch_model=gradients             # Log gradients and parameters
```

## Troubleshooting Training Issues

### Common Problems and Solutions

#### Out of Memory (OOM) Errors

**Problem**: CUDA out of memory during training
```bash
# Solution 1: Reduce batch size
uv run src/mlops_project/train.py data.batch_size=8

# Solution 2: Use gradient accumulation
uv run src/mlops_project/train.py \
    data.batch_size=8 \
    training.accumulate_grad_batches=4

# Solution 3: Use smaller model variant
uv run src/mlops_project/train.py model=efficientnet model.variant=b0
```

#### Slow Training Performance

**Problem**: Training is slower than expected
```bash
# Solution 1: Increase number of data workers
uv run src/mlops_project/train.py data.num_workers=8

# Solution 2: Use mixed precision
uv run src/mlops_project/train.py training.precision=16

# Solution 3: Optimize batch size
uv run src/mlops_project/train.py data.batch_size=64  # Larger if memory allows
```

#### Model Not Converging

**Problem**: Loss not decreasing or oscillating
```bash
# Solution 1: Reduce learning rate
uv run src/mlops_project/train.py training.learning_rate=0.0001

# Solution 2: Add learning rate scheduling
uv run src/mlops_project/train.py \
    training.lr_scheduler=reduce_on_plateau \
    training.lr_patience=5

# Solution 3: Check data loading
uv run src/mlops_project/train.py \
    data.subsample_percentage=0.01 \
    training.max_epochs=1                   # Quick debug run
```

### Debugging Configurations

#### Quick Debug Setup
```bash
# Fast debug configuration
uv run src/mlops_project/train.py \
    model=baseline_cnn \
    data.subsample_percentage=0.001 \       # Tiny dataset
    training.max_epochs=1 \
    data.batch_size=2 \
    training.limit_train_batches=10 \       # Only 10 batches
    training.limit_val_batches=5
```

#### Data Pipeline Debugging
```bash
# Check data loading without training
uv run python -c "
from src.mlops_project.data import HAM10000DataModule
dm = HAM10000DataModule('data/ham10000', batch_size=4)
dm.setup('fit')
batch = next(iter(dm.train_dataloader()))
print(f'Batch shape: {batch[0].shape}, Labels: {batch[1]}')
"
```

## Performance Optimization

### Training Speed Optimization

#### Optimized Training Configuration
```bash
# High-performance training setup
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b2 \
    data.batch_size=32 \
    data.num_workers=8 \
    training.precision=16 \
    training.accelerator=gpu \
    training.compile=true \                 # PyTorch 2.0+ compilation
    data.pin_memory=true \
    data.persistent_workers=true
```

#### Memory Optimization
```bash
# Memory-efficient training
uv run src/mlops_project/train.py \
    training.accumulate_grad_batches=4 \
    training.gradient_clip_val=1.0 \
    training.enable_checkpointing=true \    # Gradient checkpointing
    data.batch_size=16
```

### Model Quality Optimization

#### Best Practices Configuration
```bash
# Production-quality training
uv run src/mlops_project/train.py \
    model=efficientnet \
    model.variant=b3 \
    model.pretrained=true \
    training.max_epochs=100 \
    training.early_stopping_patience=15 \
    training.learning_rate=0.0001 \
    training.weight_decay=0.01 \
    training.lr_scheduler=cosine \
    data.augment=true \
    data.image_size=300 \
    wandb.enabled=true
```

## Output and Results

### Training Outputs

After training completion, you'll find:

```
models/
├── efficientnet/
│   ├── version_0/
│   │   ├── checkpoints/
│   │   │   ├── epoch=024-val_loss=0.342.ckpt  # Best checkpoint
│   │   │   ├── epoch=049-val_loss=0.356.ckpt  # Other saved checkpoints
│   │   │   └── last.ckpt                      # Latest checkpoint
│   │   ├── hparams.yaml                       # Hyperparameters
│   │   └── events.out.tfevents.*             # TensorBoard logs
│   
logs/
├── train_2024-01-23_15-30-42.log             # Training logs
└── lightning_logs/                            # Lightning logs

reports/
└── training_metrics.csv                       # Training metrics summary
```

### Checkpoint Management

```bash
# Resume training from checkpoint
uv run src/mlops_project/train.py \
    training.resume_from_checkpoint=models/efficientnet/version_0/checkpoints/last.ckpt

# Load specific checkpoint for evaluation
uv run src/mlops_project/evaluate.py \
    --checkpoint models/efficientnet/version_0/checkpoints/epoch=024-val_loss=0.342.ckpt
```

---

*Next: Learn how to [evaluate your trained models](evaluation.md) and measure their performance.*