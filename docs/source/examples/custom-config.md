# Custom Configuration Examples

Advanced Hydra configuration patterns for customizing the MLOps pipeline.

## Configuration Fundamentals

### Basic Configuration Override

```bash
# Override single parameters
uv run src/mlops_project/train.py training.learning_rate=0.001

# Override multiple parameters
uv run src/mlops_project/train.py \
    training.learning_rate=0.001 \
    data.batch_size=64 \
    model.dropout_rate=0.3
```

### Configuration Groups

```bash
# Switch between model architectures
uv run src/mlops_project/train.py model=baseline_cnn
uv run src/mlops_project/train.py model=resnet  
uv run src/mlops_project/train.py model=efficientnet

# Use different model variants
uv run src/mlops_project/train.py model=efficientnet model.variant=b0
uv run src/mlops_project/train.py model=efficientnet model.variant=b3
```

## Creating Custom Configurations

### Custom Model Configuration

Create a new model configuration file:

```yaml
# configs/model/custom_efficientnet.yaml
name: efficientnet
variant: b2
num_classes: 2
pretrained: true
dropout_rate: 0.4
learning_rate: 0.0001

# Custom training parameters
training_config:
  freeze_backbone_epochs: 3
  use_mixup: true
  mixup_alpha: 0.2
  
# Custom augmentation
augmentation:
  rotation_range: [-30, 30]
  zoom_range: [0.8, 1.2]
  brightness_range: [0.8, 1.2]
```

Use your custom configuration:

```bash
uv run src/mlops_project/train.py model=custom_efficientnet
```

### Custom Training Configuration

```yaml
# configs/training/aggressive.yaml
max_epochs: 100
learning_rate: 0.0001
weight_decay: 0.01
optimizer: adamw

# Learning rate scheduling
lr_scheduler: cosine
warmup_epochs: 5
min_lr: 0.00001

# Regularization
dropout_rate: 0.5
label_smoothing: 0.1

# Early stopping
early_stopping_patience: 20
monitor_metric: val_f1_score
mode: max

# Optimization
gradient_clip_val: 1.0
accumulate_grad_batches: 1
precision: 16
```

### Custom Data Configuration

```yaml
# configs/data/augmented.yaml
data_dir: "data/ham10000"
batch_size: 32
image_size: 300
num_workers: 8

# Advanced augmentation
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotation_degrees: 45
  color_jitter:
    brightness: 0.3
    contrast: 0.3
    saturation: 0.3
    hue: 0.1
  random_resized_crop:
    scale: [0.8, 1.0]
    ratio: [0.8, 1.2]
  gaussian_blur:
    sigma: [0.1, 2.0]
    probability: 0.3

# Class balancing
use_weighted_sampling: true
oversample_minority: true

# Preprocessing
normalize: true
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```

## Advanced Configuration Patterns

### Hierarchical Configuration

```yaml
# configs/experiment/medical_ai.yaml
defaults:
  - model: efficientnet
  - training: medical_optimized
  - data: clinical_augmented
  - callbacks: medical_callbacks
  - logger: wandb

# Override specific settings
model:
  variant: b3
  pretrained: true
  
training:
  max_epochs: 50
  early_stopping_patience: 15
  
data:
  image_size: 300
  batch_size: 24

# Experiment metadata
experiment:
  name: "medical_ai_efficientnet_b3"
  description: "Production-ready medical AI model"
  tags: ["medical", "production", "efficientnet"]
  notes: "Optimized for clinical deployment"
```

### Multi-Environment Configuration

#### Development Environment
```yaml
# configs/env/development.yaml
defaults:
  - model: baseline_cnn
  - training: quick
  - data: subset

data:
  subsample_percentage: 0.01
  num_workers: 2

training:
  max_epochs: 3
  fast_dev_run: false
  limit_train_batches: 50
  
wandb:
  enabled: false
  mode: disabled

# Resource constraints
hardware:
  accelerator: cpu
  devices: 1
  precision: 32
```

#### Production Environment
```yaml
# configs/env/production.yaml
defaults:
  - model: efficientnet
  - training: optimized
  - data: full

model:
  variant: b4
  pretrained: true

training:
  max_epochs: 75
  early_stopping_patience: 20
  precision: 16
  
wandb:
  enabled: true
  project: "production-models"
  
# High-performance settings
hardware:
  accelerator: gpu
  devices: 2
  strategy: ddp
```

### Conditional Configuration

```yaml
# configs/config.yaml with conditionals
defaults:
  - model: efficientnet
  - training: standard
  - data: standard
  - _self_

# Conditional overrides based on model choice
model:
  _target_: ${oc.select:model.name,efficientnet}
  
# Dynamic image size based on model variant
data:
  image_size: ${model.input_sizes.${model.variant}}
  
# Adaptive batch size based on available memory
training:
  batch_size: ${oc.select:hardware.gpu_memory,"16GB":32,"24GB":48,"12GB":16,16}

# Environment-specific settings
wandb:
  enabled: ${oc.select:env.type,"production":true,"development":false,false}
```

## Configuration Composition

### Multi-Run Configurations

#### Hyperparameter Sweeps
```bash
# Multi-run with parameter sweeps
uv run src/mlops_project/train.py -m \
    model.variant=b0,b1,b2 \
    training.learning_rate=0.01,0.001,0.0001 \
    data.batch_size=16,32,64
```

#### Model Comparison Study
```bash
# Compare different architectures
uv run src/mlops_project/train.py -m \
    model=baseline_cnn,resnet,efficientnet \
    training.max_epochs=25 \
    wandb.tags=["model_comparison"]
```

### Configuration Interpolation

```yaml
# configs/model/interpolated.yaml
name: efficientnet
variant: b2

# Interpolate values
input_size: ${model.input_sizes.${model.variant}}
dropout_rate: ${oc.select:model.variant,"b0":0.2,"b1":0.2,"b2":0.3,"b3":0.3,0.4}

# Dynamic learning rate based on batch size
learning_rate: ${oc.select:data.batch_size,"16":0.0002,"32":0.0001,"64":0.00005,0.0001}

# Compute effective batch size
effective_batch_size: ${oc.eval:"${data.batch_size} * ${training.accumulate_grad_batches}"}
```

### Config Validation

```yaml
# configs/validation/schema.yaml
model:
  name: 
    _target_: str
    _choices: ["baseline_cnn", "resnet", "efficientnet"]
  variant:
    _target_: str
    _choices: ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    _default: "b0"
    
training:
  max_epochs:
    _target_: int
    _min: 1
    _max: 1000
  learning_rate:
    _target_: float
    _min: 0.000001
    _max: 1.0
    
data:
  batch_size:
    _target_: int
    _min: 1
    _max: 512
  image_size:
    _target_: int
    _choices: [224, 240, 260, 300, 380, 456, 528, 600]
```

## Custom Callbacks Configuration

### Advanced Callback Setup

```yaml
# configs/callbacks/medical_callbacks.yaml
model_checkpoint:
  _target_: pytorch_lightning.callbacks.ModelCheckpoint
  dirpath: "models/${model.name}/"
  filename: "${model.name}-{epoch:03d}-{val_f1_score:.3f}"
  monitor: "val_f1_score"
  mode: "max"
  save_top_k: 3
  save_last: true
  every_n_epochs: 1

early_stopping:
  _target_: pytorch_lightning.callbacks.EarlyStopping
  monitor: "val_f1_score"
  patience: 15
  mode: "max"
  min_delta: 0.001
  verbose: true

learning_rate_monitor:
  _target_: pytorch_lightning.callbacks.LearningRateMonitor
  logging_interval: "epoch"
  log_momentum: true

# Custom medical AI callback
medical_validator:
  _target_: src.mlops_project.callbacks.MedicalAIValidator
  sensitivity_threshold: 0.95
  specificity_threshold: 0.85
  validate_every_n_epochs: 5
  
# Model profiling callback
model_profiler:
  _target_: pytorch_lightning.callbacks.DeviceStatsMonitor
  cpu_stats: true
  gpu_stats: true

# Gradient monitoring
gradient_monitor:
  _target_: src.mlops_project.callbacks.GradientMonitor
  log_frequency: 100
  plot_gradients: true
```

## Environment-Specific Configurations

### Docker Configuration

```yaml
# configs/deploy/docker.yaml
model:
  export_format: "onnx"
  optimization: "speed"  # or "size", "accuracy"
  
serving:
  framework: "onnxruntime"
  batch_size: 1
  timeout: 30
  workers: 4
  
resources:
  cpu_cores: 4
  memory_gb: 8
  gpu_memory_gb: 0  # CPU-only deployment
```

### Cloud Training Configuration

```yaml
# configs/cloud/gcp.yaml
training:
  accelerator: "gpu"
  devices: 4
  strategy: "ddp"
  precision: 16
  
cloud:
  instance_type: "n1-highmem-8"
  gpu_type: "nvidia-tesla-v100"
  gpu_count: 4
  disk_size_gb: 100
  
storage:
  bucket_name: "mlops-skin-lesion-models"
  model_path: "gs://mlops-skin-lesion-models/checkpoints/"
  data_path: "gs://mlops-skin-lesion-data/ham10000/"
  
monitoring:
  log_to_cloud: true
  metrics_frequency: 10
```

## Configuration Best Practices

### 1. Modular Configuration Structure

```
configs/
├── config.yaml              # Main configuration
├── model/
│   ├── baseline_cnn.yaml
│   ├── resnet.yaml
│   └── efficientnet.yaml
├── training/
│   ├── quick.yaml
│   ├── standard.yaml
│   └── production.yaml
├── data/
│   ├── minimal.yaml
│   ├── augmented.yaml
│   └── clinical.yaml
├── experiment/
│   ├── development.yaml
│   ├── research.yaml
│   └── production.yaml
└── deploy/
    ├── local.yaml
    ├── docker.yaml
    └── cloud.yaml
```

### 2. Configuration Naming Conventions

```bash
# Use descriptive, hierarchical names
uv run src/mlops_project/train.py \
    experiment=medical_ai/efficientnet_b3/production \
    model.name=efficientnet \
    model.variant=b3 \
    +experiment.version="v1.2.3"
```

### 3. Configuration Documentation

```yaml
# configs/model/efficientnet.yaml
# EfficientNet model configuration for skin lesion classification
# 
# Variants available:
#   - b0: Fastest, lowest accuracy (5.3M params)
#   - b1: Balanced speed/accuracy (7.8M params)
#   - b2: Good for most use cases (9.1M params)
#   - b3: Production recommended (12M params)
#   - b4+: Maximum accuracy, requires high-end GPU

name: efficientnet
variant: b2  # Change to b0-b7 based on requirements

# Model architecture
num_classes: 2  # Binary classification (malignant/benign)
pretrained: true  # Use ImageNet weights (recommended)
dropout_rate: 0.3  # Regularization strength

# Training parameters
learning_rate: 0.0001  # Conservative for transfer learning
weight_decay: 0.01     # L2 regularization

# Model-specific settings
freeze_backbone_epochs: 0  # Set to >0 for gradual unfreezing
use_exponential_moving_average: false  # EMA for stability
```

---

*Continue to [API Usage Examples](api-usage.md) for practical inference patterns.*