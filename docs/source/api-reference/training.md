# Training API Reference

Documentation for training orchestration, optimization, and experiment management.

## Training Module

::: src.mlops_project.train.train_model

## Evaluation Functions

::: src.mlops_project.evaluate.evaluate_model

::: src.mlops_project.evaluate.compute_metrics

## Training Configuration

### Hydra Configuration Structure

```python
@dataclass
class TrainingConfig:
    max_epochs: int = 25
    learning_rate: float = 0.0001
    batch_size: int = 32
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    precision: Union[int, str] = 32
    
@dataclass  
class ModelConfig:
    name: str = "efficientnet"
    variant: str = "b0"
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.3
    
@dataclass
class DataConfig:
    data_dir: str = "data/ham10000"
    batch_size: int = 32
    image_size: int = 224
    num_workers: int = 4
    subsample_percentage: float = 1.0
    augment: bool = True
```

## PyTorch Lightning Integration

### LightningModule Methods

The models inherit from `pytorch_lightning.LightningModule` and implement:

```python
class SkinLesionClassifier(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        """Training step implementation"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step implementation"""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, labels), on_epoch=True)
        return {'val_loss': loss, 'preds': preds, 'labels': labels}
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
```

### Callbacks

#### ModelCheckpoint

```python
checkpoint_callback = ModelCheckpoint(
    dirpath=f"models/{config.model.name}/",
    filename="{epoch:03d}-{val_loss:.3f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    verbose=True
)
```

#### EarlyStopping

```python
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=config.training.early_stopping_patience,
    mode="min",
    verbose=True
)
```

#### LearningRateMonitor

```python
lr_monitor = LearningRateMonitor(
    logging_interval="epoch",
    log_momentum=True
)
```

## Optimization Strategies

### Learning Rate Scheduling

| Scheduler | Description | Best For |
|-----------|-------------|----------|
| `cosine` | Cosine annealing | Long training runs |
| `step` | Step decay | General purpose |
| `reduce_on_plateau` | Adaptive reduction | Unstable training |
| `exponential` | Exponential decay | Fine-tuning |

### Advanced Techniques

#### Mixed Precision Training

```python
trainer = pl.Trainer(
    precision=16,           # or "bf16" for bfloat16
    amp_backend="native"    # PyTorch native AMP
)
```

#### Gradient Accumulation

```python
trainer = pl.Trainer(
    accumulate_grad_batches=4,  # Effective batch size = 4 × actual batch size
    gradient_clip_val=1.0       # Gradient clipping
)
```

#### Multi-GPU Training

```python
trainer = pl.Trainer(
    accelerator="gpu",
    devices=2,                  # Number of GPUs
    strategy="ddp"              # Distributed Data Parallel
)
```

## Metrics and Logging

### Tracked Metrics

| Metric | Description | Medical Relevance |
|--------|-------------|-------------------|
| **Accuracy** | Overall correctness | General performance |
| **Sensitivity** | True positive rate | Malignant detection rate |
| **Specificity** | True negative rate | Benign detection rate |
| **Precision** | Positive predictive value | Diagnostic confidence |
| **F1 Score** | Harmonic mean of precision/recall | Balanced performance |
| **AUC-ROC** | Area under ROC curve | Discrimination ability |
| **AUC-PR** | Area under PR curve | Performance on imbalanced data |

### Weights & Biases Integration

```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="skin-lesion-classification",
    name=f"{config.model.name}-{config.model.variant}",
    tags=[config.model.name, "production"]
)

trainer = pl.Trainer(logger=wandb_logger)
```

#### Logged Artifacts

- **Model Checkpoints**: Best and latest model weights
- **Configuration Files**: Hydra configs and hyperparameters
- **Training Plots**: Loss curves, learning rate schedules
- **Validation Images**: Sample predictions with confidence scores
- **Confusion Matrices**: Performance visualization
- **Model Architecture**: Network graph and parameter counts

## Training Monitoring

### Progress Tracking

```python
# Rich progress bar for enhanced terminal output
from pytorch_lightning.callbacks import RichProgressBar

trainer = pl.Trainer(
    callbacks=[RichProgressBar()],
    log_every_n_steps=10,
    val_check_interval=0.25    # Validate 4 times per epoch
)
```

### Model Profiling

```python
trainer = pl.Trainer(
    profiler="simple",          # or "advanced", "pytorch"
    max_epochs=1,              # Profile single epoch
    limit_train_batches=100    # Limit for profiling
)
```

## Error Handling

### Common Training Issues

#### CUDA Out of Memory

```python
# Automatic solutions in training script
try:
    trainer.fit(model, data_module)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size automatically
        config.data.batch_size //= 2
        data_module.batch_size = config.data.batch_size
        trainer.fit(model, data_module)
```

#### NaN Loss Detection

```python
# Terminate early on NaN loss
trainer = pl.Trainer(
    detect_anomaly=True,        # Enable anomaly detection
    terminate_on_nan=True       # Stop on NaN
)
```

### Debugging Mode

```python
# Fast debug configuration
trainer = pl.Trainer(
    fast_dev_run=True,          # Single batch through train/val/test
    overfit_batches=10,         # Overfit on small subset
    limit_train_batches=0.01    # Use 1% of training data
)
```

## Performance Benchmarks

### Training Speed Comparison

| Configuration | Images/sec | Memory (GB) | Time (EfficientNet-B3) |
|---------------|------------|-------------|------------------------|
| Baseline | 15 | 4 | 120 min |
| Mixed Precision | 28 | 3 | 65 min |
| + Multi-GPU (2x) | 52 | 6 | 35 min |
| + Optimized DataLoader | 64 | 8 | 28 min |

### Model Convergence

| Model | Epochs to Convergence | Best Val Accuracy | Training Time |
|-------|----------------------|-------------------|---------------|
| BaselineCNN | 15 | 74.2% | 12 min |
| ResNet-18 | 20 | 82.3% | 32 min |
| EfficientNet-B0 | 25 | 85.1% | 48 min |
| EfficientNet-B3 | 30 | 88.7% | 95 min |

*Benchmarks on NVIDIA V100 GPU with full HAM10000 dataset*