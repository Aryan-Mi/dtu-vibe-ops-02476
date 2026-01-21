import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import wandb
from mlops_project.dataloader import create_dataloaders, set_seed, subsample_dataloader
from mlops_project.model import BaselineCNN, EfficientNet, ResNet
from mlops_project.subsample import subsample_dataset

if TYPE_CHECKING:
    from pytorch_lightning.loggers.logger import Logger

try:
    from google.cloud import storage
except ImportError:
    storage = None


def upload_to_gcs(local_path: str | Path, gcs_bucket: str, gcs_path: str) -> str | None:
    """Upload a file to GCS bucket.

    Args:
        local_path: Local file path to upload
        gcs_bucket: GCS bucket name (without gs:// prefix)
        gcs_path: Destination path in GCS bucket

    Returns:
        GCS URI (gs://bucket/path) if successful, None otherwise
    """
    if storage is None:
        print("  ⚠ google-cloud-storage not available, skipping GCS upload")
        return None

    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{gcs_bucket}/{gcs_path}"
        print(f"  ✓ Uploaded to {gcs_uri}")
        return gcs_uri
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ Failed to upload to GCS: {str(e)}")
        return None


def train_model(
    model: pl.LightningModule,
    model_name: str,
    train_loader,  # training dataloader
    val_loader,  # validation dataloader
    epochs: int = 10,
    patience: int = 7,
    output_dir: str = "models",
    wandb_logger: WandbLogger | None = None,
) -> tuple[pl.LightningModule, dict]:
    """Train a given model and return the trained model and training metrics."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # output path for logging and checkpoint purposes
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # stop training model when it stops improving
    early_stopping = EarlyStopping("val_loss", patience=patience)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Set up loggers
    loggers: list[Logger] = [CSVLogger(save_dir="logs/")]
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=[early_stopping, checkpoint_callback],
    )

    print(f"Training {model_name}...")
    trainer.fit(model, train_loader, val_loader)

    metrics = {
        "model_name": model_name,
        "best_val_loss": checkpoint_callback.best_model_score.item()
        if checkpoint_callback.best_model_score is not None
        else None,
        "checkpoint_path": str(checkpoint_callback.best_model_path),
    }

    return model, metrics


def create_model(cfg: DictConfig) -> pl.LightningModule:
    """Create a model based on the configuration.

    Args:
        cfg: Hydra configuration containing model parameters

    Returns:
        Instantiated PyTorch Lightning model
    """
    model_name = cfg.model.name
    model_cfg = cfg.model

    if model_name == "BaselineCNN":
        return BaselineCNN(
            num_classes=model_cfg.num_classes,
            input_dim=model_cfg.input_dim,
            input_channel=model_cfg.input_channel,
            output_channels=list(model_cfg.output_channels),
            lr=model_cfg.lr,
            use_bn=model_cfg.use_bn,
        )
    if model_name == "ResNet":
        return ResNet(
            num_classes=model_cfg.num_classes,
            input_channel=model_cfg.input_channel,
            input_dim=model_cfg.input_dim,
            base_channel=model_cfg.base_channel,
            output_channels=list(model_cfg.output_channels),
            strides=list(model_cfg.strides),
            dropout=model_cfg.dropout,
            lr=model_cfg.lr,
            use_bn=model_cfg.use_bn,
        )
    if model_name == "EfficientNet":
        return EfficientNet(
            num_classes=model_cfg.num_classes,
            model_size=model_cfg.model_size,
            lr=model_cfg.lr,
            use_bn=model_cfg.use_bn,
            pretrained=model_cfg.pretrained,
            freeze_backbone=model_cfg.freeze_backbone,
        )

    msg = f"Unknown model '{model_name}'. Choose from: BaselineCNN, ResNet, EfficientNet"
    raise ValueError(msg)


def setup_wandb_logger(cfg: DictConfig, model_name: str, subsample_percentage: float | None) -> WandbLogger | None:
    """Initialize W&B logger if enabled in configuration.

    Args:
        cfg: Hydra configuration object
        model_name: Name of the model being trained
        subsample_percentage: Percentage of data used for subsampling, or None

    Returns:
        WandbLogger instance if enabled, None otherwise
    """
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    if not use_wandb:
        print("\n[3/5] W&B logging disabled")
        return None

    print("\n[3/5] Initializing W&B logging...")
    wandb_cfg = cfg.wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    run_name = wandb_cfg.get("run_name") or f"{model_name}-{subsample_percentage or 'full'}"
    wandb_logger = WandbLogger(
        project=wandb_cfg.get("project", "skin-lesion-classification"),
        name=run_name,
        config=wandb_config,
        log_model=wandb_cfg.get("log_model", True),
    )
    print(f"  W&B logging enabled: {wandb_cfg.get('project')}/{run_name}")
    return wandb_logger


def export_model_to_onnx(model: pl.LightningModule, checkpoint_path: Path, image_size: int) -> Path:
    """Export trained model to ONNX format.

    Args:
        model: Trained PyTorch Lightning model
        checkpoint_path: Path to the model checkpoint
        image_size: Size of input images

    Returns:
        Path to the exported ONNX model
    """
    onnx_model_path = checkpoint_path.with_suffix(".onnx")
    dummy = torch.randn(1, 3, image_size, image_size, device="cpu")

    model.eval()
    model.to("cpu")
    model.to_onnx(
        file_path=str(onnx_model_path),
        input_sample=dummy,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        export_params=True,
        opset_version=17,
    )
    return onnx_model_path


def upload_models_to_gcs(
    gcs_bucket: str,
    model_name: str,
    checkpoint_path: Path,
    onnx_model_path: Path,
    results_path: Path,
) -> tuple[str | None, str | None, str | None]:
    """Upload models and results to GCS bucket.

    Args:
        gcs_bucket: GCS bucket name (without gs:// prefix)
        model_name: Name of the model
        checkpoint_path: Path to checkpoint file
        onnx_model_path: Path to ONNX model file
        results_path: Path to results JSON file

    Returns:
        Tuple of (checkpoint_gcs_path, onnx_gcs_path, results_gcs_path)
    """
    # Remove gs:// prefix if present
    gcs_bucket = gcs_bucket.replace("gs://", "")
    print(f"\nUploading models to GCS bucket: {gcs_bucket}")

    # Upload checkpoint
    checkpoint_gcs_key = f"models/{model_name}/{checkpoint_path.name}"
    checkpoint_gcs_path = upload_to_gcs(checkpoint_path, gcs_bucket, checkpoint_gcs_key)

    # Upload ONNX model
    onnx_gcs_key = f"models/{model_name}/{onnx_model_path.name}"
    onnx_gcs_path = upload_to_gcs(onnx_model_path, gcs_bucket, onnx_gcs_key)

    # Upload training results JSON
    results_gcs_key = f"models/{model_name}/training_results.json"
    results_gcs_path = upload_to_gcs(results_path, gcs_bucket, results_gcs_key)

    return checkpoint_gcs_path, onnx_gcs_path, results_gcs_path


def save_training_results(
    cfg: DictConfig,
    model_name: str,
    training_metrics: dict,
    checkpoint_path: Path,
    onnx_model_path: Path,
    checkpoint_gcs_path: str | None,
    onnx_gcs_path: str | None,
    results_gcs_path: str | None,
    output_dir: str,
) -> Path:
    """Save training results to JSON file.

    Args:
        cfg: Hydra configuration object
        model_name: Name of the trained model
        training_metrics: Dictionary containing training metrics
        checkpoint_path: Path to model checkpoint
        onnx_model_path: Path to ONNX model
        checkpoint_gcs_path: GCS path to checkpoint (if uploaded)
        onnx_gcs_path: GCS path to ONNX model (if uploaded)
        results_gcs_path: GCS path to results JSON (if uploaded)
        output_dir: Output directory for results

    Returns:
        Path to the saved results JSON file
    """
    results_path = Path(output_dir) / "training_results.json"
    results = {
        "model": model_name,
        "val_loss": training_metrics["best_val_loss"],
        "checkpoint_path": training_metrics["checkpoint_path"],
        "checkpoint_gcs_path": checkpoint_gcs_path,
        "onnx_model_path": str(onnx_model_path),
        "onnx_gcs_path": onnx_gcs_path,
        "results_gcs_path": results_gcs_path,
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results_path


def load_data(cfg: DictConfig) -> tuple:
    """Load training data with optional subsampling.

    Args:
        cfg: Hydra configuration object

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_path = cfg.data.data_path
    subsample_percentage = cfg.data.subsample_percentage

    if subsample_percentage is not None:
        print(f"\n[1/5] Subsampling dataset ({subsample_percentage * 100:.1f}%)...")
        metadata_path = Path(data_path) / "metadata" / "HAM10000_metadata.csv"

        subsample_result = subsample_dataset(
            metadata_path=metadata_path,
            percentage=subsample_percentage,
            random_seed=cfg.data.subsample_seed,
        )

        # Print subsample statistics
        total_subsampled = sum(len(images) for images in subsample_result.values())
        print(f"  Subsampled {total_subsampled} images")
        for dx_category, images in subsample_result.items():
            print(f"    {dx_category}: {len(images)} images")

        # Create dataloaders from subsampled data
        return subsample_dataloader(
            data_path=data_path,
            subsample_result=subsample_result,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            random_seed=cfg.seed,
        )

    print("\n[1/5] Loading full dataset...")
    return create_dataloaders(
        data_path=data_path,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        random_seed=cfg.seed,
    )


def print_model_info(model_name: str, cfg: DictConfig) -> None:
    """Print model configuration information.

    Args:
        model_name: Name of the model
        cfg: Hydra configuration object
    """
    print(f"  Model: {model_name}")
    if model_name == "ResNet":
        print(f"    Base channel: {cfg.model.base_channel}")
        print(f"    Output channels: {list(cfg.model.output_channels)}")
    if model_name == "EfficientNet":
        print(f"    Model size: {cfg.model.model_size}")
        print(f"    Pretrained: {cfg.model.pretrained}")
        print(f"    Freeze backbone: {cfg.model.freeze_backbone}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train specified model on skin lesion dataset with Hydra configuration and W&B logging.

    Args:
        cfg: Hydra configuration object
    """
    # Print resolved configuration
    print("=" * 80)
    print("SKIN LESION CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    print("\nResolved Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    model_name = cfg.model.name
    output_dir = cfg.paths.output_dir
    subsample_percentage = cfg.data.subsample_percentage
    use_wandb = cfg.get("wandb", {}).get("enabled", False)

    train_loader, val_loader, test_loader = load_data(cfg)

    print(f"\n[2/5] Initializing {model_name}...")
    model = create_model(cfg)
    print_model_info(model_name, cfg)

    wandb_logger = setup_wandb_logger(cfg, model_name, subsample_percentage)

    print("\n[4/5] Training model...")
    try:
        _, training_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
            output_dir=output_dir,
            wandb_logger=wandb_logger,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: Failed to train {model_name}: {str(e)}")
        if use_wandb:
            wandb.finish()
        return

    print("\n[5/5] Saving results...")
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Validation Loss: {training_metrics['best_val_loss']:.4f}")
    print("=" * 80)

    checkpoint_path = Path(training_metrics["checkpoint_path"])
    image_size = int(cfg.data.image_size)

    # Export the trained model to ONNX format
    onnx_model_path = export_model_to_onnx(model, checkpoint_path, image_size)

    # Save training results (models will be tracked with DVC after training)
    results_path = save_training_results(
        cfg=cfg,
        model_name=model_name,
        training_metrics=training_metrics,
        checkpoint_path=checkpoint_path,
        onnx_model_path=onnx_model_path,
        checkpoint_gcs_path=None,
        onnx_gcs_path=None,
        results_gcs_path=None,
        output_dir=output_dir,
    )

    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Model checkpoint: {training_metrics['checkpoint_path']}")
    print(f"✓ ONNX model exported to: {onnx_model_path}")
    print("\n📦 Models saved locally. They will be tracked with DVC and pushed to GCS by the entrypoint script.")

    # Finish W&B run
    if use_wandb:
        wandb.finish()
        print("\nW&B run finished. View results at: https://wandb.ai")


if __name__ == "__main__":
    # Hydra Usage Examples:
    #
    # Train with default config (BaselineCNN):
    #   uv run src/mlops_project/train.py
    #
    # Train ResNet (override model config group):
    #   uv run src/mlops_project/train.py model=resnet
    #
    # Train EfficientNet:
    #   uv run src/mlops_project/train.py model=efficientnet
    #
    # Override specific parameters:
    #   uv run src/mlops_project/train.py model=resnet training.max_epochs=10 data.batch_size=16
    #
    # Use subsampling (10% of data):
    #   uv run src/mlops_project/train.py data.subsample_percentage=0.1
    #
    # Quick test (1% data, 5 epochs):
    #   uv run src/mlops_project/train.py data.subsample_percentage=0.01 training.max_epochs=5
    #
    # Change learning rate:
    #   uv run src/mlops_project/train.py model=efficientnet model.lr=0.0001
    #
    # EfficientNet with different model size:
    #   uv run src/mlops_project/train.py model=efficientnet model.model_size=b3 data.image_size=300
    #
    # Enable W&B logging:
    #   uv run src/mlops_project/train.py wandb.enabled=true
    #
    # W&B with custom project and run name:
    #   uv run src/mlops_project/train.py wandb.enabled=true wandb.project=my-project wandb.run_name=experiment-1
    #
    # Full example with W&B:
    #   uv run src/mlops_project/train.py model=efficientnet data.subsample_percentage=0.1 wandb.enabled=true
    train()
