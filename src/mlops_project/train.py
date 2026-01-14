import json
from pathlib import Path

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


# Train models written in model.py and store results for comparison and selection of best model.
# pytorch lightning reference source: https://lightning.ai/pages/community/tutorial/step-by-step-walk-through-of-pytorch-lightning/
def train_model(
    model: pl.LightningModule,
    model_name: str,
    train_loader,  # training dataloader
    val_loader,  # validation dataloader
    epochs: int = 10,
    patience: int = 7,
    output_dir: str = "outputs/models",
    wandb_logger: WandbLogger | None = None,
) -> tuple[pl.LightningModule, dict]:
    """Train a given model and return the trained model and training metrics."""
    # Troubleshooting purposes - training slower on cpu
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # output path for logging and checkpoint purposes
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # stop training model when it stops improving
    early_stopping = EarlyStopping("val_loss", patience=patience)

    # model checkpoint for saving epoch results during training
    # For picking the best model and also troubleshooting :)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Set up loggers
    loggers = [CSVLogger(save_dir="logs/")]
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    # train model
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


def _get_dataloaders(
    data_path: str,
    subsample_percentage: float | None,
    subsample_seed: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    random_seed: int,
):
    if subsample_percentage is not None:
        print(f"\n[1/5] Subsampling dataset ({subsample_percentage * 100:.1f}%)...")
        metadata_path = Path(data_path) / "metadata" / "HAM10000_metadata.csv"

        subsample_result = subsample_dataset(
            metadata_path=metadata_path,
            percentage=subsample_percentage,
            random_seed=subsample_seed,
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
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
        )

    print("\n[1/4] Loading full dataset...")
    return create_dataloaders(
        data_path=data_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed,
    )


def _initialize_model(
    model_name: str,
    num_classes: int,
    learning_rate: float,
    efficientnet_size: str,
    pretrained: bool,
    freeze_backbone: bool,
) -> pl.LightningModule | None:
    if model_name == "BaselineCNN":
        return BaselineCNN(
            num_classes=num_classes,
            lr=learning_rate,
        )
    if model_name == "ResNet":
        return ResNet(num_classes=num_classes, lr=learning_rate)
    if model_name == "EfficientNet":
        return EfficientNet(
            num_classes=num_classes,
            model_size=efficientnet_size,
            lr=learning_rate,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    print(f"  ERROR: Unknown model '{model_name}'. Choose from: BaselineCNN, ResNet, EfficientNet")
    return None


# Creation of CLI interaction with train.py (vibecoded)
@app.command()
def train(
    model_name: str = typer.Option(..., help="Model to train:BaselineCNN, ResNet, or EfficientNet"),
    data_path: str = typer.Option("data/raw/ham10000", help="Path to the data folder"),
    output_dir: str = typer.Option("outputs/models", help="Directory to save trained models"),
    subsample_percentage: float | None = typer.Option(None, help="Percentage of data to subsample (0.0-1.0)"),
    subsample_seed: int = typer.Option(42, help="Random seed for subsampling"),
    image_size: int = typer.Option(224, help="Image size for training"),
    batch_size: int = typer.Option(32, help="Batch size"),
    num_workers: int = typer.Option(4, help="Number of data loading workers"),
    max_epochs: int = typer.Option(20, help="Maximum number of training epochs"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    num_classes: int = typer.Option(7, help="Number of classes (7 for HAM10000)"),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility"),
    # model-specific config options
    efficientnet_size: str = typer.Option("b0", help="EfficientNet model size (b0-b7)"),
    pretrained: bool = typer.Option(True, help="Use pretrained weights (for ResNet/EfficientNet)"),
    freeze_backbone: bool = typer.Option(False, help="Freeze backbone layers (for ResNet/EfficientNet)"),
) -> None:
    """Train specified model on skin lesion dataset with optional subsampling."""
    set_seed(random_seed)

    print("=" * 80)
    print("SKIN LESION CLASSIFICATION - MODEL TRAINING")
    print("=" * 80)
    print("\nResolved Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed for reproducibility
    set_seed(cfg.seed)

    model_name = cfg.model.name
    data_path = cfg.data.data_path
    output_dir = cfg.paths.output_dir
    subsample_percentage = cfg.data.subsample_percentage

    # Step 1: Load data (with optional subsampling)
    if subsample_percentage is not None:
        print(f"\n[1/5] Subsampling dataset ({subsample_percentage * 100:.1f}%)...")
        metadata_path = Path(data_path) / "metadata" / "HAM10000_metadata.csv"

        subsample_result = subsample_dataset(
            metadata_path=metadata_path,
            percentage=subsample_percentage,
            random_seed=subsample_seed,
        )

        # Print subsample statistics
        total_subsampled = sum(len(images) for images in subsample_result.values())
        print(f"  Subsampled {total_subsampled} images")
        for dx_category, images in subsample_result.items():
            print(f"    {dx_category}: {len(images)} images")

        # Create dataloaders from subsampled data
        train_loader, val_loader, test_loader = subsample_dataloader(
            data_path=data_path,
            subsample_result=subsample_result,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
        )
    else:
        print("\n[1/4] Loading full dataset...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
        )

    # Step 2: Define model to train
    print(f"\n[2/4] Initializing {model_name}...")

    # Validate and create the specified model
    if model_name == "BaselineCNN":
        model = BaselineCNN(
            num_classes=num_classes,
            lr=learning_rate,
        )
    elif model_name == "ResNet":
        model = ResNet(num_classes=num_classes, lr=learning_rate)
    elif model_name == "EfficientNet":
        model = EfficientNet(
            num_classes=num_classes,
            model_size=efficientnet_size,
            lr=learning_rate,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        print(f"  ERROR: Unknown model '{model_name}'. Choose from: BaselineCNN, ResNet, EfficientNet")
        return

    print(f"  Model: {model_name}")
    if model_name == "ResNet":
        print(f"    Base channel: {cfg.model.base_channel}")
        print(f"    Output channels: {list(cfg.model.output_channels)}")
    if model_name == "EfficientNet":
        print(f"    Model size: {cfg.model.model_size}")
        print(f"    Pretrained: {cfg.model.pretrained}")
        print(f"    Freeze backbone: {cfg.model.freeze_backbone}")

    # Step 3: Train all models
    print("\n[3/4] Training models...")
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
        return

    # Step 4: Save results
    print("\n[4/4] Saving results...")

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Validation Loss: {training_metrics['best_val_loss']:.4f}")
    print("=" * 80)

    # Save results summary
    results_path = Path(output_dir) / "training_results.json"
    results = {
        "model": model_name,
        "val_loss": training_metrics["best_val_loss"],
        "checkpoint_path": training_metrics["checkpoint_path"],
        "configuration": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Model checkpoint: {training_metrics['checkpoint_path']}")

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
    train()
