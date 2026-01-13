import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

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

    # train model
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[early_stopping, checkpoint_callback],
    )

    print(f"Training {model_name}...")
    trainer.fit(model, train_loader, val_loader)

    metrics = {
        "model_name": model_name,
        "best_val_loss": float(checkpoint_callback.best_model_score),
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


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train specified model on skin lesion dataset with Hydra configuration.

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
    data_path = cfg.data.data_path
    output_dir = cfg.paths.output_dir
    subsample_percentage = cfg.data.subsample_percentage

    # Step 1: Load data (with optional subsampling)
    if subsample_percentage is not None:
        print(f"\n[1/4] Subsampling dataset ({subsample_percentage * 100:.1f}%)...")
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
        train_loader, val_loader, test_loader = subsample_dataloader(
            data_path=data_path,
            subsample_result=subsample_result,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            random_seed=cfg.seed,
        )
    else:
        print("\n[1/4] Loading full dataset...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            image_size=cfg.data.image_size,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            random_seed=cfg.seed,
        )

    # Step 2: Define model to train
    print(f"\n[2/4] Initializing {model_name}...")
    model = create_model(cfg)

    print(f"  Model: {model_name}")
    if model_name == "ResNet":
        print(f"    Base channel: {cfg.model.base_channel}")
        print(f"    Output channels: {list(cfg.model.output_channels)}")
    if model_name == "EfficientNet":
        print(f"    Model size: {cfg.model.model_size}")
        print(f"    Pretrained: {cfg.model.pretrained}")
        print(f"    Freeze backbone: {cfg.model.freeze_backbone}")

    # Step 3: Train the model
    print("\n[3/4] Training model...")
    try:
        _, training_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
            output_dir=output_dir,
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


if __name__ == "__main__":
    # Hydra Usage Examples:
    #
    # Train with default config (BaselineCNN):
    #   python -m mlops_project.train
    #
    # Train ResNet (override model config group):
    #   python -m mlops_project.train model=resnet
    #
    # Train EfficientNet:
    #   python -m mlops_project.train model=efficientnet
    #
    # Override specific parameters:
    #   python -m mlops_project.train model=resnet training.max_epochs=10 data.batch_size=16
    #
    # Use subsampling (10% of data):
    #   python -m mlops_project.train data.subsample_percentage=0.1
    #
    # Quick test (1% data, 5 epochs):
    #   python -m mlops_project.train data.subsample_percentage=0.01 training.max_epochs=5
    #
    # Change learning rate:
    #   python -m mlops_project.train model=efficientnet model.lr=0.0001
    #
    # EfficientNet with different model size:
    #   python -m mlops_project.train model=efficientnet model.model_size=b3 data.image_size=300
    train()
