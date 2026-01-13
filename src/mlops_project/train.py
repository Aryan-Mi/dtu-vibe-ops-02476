import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from mlops_project.dataloader import create_dataloaders, set_seed, subsample_dataloader
from mlops_project.model import BaselineCNN, EfficientNet, ResNet
from mlops_project.subsample import subsample_dataset

app = typer.Typer()


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
        print("\n[1/5] Loading full dataset...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
        )

    # Step 2: Define model to train
    print(f"\n[2/5] Initializing {model_name}...")

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
    if model_name in ["ResNet", "EfficientNet"]:
        print(f"    Pretrained: {pretrained}")
        print(f"    Freeze backbone: {freeze_backbone}")
    if model_name == "EfficientNet":
        print(f"    Model size: {efficientnet_size}")

    # Step 3: Train all models
    print("\n[3/5] Training models...")
    try:
        trained_model, training_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            epochs=max_epochs,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"  ERROR: Failed to train {model_name}: {str(e)}")
        return

    # Step 4: Evaluate all models on test set
    print("\n[4/5] Evaluating models on test set...")
    try:
        evaluation_results = evaluate(model=trained_model, test_loader=test_loader, model_name=model_name)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: Failed to evaluate {model_name}: {str(e)}")
        return

    # Step 5: Select best model and save results
    print("\n[5/5] Selecting best model...")

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Validation Loss: {training_metrics['best_val_loss']:.4f}")
    print(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
    print("=" * 80)

    # Save results summary
    results_path = Path(output_dir) / "training_results.json"
    results = {
        "model": model_name,
        "test_accuracy": evaluation_results["test_accuracy"],
        "val_loss": training_metrics["best_val_loss"],
        "checkpoint_path": training_metrics["checkpoint_path"],
        "configuration": {
            "data_path": data_path,
            "subsample_percentage": subsample_percentage,
            "image_size": image_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "efficientnet_size": efficientnet_size if model_name == "EfficientNet" else None,
            "pretrained": pretrained if model_name in ["ResNet", "EfficientNet"] else None,
            "freeze_backbone": freeze_backbone if model_name in ["ResNet", "EfficientNet"] else None,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Model checkpoint: {training_metrics['checkpoint_path']}")


if __name__ == "__main__":
    # Example usage:
    # Train BaselineCNN:
    # python train.py --model-name BaselineCNN --data-path data/raw/ham10000 --max-epochs 20
    #
    # Train ResNet with pretrained weights:
    # python train.py --model-name ResNet --pretrained --max-epochs 20
    #
    # Train EfficientNet B3 with frozen backbone:
    # python train.py --model-name EfficientNet --efficientnet-size b3 --freeze-backbone --max-epochs 15
    #
    # With subsampling (30% of data):
    # python train.py --model-name ResNet --subsample-percentage 0.3 --max-epochs 10
    #
    # *BEST FOR QUICK TESTING* (1% data, 5 epochs):
    # python train.py --model-name BaselineCNN --subsample-percentage 0.01 --max-epochs 5 --batch-size 16
    app()
