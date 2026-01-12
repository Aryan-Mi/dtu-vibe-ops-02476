import json
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
import typer

from torch.utils.data import DataLoader
from model import BaselineCNN, ResNet, EfficientNet
from subsample import subsample_dataset
from dataloader import create_dataloaders, set_seed
from data import CancerDataset, get_transforms

app = typer.Typer()


# Currently dataloader.py only deals with the entire dataset, making it difficult to subsample.
# Copying dataloader function and modifying for subsample purposes here...
# It may be more efficient long term to have the dataloader function take an input variable representing subsampling (with 0 or 1 meaning no subsampling).
def subsample_dataloader(
    data_path: str,
    subsample_result: dict,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.525,
    val_ratio: float = 0.175,
    test_ratio: float = 0.30,
    random_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    set_seed(random_seed)

    metadata_path = data_path + "/metadata" + "/HAM10000_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    #get list of image_ids
    subsampled_image_ids = []
    for category, images in subsample_result.items():
        subsampled_image_ids.extend([img["image_id"] for img in images])
    #retrieve subsample indices
    subsampled_indices = metadata[metadata["image_id"].isin(subsampled_image_ids)].index.tolist()
    subsampled_metadata = metadata.iloc[subsampled_indices]

    # Get unique lesion IDs
    unique_lesions = subsampled_metadata["lesion_id"].unique()

    # First split: separate test set
    train_val_lesions, test_lesions = train_test_split(unique_lesions, test_size=test_ratio, random_state=random_seed)

    # Second split: separate train and validation from remaining data
    # Calculate validation ratio relative to train+val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

    train_lesions, val_lesions = train_test_split(
        train_val_lesions, test_size=val_ratio_adjusted, random_state=random_seed
    )

    # Get indices for each split
    train_indices = metadata[metadata["lesion_id"].isin(train_lesions)].index.tolist()
    val_indices = metadata[metadata["lesion_id"].isin(val_lesions)].index.tolist()
    test_indices = metadata[metadata["lesion_id"].isin(test_lesions)].index.tolist()

    print("Dataset split:")
    print(
        f"  Train: {len(train_indices)} images ({len(train_lesions)} lesions) - {len(train_indices) / len(metadata) * 100:.1f}%"  # noqa: E501
    )
    print(
        f"  Val:   {len(val_indices)} images ({len(val_lesions)} lesions) - {len(val_indices) / len(metadata) * 100:.1f}%"  # noqa: E501
    )
    print(
        f"  Test:  {len(test_indices)} images ({len(test_lesions)} lesions) - {len(test_indices) / len(metadata) * 100:.1f}%"  # noqa: E501
    )
    # Create datasets with appropriate transforms
    train_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=True),
        split_indices=train_indices,
    )

    val_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=False),
        split_indices=val_indices,
    )

    test_dataset = CancerDataset(
        data_path=str(data_path),
        transform=get_transforms(image_size=image_size, augment=False),
        split_indices=test_indices,
    )

    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=True,  # Faster GPU transfer
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader




#Train models written in model.py and store results for comparison and selection of best model.
#pytorch lightning reference source: https://lightning.ai/pages/community/tutorial/step-by-step-walk-through-of-pytorch-lightning/
def train_model(
    model: pl.LightningModule,
    model_name: str,
    train_loader, #training dataloader
    val_loader, #validation dataloader
    epochs: int = 10,
    output_dir: str = "outputs/models"
)-> tuple[pl.LightningModule, dict]:
    
    #Troubleshooting purposes - training slower on cpu
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    #output path for logging and checkpoint purposes
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    #stop training model when it stops improving
    early_stopping = EarlyStopping('val_loss', patience=7)

    #model checkpoint for saving epoch results during training
    #For picking the best model and also troubleshooting :)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    #train model
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger= CSVLogger(save_dir="logs/"),
		callbacks= [early_stopping, checkpoint_callback]
        )
    
    print(f"Training {model_name}...")
    trainer.fit(model,train_loader,val_loader)
    
    metrics = {
        "model_name": model_name,
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "checkpoint_path": str(checkpoint_callback.best_model_path),
    }
    
    return model, metrics
    



#Evaluate and compare training results
def evaluate(
    model: pl.LightningModule, 
    test_loader,
    model_name: str
)->dict:
    print(f"\nEvaluating {model_name} on test set...")
    
    #train
    trainer = pl.Trainer(
        accelerator="auto", 
        devices=1, 
        logger=False
        )
    
    #predict
    predictions = trainer.predict(model, test_loader)

    #retrieve label predictions
    all_preds = torch.cat(predictions)
    all_labels = []
    for batch in test_loader:
        _, labels = batch
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)

    #Calculate accuracy
    accuracy = (all_preds == all_labels).float().mean().item()

    metrics = {
        "model_name": model_name,
        "test_accuracy": accuracy,
    }

    print(f"\tTest Accuracy: {accuracy:.4f}")

    return metrics




# Creation of CLI interaction with train.py (vibecoded)
@app.command()
def train(
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
) -> None:
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

    # Step 2: Define models to train
    print("\n[2/5] Initializing models...")

    models_to_train = {
        "BaselineCNN": BaselineCNN(
            num_classes=num_classes,
            lr=learning_rate,
        ),
        "ResNet": ResNet(
            num_classes=num_classes,
            lr=learning_rate,
        ),
        "EfficientNet_B0": EfficientNet(
            num_classes=num_classes,
            model_size="b0",
            lr=learning_rate,
            pretrained=True,
            freeze_backbone=False,
        ),
    }

    print(f"  Models to train: {', '.join(models_to_train.keys())}")

    # Step 3: Train all models
    print("\n[3/5] Training models...")
    trained_models = {}
    training_metrics = {}

    for model_name, model in models_to_train.items():
        try:
            trained_model, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                model_name=model_name,
                epochs=max_epochs,
                output_dir=output_dir,
            )
            trained_models[model_name] = trained_model
            training_metrics[model_name] = metrics
        except Exception as e:
            print(f"  ERROR: Failed to train {model_name}: {str(e)}")
            continue

    # Step 4: Evaluate all models on test set
    print("\n[4/5] Evaluating models on test set...")
    evaluation_results = {}

    for model_name, model in trained_models.items():
        try:
            eval_metrics = evaluate(model=model, test_loader=test_loader, model_name=model_name)
            evaluation_results[model_name] = eval_metrics
        except Exception as e:
            print(f"  ERROR: Failed to evaluate {model_name}: {str(e)}")
            continue

    # Step 5: Select best model and save results
    print("\n[5/5] Selecting best model...")

    if not evaluation_results:
        print("  ERROR: No models were successfully trained and evaluated!")
        return

    # Find best model based on test accuracy
    best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]["test_accuracy"])
    best_accuracy = evaluation_results[best_model_name]["test_accuracy"]

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print("\nModel Comparison:")
    print(f"{'Model':<20} {'Val Loss':<12} {'Test Accuracy':<15}")
    print("-" * 50)

    for model_name in trained_models.keys():
        val_loss = training_metrics[model_name]["best_val_loss"]
        test_acc = evaluation_results[model_name]["test_accuracy"]
        marker = " ← BEST" if model_name == best_model_name else ""
        print(f"{model_name:<20} {val_loss:<12.4f} {test_acc:<15.4f}{marker}")

    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.4f}")
    print("=" * 80)

    # Save results summary
    results_path = Path(output_dir) / "training_results.json"
    results = {
        "best_model": best_model_name,
        "best_test_accuracy": best_accuracy,
        "training_metrics": training_metrics,
        #"evaluation_results": {k: {**v, "per_class_accuracy": str(v["per_class_accuracy"])} for k, v in evaluation_results.items()},
        "configuration": {
            "data_path": data_path,
            "subsample_percentage": subsample_percentage,
            "image_size": image_size,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Best model checkpoint: {training_metrics[best_model_name]['checkpoint_path']}")


if __name__ == "__main__":
    # Example usage:
    # python [path/to/train.py] --data-path data/raw/ham10000 --output-dir outputs/models --max-epochs 20
    #
    # With subsampling (30% of data):
    # python [path/to/train.py] --subsample-percentage 0.3 --max-epochs 10
    #
    # *BEST FOR QUICK TESTING* (1% data, 5 epochs):
    # python [path/to/train.py] --subsample-percentage 0.01 --max-epochs 5 --batch-size 16

    app()
