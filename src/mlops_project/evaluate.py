import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch

from mlops_project.model import BaselineCNN, EfficientNet, ResNet

from .dataloader import create_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(
    model_name: str,
    model_checkpoint: str,
    model_config: dict,
    data_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.525,
    val_ratio: float = 0.175,
    test_ratio: float = 0.30,
    random_seed: int = 42,
) -> None:
    """Evaluate a trained model on the test set and print accuracy."""
    try:
        default_config = {
            "BaselineCNN": {
                "num_classes": 2,
                "input_dim": 224,
                "input_channel": 3,
                "output_channels": [8, 16, 32],
                "lr": 1e-3,
                "use_bn": True,
            },
            "ResNet": {
                "num_classes": 2,
                "input_channel": 3,
                "base_channel": 32,
                "output_channels": [8, 16, 32],
                "strides": [1, 2, 2],
                "dropout": 0.1,
                "lr": 1e-3,
                "use_bn": True,
            },
            "EfficientNet": {
                "num_classes": 2,
                "model_size": "b0",
                "lr": 1e-3,
                "use_bn": True,
                "pretrained": True,
                "freeze_backbone": False,
            },
        }

        config = default_config.get(model_name, {}).copy()
        if model_config:
            config.update(model_config)

        if model_name == "EfficientNet":
            model = EfficientNet(**config)
        elif model_name == "BaselineCNN":
            model = BaselineCNN(**config)
        elif model_name == "ResNet":
            model = ResNet(**config)
        else:
            msg = f"Unknown model: {model_name}"
            raise ValueError(msg)

        model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

        _, _, test_dataloader = create_dataloaders(
            data_path=data_path,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger = WandbLogger(project="mlops-project", log_model=False)

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        predictions = trainer.predict(model, test_dataloader)

        preds = torch.cat(predictions)
        targets = []
        for batch in test_dataloader:
            targets.append(batch[1])
        targets = torch.cat(targets)

        acc = (preds.cpu() == targets).float().mean().item()
        logger.log_metrics({"test_accuracy": acc})
        print(f"Test accuracy: {acc}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Out of memory...")
        else:
            print(f"FAILED. {str(e)}")

    except (ValueError, TypeError, AttributeError) as e:
        print(f"FAILED. {str(e)}")


if __name__ == "__main__":
    evaluate(
        model_name="BaselineCNN",
        model_checkpoint="models/baseline_cnn_checkpoint.pth",
        model_config={},
        data_path="data/processed/ham10000",
        image_size=224,
        batch_size=32,
        num_workers=4,
        train_ratio=0.525,
        val_ratio=0.175,
        test_ratio=0.30,
        random_seed=42,
    )
