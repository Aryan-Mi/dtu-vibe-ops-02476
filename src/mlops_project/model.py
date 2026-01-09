from torch import nn
import torch
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    """Building blocks for the baseline CNN model."""
    def __init__(self, in_ch: int, out_ch: int, use_bn : bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))

        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BaselineCNN(pl.LightningModule):
    """Model 1: A simple baseline CNN model."""
    def __init__(self, input_dim : int = 456, input_channel : int = 3, output_channels : list = [8, 16, 32], lr: float = 1e-3, num_classes = 2, use_bn : bool = True):
        super().__init__()
        
        # 1. Stack Conv Blocks
        layers = []
        cin = input_channel
        for cout in output_channels:
            layers.append(ConvBlock(cin, cout, use_bn))
            cin = cout

        # 2. Final Classifier
        layers.append(nn.Flatten())
        layers.append(nn.Linear(output_channels[-1] * (input_dim//(2**len(output_channels))) * (input_dim//(2**len(output_channels))), output_channels[-1])) 
        layers.append(nn.ReLU())
        layers.append(nn.Linear(output_channels[-1], num_classes)) 

        self.model = nn.Sequential(*layers)

        # 3. Loss function
        self.criterion = nn.CrossEntropyLoss()

        # 4. Learning rate
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) # Returns the logits. 

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)

        loss = self.criterion(logits, target)
        pred = logits.argmax(dim=-1)
        acc = (target == pred).float().mean()

        # Logging to wandb
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss
    
    def validation_step(self, batch) -> None:
        data, target = batch
        logits = self(data)

        loss = self.criterion(logits, target)
        pred = logits.argmax(dim=-1)
        acc = (target == pred).float().mean()

        # on_epoch=True to log epoch-level metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

if __name__ == "__main__":
    model = BaselineCNN()
    print(model)