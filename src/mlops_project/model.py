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
    def __init__(self, num_classes : int = 2, input_dim : int = 456, input_channel : int = 3, output_channels : list = [8, 16, 32], lr: float = 1e-3, use_bn : bool = True):
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

    def training_step(self, batch: tuple, batch_idx):
        data, target = batch
        logits = self(data)

        loss = self.criterion(logits, target)
        pred = logits.argmax(dim=-1)
        acc = (target == pred).float().mean()

        # Logging to wandb
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss
    
    def validation_step(self, batch: tuple) -> None:
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
    
class ResidualBlock(nn.Module):
    """ResNet Residual Block. Credits to : https://d2l.ai/chapter_convolutional-modern/resnet.html"""
    def __init__(self, in_ch: int, out_ch: int, stride: int=1, use_bn: bool=True):
        super().__init__()
        # 1. Define helper
        def bn(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        # 2. Define branches / layers
        branch_layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)]
        branch_layers.append(bn(out_ch))
        branch_layers.append(nn.ReLU())
        branch_layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        branch_layers.append(bn(out_ch))
        self.branch = nn.Sequential(*branch_layers)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                bn(out_ch)
            )
        else:
            self.downsample = nn.Identity()
        
        self.final_relu = nn.ReLU()

    def forward(self, x):
        input = x
        branch_one_out = self.branch(x)

        merged_input = branch_one_out + self.downsample(input)
        out = self.final_relu(merged_input)
        return out

class ResNet(pl.LightningModule):
    """
    Miniature Flexible ResNet model.

    Strides:
    - Example stride patterns: Start with stride 2 for each stage to downsample the image.
        - 3 blocks:  strides=[1, 2, 2]          → stages: [1] | [2] | [2]
        - 4 blocks:  strides=[1, 1, 2, 2]       → stages: [1,1] | [2] | [2]
        - 5 blocks:  strides=[1, 1, 2, 1, 2]    → stages: [1,1] | [2,1] | [2]
    """
    
    def __init__(self, num_classes = 2, input_channel : int = 3, base_channel : int = 32, output_channels : list = [8, 16, 32], strides: list = [1, 2, 2], dropout=0.1, lr: float = 1e-3, use_bn : bool = True):
        super().__init__()

        # 1. Initial Layers 
        Layers = [
            nn.Conv2d(input_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channel) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        ]
        
        # 2. Stack Residual Blocks
        stride = 1
        for cout in output_channels:
            Layers.append(ResidualBlock(base_channel, cout, stride=stride, use_bn=use_bn))

            base_channel = cout
            stride = 2

        # 3. Final Classifier
        Layers.append(nn.AdaptiveAvgPool2d(1))  # Global average pooling. Will output (batch_size, channels, 1, 1)
        Layers.append(nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()) 
        Layers.append(nn.Flatten())
        Layers.append(nn.Linear(output_channels[-1], num_classes))

        self.net = nn.Sequential(*Layers)

        # 3. Loss function
        self.criterion = nn.CrossEntropyLoss()

        # 4. Learning rate
        self.lr = lr

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch: tuple, batch_idx):
        data, target = batch
        logits = self(data)

        loss = self.criterion(logits, target)
        pred = logits.argmax(dim=-1)
        acc = (target == pred).float().mean()

        # Logging to wandb
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss
    
    def validation_step(self, batch: tuple) -> None:
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