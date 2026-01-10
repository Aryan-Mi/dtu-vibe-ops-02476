from torch import nn
import torch
import pytorch_lightning as pl
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)

# 0. Mapping of EfficientNet versions to their input sizes 
# Credits to : https://www.kaggle.com/code/carlolepelaars/efficientnetb5-with-keras-aptos-2019 for the recommended size
INPUT_SIZE = {
    'b0': 224,
    'b1': 240,
    'b2': 260,
    'b3': 300,
    'b4': 380,
    'b5': 456,
    'b6': 528,
    'b7': 600,
}

# 1. Baseline CNN Model
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

        # 3. Loss function & learning rate
        self.criterion = nn.CrossEntropyLoss()
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

# 2. ResNet Model  
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
        in_ch = base_channel
        for cout, stride in zip(output_channels, strides):
            Layers.append(
                ResidualBlock(in_ch, cout, stride=stride, use_bn=use_bn)
            )
            in_ch = cout  

        # 3. Final Classifier
        Layers.append(nn.AdaptiveAvgPool2d(1))  # Global average pooling. Will output (batch_size, channels, 1, 1)
        Layers.append(nn.Flatten())
        Layers.append(nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()) 
        Layers.append(nn.Linear(output_channels[-1], num_classes))

        self.net = nn.Sequential(*Layers)

        # 3. Loss function & learning rate
        self.criterion = nn.CrossEntropyLoss()
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

# 3. EfficientNet Wrapper Model
class EfficientNet(pl.LightningModule):
    """EfficientNet model from torchvision. Based on this paper: https://arxiv.org/abs/1905.11946"""
    def __init__(self, num_classes = 2, model_size: str = 'b5', lr: float = 1e-3, use_bn : bool = True, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        # 0 . Define helper
        def disable_bn_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.Identity()) # Disabling BatchNorm by replacing them with Identity
                     
                disable_bn_layers(child) 

        # 1. Load EfficientNet Backbone (and pretrained weights if specified)
        efficientnet_configs = {
            'b0': (models.efficientnet_b0, EfficientNet_B0_Weights),
            'b1': (models.efficientnet_b1, EfficientNet_B1_Weights),
            'b2': (models.efficientnet_b2, EfficientNet_B2_Weights),
            'b3': (models.efficientnet_b3, EfficientNet_B3_Weights),
            'b4': (models.efficientnet_b4, EfficientNet_B4_Weights),
            'b5': (models.efficientnet_b5, EfficientNet_B5_Weights),
            'b6': (models.efficientnet_b6, EfficientNet_B6_Weights),
            'b7': (models.efficientnet_b7, EfficientNet_B7_Weights),
        }
        if model_size not in efficientnet_configs:
            raise ValueError(f"Unsupported size: {model_size}")

        self.model = efficientnet_configs[model_size][0](weights=efficientnet_configs[model_size][1].DEFAULT if pretrained else None) # Load pretrained weights if specified

        # 2. Freeze backbone layers if needed
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # 3. Modify the classifier to match num_classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # 4. Remove BatchNorm layers if needed. 
        if not use_bn:  
            disable_bn_layers(self.model)

        # 5. Loss function & learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
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