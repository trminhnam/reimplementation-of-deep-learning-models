"""
    Source: 
        * torchvision.models library
        * https://www.youtube.com/watch?v=ACmuBbuXn20
"""

import torch
import torch.nn as nn
from torchsummary import summary

from typing import cast

MAXPOOL = "M"

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, dropout=0.5) -> None:
        super(VGG, self).__init__()

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=512*7*7,
                out_features=4096
            ),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=4096, 
                out_features=4096
            ),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=4096,
                out_features=num_classes
            )
        )
        
    def forward(self, x):
        x = self.features(x)        
        x = x.reshape(x.shape[0], -1) # reshape to (batch_size, ...)
        x = self.classifier(x)
        return x
        
        
def make_layers(cfg, image_channels=3, batchnorm=False):
    layers = []
    in_channels = image_channels
    
    for v in cfg:
        if v == MAXPOOL:
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                )
            )
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=v, 
                kernel_size=3, 
                padding=1
            )
            if batchnorm:
                layers.extend([
                    conv2d, 
                    nn.BatchNorm2d(v), 
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    conv2d, 
                    nn.ReLU(inplace=True)
                ])
            in_channels = v
    return nn.Sequential(*layers)

def VGG16(in_features=3, num_classes=1000, batchnorm=False, dropout=0.5):
    features = make_layers(cfg=cfgs["D"], image_channels=in_features, batchnorm=batchnorm)
    return VGG(features=features, num_classes=num_classes, dropout=dropout)
    
        
def main():
    model = VGG16().to('cuda')
    model.eval()
    
    x = torch.randn(5, 3, 224, 224).to('cuda')
    y = model(x)
    print(y.shape)
    
    summary(model, (3, 224, 224))
    print(model)
    
if __name__ == "__main__":
    main()