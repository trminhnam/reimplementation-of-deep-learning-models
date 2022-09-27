"""
Source: https://ieeexplore.ieee.org/document/726791
"""

import torch
import torch.nn as nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1
        )
        
        self.avgpool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=5,
            stride=1
        )
        
        self.fc1 = nn.Linear(
            in_features=120,
            out_features=84,
        )
        
        self.fc2 = nn.Linear(
            in_features=84,
            out_features=10
        )
        
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)

        x = self.conv3(x)
        x = self.tanh(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.tanh(x)

        x = self.fc2(x)
        x = self.softmax(x)
        
        return x
        
def main():
    model = LeNet().to('cuda')
    model.eval()
    
    x = torch.randn(5, 1, 32, 32).to('cuda')
    y = model(x)
    print(y.shape)
    
    summary(model, (1, 32, 32))
    print(model)
    
if __name__ == "__main__":
    main()