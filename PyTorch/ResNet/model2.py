import torch
import torch.nn as nn
from torchsummary import summary

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        """
        * in_channels: number of channels of input
        * out_channels: number of channels of output
        * identity_downsample: a convolutional layer in which change the input size/number of channels
        * stride: stride of the convolutinonal layer of the main path
        """
        
        super(block, self).__init__()
        
        # the number of a channels after a large layers is increased by factor of 4
        self.expansion = 4
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels
        )
        
        # the first block of conv2_x to conv5_x layer will have second conv as:
        # 3x3, mid size, stride=2, pad=same
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels*self.expansion,
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=out_channels*self.expansion
        )
        
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.relu(x)
        
        return x
        
        
class ResNet(nn.Module):
    def __init__(
        self, 
        block, 
        layers, 
        image_channels, 
        num_classes
    ):
        """
        * block: residual block class
        * layers: number of layer in each layer type (conv2_x to conv5_x)
            e.g. layers = [3, 4, 6, 3] for ResNet50
        * image_channels: number channels of the input image
        * num_classes: number of classes to predict
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            # out_channels=self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            # num_features=self.in_channels
            num_features=64
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        # ResNet layers (conv2_x to conv5_x)
        self.layer1 = self._make_layer(
            block, layers[0], out_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], out_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], out_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], out_channels=512, stride=2
        )
        
        # fix the input size to a particular size
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512*4, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
    
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        """
        * stride: =2 if we implement conv3_1, conv4_1, and conv5_1
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels*4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(
                    num_features=out_channels*4
                ),
            )
            
            # the first block of each layer changes the number of channels 
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    identity_downsample=identity_downsample,
                    stride=stride,
                ),
            )
        
        # Exansion size by factor of 4 for ResNet 50, 101, 152
        self.in_channels = out_channels * 4
        
        for _ in range(num_residual_blocks - 1):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                )
            )

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)

def main():
    model = ResNet50().to('cuda')
    x = torch.randn(5, 3, 224, 224).to('cuda')
    y = model(x)
    print(y.shape)
    summary(model, (3, 224, 224))
    
if __name__ == "__main__":
    main()