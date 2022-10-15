import torch
import torch.nn as nn
from torchsummary import summary

class DepthwiseConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding, stride):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features, 
            kernel_size=kernel_size, 
            padding=padding, 
            stride=stride, 
            groups=in_features, # groups = in_features means depthwise conv
            bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_features)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
    
class PointwiseConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(PointwiseConv, self).__init__()
        # 1x1 conv with no padding, stride=1
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_features)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = DepthwiseConv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        self.pointwise_conv = PointwiseConv(
            in_features=out_features, 
            out_features=out_features
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
        
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.conv_layers = self._create_conv_layers()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv_layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = self.softmax(x)
        return x
        
    def _create_conv_layers(self):
        # construct layers
        n_filters = 32
        layers = [
            nn.Conv2d(
                in_channels=3,
                out_channels=n_filters,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=n_filters),
            nn.ReLU(inplace=True),
        ]
        print("first conv:", n_filters)
        print("=====================================")


        for i in range(3):
            for j in range(2):
                s = 2 if j == 1 else 1
                p = 1
                if i == 0 or j == 1:
                    layers.append(
                        DepthwiseSeparableConv(
                            in_features=n_filters,
                            out_features=n_filters * 2,
                            stride=s,
                            padding=p
                        )
                    )
                    n_filters *= 2
                else:
                    layers.append(
                        DepthwiseSeparableConv(
                            in_features=n_filters,
                            out_features=n_filters,
                            stride=s,
                            padding=p
                        )
                    )
                print(n_filters)
        print("=====================================")
        
        for i in range(5):
            layers.append(
                DepthwiseSeparableConv(
                    in_features=n_filters,
                    out_features=n_filters,
                    stride=1,
                    padding=1
                )
            )
            print(n_filters)
        print("=====================================")
        
        layers.append(
            DepthwiseSeparableConv(
                in_features=n_filters,
                out_features=n_filters*2,
                stride=2,
                padding=1,
            )
        )
        n_filters *= 2
        print(n_filters)
        
        layers.append(
            DepthwiseSeparableConv(
                in_features=n_filters,
                out_features=n_filters,
                stride=1,
                padding=1,
            )
        )
        print(n_filters)
        print("=====================================")
        
        return nn.Sequential(*layers)
                

if __name__ == "__main__":
    x = torch.randn(5, 3, 224, 224)
    model = MobileNetV1()
    print(model)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {model(x).shape}')
    summary(model, (3, 224, 224), device="cpu")