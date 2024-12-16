import torch
import torch.nn as nn
from torchsummary import summary

class DepthwiseConv(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        kernel_size, 
        padding, 
        stride, 
        activation=nn.ReLU6(inplace=True)
    ):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_features, # groups = in_features means depthwise conv
            bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            x = self.activation(x)
        x = self.activation(x)
        return x
    
class PointwiseConv(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        activation=nn.ReLU6(inplace=True)
    ):
        super(PointwiseConv, self).__init__()
        # 1x1 conv with no padding, stride=1
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ExpansionLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        """
        h x w x k 1x1 conv2d , ReLU6, h x w x (tk)
        """
        super(ExpansionLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_features)
        self.activation = activation
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x

class BottleNeckBlock(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        stride,
        expansion_factor, 
        activation=nn.ReLU6(inplace=True)
    ):
        super(BottleNeckBlock, self).__init__()
        self.mid_features = in_features * expansion_factor
        self.skip_connection = stride == 1 and in_features == out_features
        self.activation = activation
        
        self.expansion_layer = ExpansionLayer(
            in_features=in_features,
            out_features=self.mid_features
        )
        
        self.depthwise_conv = DepthwiseConv(
            in_features=self.mid_features,
            out_features=self.mid_features,
            kernel_size=3,
            padding=1,
            stride=stride,
            activation=activation
        )
        
        self.pointwise_conv = PointwiseConv(
            in_features=self.mid_features,
            out_features=out_features,
            activation=None
        )
        
    def forward(self, x):
        identity = x.clone()
        
        x = self.expansion_layer(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        if self.skip_connection:
            x += identity
        x = self.activation(x)
        
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_multiplier=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.width_multiplier = width_multiplier
        self.round_nearest = round_nearest
        self.block = None
        self.last_channel = 1280
        
        self.in_features = 32
        # first layer
        self.first_layer = self._make_first_layers()
        
        setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.features = self._make_layers(setting)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(
            in_channels=1280,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            bias=False
        )
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

    def _make_first_layers(self):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.in_features,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=self.in_features),
            nn.ReLU6(inplace=True)
        )

    def _make_layers(self, setting):
        features = []
        for t, c ,n, s in setting:
            features.append(self._make_bottleneck_sequence(
                out_features=c, 
                n_layers=n, 
                stride=s, 
                expansion=t
            ))
        features.append(nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_features,
                out_channels=1280,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=1280),
            nn.ReLU6(inplace=True)
        ))
        
        return nn.Sequential(*features)

    def _make_bottleneck_sequence(self, out_features, n_layers, stride, expansion):
        layers = []
        
        layers.append(
            BottleNeckBlock(
                in_features=self.in_features,
                out_features=out_features,
                stride=stride,
                expansion_factor=expansion,
            )
        )

        for i in range(1, n_layers):
            layers.append(
                BottleNeckBlock(
                    in_features=out_features,
                    out_features=out_features,
                    stride=1,
                    expansion_factor=expansion
                )
            )

        self.in_features = out_features

        return nn.Sequential(*layers)

if __name__ == "__main__":
    x = torch.randn(5, 3, 224, 224)
    # model = MobileNetV1(width_multiplier=0.25)
    model = MobileNetV2()
    print(model)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {model(x).shape}')
    summary(model, (3, 224, 224), device="cpu")