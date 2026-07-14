import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class GroupConvBlock(nn.Module):
    """Pointwise projection -> grouped spatial convolution -> pointwise mixing."""

    def __init__(self, in_channels, out_channels, groups=4, stride=1):
        super().__init__()
        hidden = out_channels
        if hidden % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.block = nn.Sequential(
            ConvBNReLU(in_channels, hidden, kernel_size=1),
            ConvBNReLU(hidden, hidden, kernel_size=3, stride=stride, groups=groups),
            ConvBNReLU(hidden, out_channels, kernel_size=1),
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_residual else out


class GCNN(nn.Module):
    """Lightweight grouped-convolution network for 43-class GTSRB recognition."""

    def __init__(self, num_classes=43, groups=4, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=1),
            GroupConvBlock(32, 64, groups=groups, stride=2),
            GroupConvBlock(64, 64, groups=groups),
            GroupConvBlock(64, 128, groups=groups, stride=2),
            GroupConvBlock(128, 128, groups=groups),
            GroupConvBlock(128, 256, groups=groups, stride=2),
            GroupConvBlock(256, 256, groups=groups),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BaselineCNN(nn.Module):
    """Traditional CNN baseline used for parameter, speed and accuracy comparison."""

    def __init__(self, num_classes=43, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(2),
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 43):
    name = name.lower()
    if name == "gcnn":
        return GCNN(num_classes=num_classes)
    if name in {"cnn", "baseline"}:
        return BaselineCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}. Choose 'gcnn' or 'cnn'.")
