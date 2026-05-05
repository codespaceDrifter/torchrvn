# standard CNN for image classification
# conv blocks: Conv2d (same padding) -> BatchNorm -> ReLU -> MaxPool2d
# then global average pool -> linear classifier
#
# max pooling picks the strongest activation per region — standard for CNNs (ResNet, VGG, etc.)
# global average pooling at the end replaces flattening — fewer params, acts as regularization

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, channels=(32, 64, 128), pool_size=2):
        super().__init__()

        layers = []
        for out_channels in channels:
            layers.extend([
                # same padding: padding = kernel_size // 2 keeps spatial dims unchanged
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # (batch, C, H, W) -> (batch, C, H/pool_size, W/pool_size)
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            ])
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)
        # (batch, channels[-1], H/pool^n, W/pool^n) -> (batch, channels[-1], 1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # (batch, channels[-1]) -> (batch, num_classes)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        x = self.conv_blocks(x)
        # (batch, channels[-1], H/pool^n, W/pool^n) -> (batch, channels[-1])
        x = self.global_avg_pool(x).flatten(1)
        x = self.classifier(x)
        # (batch, num_classes)
        return x
