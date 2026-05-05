# standard MLP for image classification
# flattens input, passes through hidden layers with BatchNorm + ReLU
# then linear classifier

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=784, num_classes=10, hidden_dims=(256, 128)):
        super().__init__()

        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend([
                # (batch, in_dim) -> (batch, out_dim)
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            ])
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)
        # (batch, hidden_dims[-1]) -> (batch, num_classes)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        # x: (batch, in_channels, H, W) or (batch, input_dim)
        # (batch, *) -> (batch, input_dim)
        x = x.flatten(1)
        x = self.hidden(x)
        # (batch, hidden_dims[-1]) -> (batch, num_classes)
        x = self.classifier(x)
        # (batch, num_classes)
        return x
