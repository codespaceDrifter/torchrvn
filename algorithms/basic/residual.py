import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, sublayer, d_model, dropout=0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, **kwargs):
        normalized = self.norm(input)
        sublayer_output = self.sublayer(**kwargs)
        dropout_output = self.dropout(sublayer_output)
        return input + dropout_output
