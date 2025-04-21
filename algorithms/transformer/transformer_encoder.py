import torch.nn as nn
from ..basic.ffw import FeedForward
from ..basic.residual import Residual
from .mha import MultiHeadAttention

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, attention_block: nn.Module, feed_forward_block: nn.Module, dropout=0.1):
        super().__init__()
        self.attention =  attention_block
        self.feed_forward = feed_forward_block
        self.res1 = Residual(self.attention, d_model, dropout)
        self.res2 = Residual(self.feed_forward, d_model, dropout)

    def forward(self, x, padding_mask):
        x = self.res1(x, Q = x, K = x, V = x, mask = padding_mask)
        x = self.res2(x, x = x)
        return x
