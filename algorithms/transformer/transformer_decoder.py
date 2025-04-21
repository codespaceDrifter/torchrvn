import torch.nn as nn
from ..basic.ffw import FeedForward
from ..basic.residual import Residual
from .mha import MultiHeadAttention

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, self_attention_block: nn.Module, cross_attention_block: nn.Module, feed_forward_block: nn.Module, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.res1 = Residual(self.self_attention, d_model, dropout)
        self.res2 = Residual(self.cross_attention, d_model, dropout)
        self.res3 = Residual(self.feed_forward, d_model, dropout)

    def forward(self, x, encoder_output, self_mask, cross_mask):

        x = self.res1(x, Q = x, K = x, V = x, mask = self_mask)
        x = self.res2(x, Q = x, K = encoder_output, V = encoder_output, mask = cross_mask)
        x = self.res3(x, x = x)
        return x