# to implement: jump relu. disable band. detach trick for backprop?

import torch
import torch.nn as nn

"""
Localized SAE.
Top K selection. set k = 0 to not use it.
Loss has recon and sparsity. set sparsity = 0 to not use it.
Usually we either use top k or sparsity loss

Each band is a learned center c_i (prototype vector) with bandwidth sigma_i.
how many (distance/bandwith)^2 from center

decoder is unit normed. shaped (num_features, d_model) so each ROW is a feature

Activation: z_i = exp(-||x - c_i||^2 / sigma_i^2), kept only if in top-k per input.
Reconstruction: x_hat = z @ W_dec (a learned Linear, no bias, not tied to centers).
"""

class SAE(nn.Module):
    def __init__(self, d_model: int, expansion: int, k: int = 0):
        super().__init__()
        self.d_model = d_model
        self.num_features = d_model * expansion
        self.k = k

        # centers: (num_features, d_model) - prototype vectors, magnitude meaningful
        self.centers = nn.Parameter(torch.randn(self.num_features, d_model) * 0.1)
        # ln_sigma for positivity; (num_features,). always exp before using.
        self.ln_sigma = nn.Parameter(torch.zeros(self.num_features))
        # encoder
        self.encoder = nn.Linear (d_model, self.num_features)
        # decoder: (num_features, d_model) - each row = one feature's reconstruction direction. unit-normed on-the-fly in decode.
        # init: random direction on unit sphere in d_model-space
        W_dec_init = torch.randn(self.num_features, d_model)
        W_dec_init = W_dec_init / W_dec_init.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec_init)

    # (batch, d_model) -> (batch, num_features)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # gating
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c -- avoids materializing (batch, num_features, d_model)
        # (batch, 1)
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        # (1, num_features)
        c_sq = (self.centers ** 2).sum(dim=-1).unsqueeze(0)
        # (batch, d_model) @ (d_model, num_features) -> (batch, num_features)
        xc = x @ self.centers.t()
        # (batch, num_features); clamp for fp cancellation when x ~ c
        sq_dist = (x_sq + c_sq - 2 * xc).clamp(min=0)
        # (num_features,)
        sigma_sq = torch.exp (self.ln_sigma) ** 2
        # (batch,num_features)
        gate_scale = torch.exp( - (sq_dist / sigma_sq))

        # encoder raw
        # (batch, num_features)
        encoder_raw = torch.relu(self.encoder(x))

        # z full
        # (batch, num_features)
        z_full =  encoder_raw * gate_scale

        # top k mask
        if self.k == 0:
            return z_full
        # (batch, k).
        # top k and scatter uses the indexing format: the collapsed dimension is specified through value
        topk_vals, topk_idx = z_full.topk(self.k, dim = -1)
        z = torch.zeros_like (z_full)
        z.scatter_(dim = -1, index=topk_idx, src=topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # (num_features, d_model) with unit-norm rows; gradient flows through the normalization
        W = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # (batch, num_features) @ (num_features, d_model) -> (batch, d_model)
        return z @ W

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # pass sparsity_w=0.0 to skip sparsity loss entirely (no graph built, no VRAM held)
    def loss(self, x: torch.Tensor, recon_w: float = 1.0, sparsity_w: float = 0.0) -> dict:
        x_hat, z = self.forward(x)
        # (batch, d_model) -> scalar
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        loss = recon_w * recon
        if sparsity_w > 0:
            # (batch, num_features) -> scalar; L1 on activations (z >= 0 from ReLU)
            sparsity = z.sum(dim=-1).mean()
            loss = loss + sparsity_w * sparsity
        return loss
