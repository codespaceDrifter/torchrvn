import torch
import torch.nn as nn

# jumprelu, upper lower bound, cone gate, l0 loss  

class SAE (nn.Module):
    def __init__(self, embed_dim, expansion_factor, band_eps = 0.001):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim * expansion_factor
        self.band_eps = band_eps

        # dictionary weights
        self.encoder_weight = nn.Parameter(torch.randn(self.feature_dim, self.embed_dim) / self.embed_dim ** 0.5) # (out, in) format
        self.encoder_bias = nn.Parameter(torch.randn(self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_weight = nn.Parameter(torch.randn(self.embed_dim, self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_bias = nn.Parameter(torch.randn(self.embed_dim) / self.embed_dim ** 0.5)

        # gate weights
        self.slab_gate_lower_bound = nn.Parameter(torch.zeros(self.feature_dim,))
        self.slab_gate_upper_bound = nn.Parameter(torch.full((self.feature_dim,), 10.0))
        self.cone_gate_upper_bound = nn.Parameter(torch.full((self.feature_dim,), 1.57)) # initalize to right angle


    # straight through estimator
    # we want to use hard for forward but use soft estimator for backprop
    def ste_gate (self, arg, threshhold, lower_bound):
        if lower_bound == True:
            hard = (arg >= threshhold).float()
            smooth = torch.sigmoid ((arg - threshhold) / self.band_eps)
        else:
            hard = (arg < threshhold).float()
            smooth = torch.sigmoid ((threshhold - arg) / self.band_eps)
        # forward value = hard, backward due to detach gradient uses soft
        return smooth + (hard - smooth).detach()

    def encode (self, input):
        pre_gate_features = (input - self.decoder_bias) @ self.encoder_weight.T + self.encoder_bias

        slab_lower_gate = self.ste_gate(pre_gate_features, self.slab_gate_lower_bound, True)
        slab_upper_gate = self.ste_gate(pre_gate_features,self.slab_gate_upper_bound, False)
        angles = torch.acos(
            ((input - self.decoder_bias) @ self.encoder_weight.T 
            / (input - self.decoder_bias).norm(dim=-1, keepdim=True) / self.encoder_weight.norm(dim=-1))
            .clamp(-1 + 1e-5, 1- 1e-5))
        cone_upper_gate = self.ste_gate(angles, self.cone_gate_upper_bound, False)
        combined_gate = slab_lower_gate * slab_upper_gate * cone_upper_gate

        features = pre_gate_features * combined_gate
        return features, combined_gate

    def decode (self, features):
        return features @ self.decoder_weight.T



# trains SAE from streaming activations of the model
# assumes a sparsity_coeff is initalized at 1e-3
def train_SAE (sae:SAE, activation, optimizer, target_active, sparsity_coeff):
    features, combined_gate = sae.encode(activation)

    error = (combined_gate.sum(-1) - target_active).mean()
    sparsity_coeff *= (1 + 0.01 * ( 1 if error.item() > 0 else -1 ))
    l0_loss = torch.clamp (error, min=0) * sparsity_coeff

    pred = sae.decode(features)
    recon_loss = (pred-activation).pow(2).sum(-1).mean()
    loss = l0_loss + recon_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # keep decoder feature columns uniform
    with torch.no_grad():
        sae.decoder_weight /= sae.decoder_weight.norm (dim = 0, keepdim = True)
    return sparsity_coeff, loss.item(), l0_loss.item(), recon_loss.item()
