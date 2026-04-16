"""
PatchTST-based glucose forecasting model (event-centric variant).

This variant uses sinusoidal positional encoding instead of learnable
embeddings, following the original Transformer formulation by Vaswani et al. [3].
RevIN normalisation is applied inline following Kim et al. [4].

Based on
---------
- Nie et al. [2] — PatchTST architecture (patch projection, Transformer encoder,
  linear head). Reference implementation: https://github.com/yuqinie98/PatchTST
- Vaswani et al. [3] — sinusoidal positional encoding formula.
  Reference implementation pattern: https://github.com/pytorch/pytorch
  (torch.nn.Transformer tutorial).
- RevIN: https://github.com/ts-kim/RevIN (Kim et al. [4])

References
----------
[2]  Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time
     series is worth 64 words: Long-term forecasting with transformers.
     ICLR 2023. https://openreview.net/forum?id=Jbdc0vTOcol

[3]  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
     Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all
     you need. NeurIPS 2017.

[4]  Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022).
     Reversible instance normalization for accurate time-series forecasting
     against distribution shift. ICLR 2022.
     https://openreview.net/forum?id=cGDAkQo1C0p

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
     Proposal. Internal unpublished manuscript.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class PatchTST(nn.Module):
    """
    Minimal PatchTST-style forecaster for univariate series using PyTorch TransformerEncoder.
    Input:  (B, lookback, 1)
    Output: (B, horizon)
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        # number of patches
        n_patches = 1 + (self.lookback - self.patch_len) // self.stride
        if n_patches <= 0:
            raise ValueError("Invalid patch settings: lookback must be >= patch_len.")
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=4096)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        x = x.squeeze(-1)  # (B, T)

        # RevIN: normalize by instance statistics to prevent temporal shift
        x_mean = x.mean(dim=1, keepdim=True)          # (B, 1)
        x_std = x.std(dim=1, keepdim=True) + 1e-8     # (B, 1)
        x = (x - x_mean) / x_std

        # (B, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # (B, n_patches, d_model)
        z = self.patch_proj(patches)
        z = self.pos_enc(z)
        z = self.encoder(z)
        z = self.norm(z)

        flat = z.reshape(z.size(0), -1)  # (B, n_patches * d_model)
        yhat = self.head(flat)           # (B, horizon)

        # RevIN: denormalize predictions back to original scale
        yhat = yhat * x_std + x_mean
        return yhat
