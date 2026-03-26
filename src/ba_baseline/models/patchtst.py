"""
PatchTST-based glucose forecasting model.

PatchTST decomposes a time series into overlapping patches which are then
projected and processed by a standard Transformer encoder. Compared to
point-wise attention, the patch-based approach captures local temporal
patterns more efficiently and reduces the sequence length seen by the encoder.

Instance normalisation (RevIN) is applied before the encoder and reversed
after the head to stabilise training across patients with different glucose
ranges.

PatchTST is used as an advanced Transformer baseline alongside LSTM, following
the internal proposal of the Pattern Recognition Group [11].

References

[2] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series
    is worth 64 words: Long-term forecasting with transformers. In The Eleventh
    International Conference on Learning Representations (ICLR 2023).
    https://openreview.net/forum?id=Jbdc0vTOcol

[4] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022).
    Reversible instance normalization for accurate time-series forecasting
    against distribution shift. In International Conference on Learning
    Representations (ICLR 2022).
    https://openreview.net/forum?id=cGDAkQo1C0p

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    PatchTST forecaster for univariate series.

    Input:  (B, lookback, 1)
    Output: (B, horizon)

    The architecture follows Nie et al. [2]: the input series is split into
    overlapping patches, projected to d_model, enriched with learnable
    positional encodings (as in the original PatchTST implementation), and
    processed by a standard Transformer encoder with Pre-LayerNorm. A linear
    head maps the flattened patch representations to the forecast horizon.

    Reversible Instance Normalisation (RevIN) [4] is applied before the encoder
    and reversed after the head. RevIN normalises each sample independently by
    its own mean and standard deviation, which stabilises training when patients
    exhibit substantially different baseline glucose levels and inter-patient
    distribution shift is present. This is the recommended normalisation
    strategy for PatchTST on heterogeneous patient cohorts [4].

    Parameters

    lookback : int
        Length of the input context window (in time steps).
    horizon : int
        Number of future time steps to predict.
    patch_len : int
        Length of each patch (in time steps).
    stride : int
        Step size between consecutive patches.
    d_model : int
        Transformer embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    dim_ff : int
        Feed-forward hidden dimension inside the encoder.
    dropout : float
        Dropout probability.

    References

    [2] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
        A time series is worth 64 words: Long-term forecasting with
        transformers. ICLR 2023.
        https://openreview.net/forum?id=Jbdc0vTOcol

    [3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
        Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all
        you need. NeurIPS 2017.

    [4] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022).
        Reversible instance normalization for accurate time-series forecasting
        against distribution shift. ICLR 2022.
        https://openreview.net/forum?id=cGDAkQo1C0p
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        n_patches = 1 + (self.lookback - self.patch_len) // self.stride
        if n_patches <= 0:
            raise ValueError("Invalid patch settings: lookback must be >= patch_len.")
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(self.patch_len, d_model)

        # Learnable positional encoding, following the original PatchTST
        # implementation [2].
        self.pos_enc = nn.Embedding(n_patches, d_model)

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

        # RevIN normalisation [4]: stabilises training across patients with
        # different glucose baseline levels.
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1)
        x_std = x.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
        x = (x - x_mean) / x_std

        # Patching: (B, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        z = self.patch_proj(patches)  # (B, n_patches, d_model)

        # Learnable positional encoding [2]
        positions = torch.arange(self.n_patches, device=x.device)
        z = z + self.pos_enc(positions)  # (B, n_patches, d_model)

        z = self.encoder(z)
        z = self.norm(z)

        flat = z.reshape(z.size(0), -1)  # (B, n_patches * d_model)
        yhat = self.head(flat)  # (B, horizon)

        # RevIN denormalisation
        yhat = yhat * x_std + x_mean
        return yhat
