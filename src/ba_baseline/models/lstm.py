"""
LSTM-based glucose forecasting model.

The Long Short-Term Memory (LSTM) network is a recurrent neural network
architecture designed to learn long-range dependencies in sequential data
via a gating mechanism (forget, input, output gates) that controls information
flow through a cell state.

LSTMs are an established baseline for blood glucose prediction from CGM data
and serve as the primary baseline model in this work, following Hüni [8]
and the internal proposal of the Pattern Recognition Group [11].

References
----------
[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation, 9(8), 1735–1780.
    https://doi.org/10.1162/NECO.1997.9.8.1735

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.
"""

import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM-based multi-step forecaster for univariate time series.

    Takes a lookback window of shape (B, T, 1) and produces a direct
    multi-horizon forecast of shape (B, horizon) via a linear head applied
    to the last hidden state.

    Parameters
    ----------
    input_size : int
        Number of input features (1 for univariate CGM series).
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers (ignored if num_layers=1).
    horizon : int
        Number of future time steps to predict.

    References
    ----------
    [1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
        Neural Computation, 9(8), 1735–1780.
        https://doi.org/10.1162/NECO.1997.9.8.1735
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)  # (B, T, hidden_size)
        last = out[:, -1, :]   # (B, hidden_size)
        yhat = self.head(last) # (B, horizon)
        return yhat
