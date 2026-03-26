"""
Sliding-window PyTorch Dataset for multi-patient CGM time series.

The rolling (sliding-window) prediction strategy is standard in deep
learning for time series forecasting: a fixed-length lookback window is
extracted at every time step, and the model is trained to predict the
subsequent horizon steps, following Lim and Zohren [5]. This strategy is
also adopted in the internal proposal of the Pattern Recognition Group [11].

Per-patient z-score normalisation (zero mean, unit variance) is applied
to both the input window x and the target window y using statistics
computed exclusively on the training split of the respective patient.
Normalising the targets ensures that the MSE loss is on a consistent
scale across patients with different absolute glucose levels, following
standard practice in blood glucose forecasting, as demonstrated by Nemat et al. [6].

References
----------
[5] Lim, B., & Zohren, S. (2020). Time series forecasting with deep
    learning: A survey. Philosophical Transactions of the Royal Society A,
    379(2194). https://doi.org/10.1098/rsta.2020.0209

[6] Nemat, H., Khadem, H., Elliott, J., & Benaissa, M. (2024). Data-driven
    blood glucose level prediction in type 1 diabetes: a comprehensive
    comparative analysis. Scientific Reports, 14(1), 21863.
    https://doi.org/10.1038/s41598-024-70277-x

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
    Proposal. Internal unpublished manuscript.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiPatientWindowDataset(Dataset):
    """
    Sliding-window dataset over multiple patient CGM time series.

    For each patient series, all valid windows of length
    (lookback + horizon) are extracted with a stride of one step,
    following the rolling prediction strategy of Lim and Zohren [5].

    Each sample:
      x : (lookback, 1)  — normalised input context window
      y : (horizon,)     — normalised target trajectory

    If mean and std are provided (computed from the training split),
    both x and y are z-score normalised. The caller is responsible for
    inverse-transforming model outputs before computing metrics in the
    original unit (mg/dL).
    """

    def __init__(
        self,
        series_by_patient: dict,
        patient_ids,
        lookback: int,
        horizon: int,
        mean=None,
        std=None,
    ):
        self.mean = float(mean) if mean is not None else None
        self.std = float(std) if std is not None else None
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        self.series = {}
        self.index = []  # list of (pid, start_idx)

        for pid in patient_ids:
            pid = str(pid)
            s = np.asarray(series_by_patient[pid], dtype=np.float32)
            if len(s) < self.lookback + self.horizon + 1:
                continue
            self.series[pid] = s
            max_start = len(s) - (self.lookback + self.horizon)
            for i in range(max_start):
                self.index.append((pid, i))

        if len(self.index) == 0:
            raise ValueError("No samples available")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, i = self.index[int(idx)]
        s = self.series[pid]

        x = s[i : i + self.lookback]  # (lookback,)
        y = s[i + self.lookback : i + self.lookback + self.horizon]  # (horizon,)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
            y = (y - self.mean) / (self.std + 1e-8)

        x = torch.from_numpy(x).unsqueeze(-1)  # (lookback, 1)
        y = torch.from_numpy(y)  # (horizon,)
        return x, y
