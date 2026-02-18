import numpy as np
import torch
from torch.utils.data import Dataset


class MultiPatientWindowDataset(Dataset):
    """
    Sliding windows over multiple patient time series.

    Each sample:
      x: (lookback, 1)
      y: (horizon,)
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
            raise ValueError(
                "No samples available: check lookback/horizon or patient lengths."
            )

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
