import numpy as np


def split_patients(patient_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split by patient IDs to avoid leakage.
    Returns: train_ids, val_ids, test_ids (lists of str)
    """
    pids = list(map(str, patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    n = len(pids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = pids[:n_train]
    val_ids = pids[n_train : n_train + n_val]
    test_ids = pids[n_train + n_val :]

    return train_ids, val_ids, test_ids


def temporal_split_series(series_by_patient, train_ratio=0.6, val_ratio=0.2):
    """
    Temporal split of each patient's time series, preserving temporal order.
    First train_ratio -> train, next val_ratio -> val, remainder -> test.
    Returns: train_series, val_series, test_series (dicts pid -> np.ndarray)
    """
    train_series, val_series, test_series = {}, {}, {}
    for pid, s in series_by_patient.items():
        n = len(s)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_series[pid] = s[:n_train]
        val_series[pid] = s[n_train : n_train + n_val]
        test_series[pid] = s[n_train + n_val :]
    return train_series, val_series, test_series
