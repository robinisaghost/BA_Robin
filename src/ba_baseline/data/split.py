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
