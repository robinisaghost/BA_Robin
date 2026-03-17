"""
Data splitting utilities for patient-level CGM time series.

Two splitting strategies are provided:

- temporal_split_series: splits each patient's time series chronologically
  (train → val → test) to prevent any form of temporal data leakage. This
  is the primary strategy used throughout this work, following van den Hoek
  (2026) and Hüni (2023).

- split_patients: splits at the patient level (used for cross-patient
  generalisation experiments, currently not the primary evaluation strategy).

References
----------
van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose
    Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University
    of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
"""

import numpy as np


def split_patients(patient_ids, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split by patient IDs to avoid cross-patient leakage.

    Returns
    -------
    train_ids, val_ids, test_ids : lists of str
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
    Chronological split of each patient's time series to prevent temporal leakage.

    The series is divided in order: the first train_ratio fraction becomes the
    training set, the next val_ratio fraction the validation set, and the
    remainder the test set. Temporal order is strictly preserved.

    Split ratios follow van den Hoek (2026): 60% train, 20% validation, 20% test.

    Parameters
    ----------
    series_by_patient : dict
        Mapping from patient ID (str) to NumPy array of glucose values.
    train_ratio : float
        Fraction of each series used for training.
    val_ratio : float
        Fraction of each series used for validation.

    Returns
    -------
    train_series, val_series, test_series : dicts mapping patient ID to np.ndarray.

    References
    ----------
    van den Hoek, R. (2026). Bachelor Thesis, University of Bern.
    Hüni, F. (2023). Bachelor Thesis, University of Bern.
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
