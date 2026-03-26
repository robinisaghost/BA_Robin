"""
Data splitting utilities for patient-level CGM time series.

The splitting strategy provided:

- temporal_split_series: splits each patient's time series chronologically
  (train → val → test) to prevent any form of temporal data leakage. This
  is the primary strategy used throughout this work, following van den Hoek
  [7] and Hüni [8].


References

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose
    Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University
    of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.
"""


def temporal_split_series(series_by_patient, train_ratio=0.6, val_ratio=0.2):
    """
    Chronological split of each patient's time series to prevent temporal leakage.

    The series is divided in order: the first train_ratio fraction becomes the
    training set, the next val_ratio fraction the validation set, and the
    remainder the test set. Temporal order is strictly preserved.

    Split ratios follow van den Hoek [7]: 60% train, 20% validation, 20% test.

    Parameters

    series_by_patient : dict
        Mapping from patient ID (str) to NumPy array of glucose values.
    train_ratio : float
        Fraction of each series used for training.
    val_ratio : float
        Fraction of each series used for validation.

    Returns

    train_series, val_series, test_series : dicts mapping patient ID to np.ndarray.

    References

    [7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
        Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).
        Supervisor: PD Dr. Kaspar Riesen.
    [8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of
        long short-term memory and graph attention network based approaches.
        Bachelor Thesis, University of Bern, Faculty of Science (INF).
        Supervisor: PD Dr. Kaspar Riesen.
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
