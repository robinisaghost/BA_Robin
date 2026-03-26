"""
Evaluation metrics for CGM-based glucose forecasting and event detection.

Three categories of metrics are implemented, following van den Hoek [7]
and the internal proposal of the Pattern Recognition Group [11]:

1. Forecasting metrics: RMSE and MAE — standard pointwise regression metrics.

2. Timeshift metric: best-lag Δ* computed within a bounded window ±D by
   maximising cross-correlation between predicted and true trajectories.
   This quantifies the temporal misalignment that inflates RMSE even when
   the predicted shape is clinically plausible.

3. Event metrics: precision, recall, and F1-score for hypoglycemia event
   detection with a time tolerance τ. A predicted threshold crossing is
   counted as a True Positive if it falls within [tc − τ, tc + τ] of the
   true crossing tc. The F_β score is configurable (β = 1 for F1 per the
   PRG Proposal [11]; β = 2 for F2 if recall is to be weighted more
   heavily for clinical use, per Hüni [8]).

References

[7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose
    Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University
    of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

[8] Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
    short-term memory and graph attention network based approaches. Bachelor
    Thesis, University of Bern, Faculty of Science (INF).
    Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    """
    Root mean squared error.

    References

    [11] Pattern Recognition Group, University of Bern. Glucose Prediction
        Proposal. Internal unpublished manuscript.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def best_lag_crosscorr(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int) -> int:
    """
    Find the lag Δ* in [-max_lag, max_lag] that maximises cross-correlation
    between y_true and y_pred.

    This is the timeshift metric described by van den Hoek [7]: a large
    positive Δ* indicates the prediction is systematically shifted forward
    relative to the ground truth, inflating pointwise RMSE even when the
    predicted trajectory shape is correct.

    Parameters

    y_true : np.ndarray
        Ground-truth glucose trajectory.
    y_pred : np.ndarray
        Predicted glucose trajectory.
    max_lag : int
        Maximum lag D in both directions (in time steps).

    Returns

    int
        Best lag Δ* in [-max_lag, max_lag].

    References

    [7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
        Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()

    denom = (np.linalg.norm(y_true) * np.linalg.norm(y_pred)) + 1e-12

    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = y_true[-lag:]
            b = y_pred[: len(a)]
        elif lag > 0:
            a = y_true[:-lag]
            b = y_pred[lag : lag + len(a)]
        else:
            a = y_true
            b = y_pred
        if len(a) < 20:
            continue
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def crossing_times(series: np.ndarray, threshold: float, direction: str = "below"):
    """
    Detect threshold crossing times in a glucose series.

    Parameters

    series : np.ndarray
        Glucose time series.
    threshold : float
        Glucose threshold (e.g. 70 mg/dL for hypoglycemia).
    direction : str
        'below': detects crossings from above to below threshold (hypoglycemia onset).
        'above': detects crossings from below to above threshold (hyperglycemia onset).

    References

    [7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
        Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).
    """
    s = np.asarray(series)
    events = []
    for t in range(1, len(s)):
        if direction == "below":
            if s[t - 1] >= threshold and s[t] < threshold:
                events.append(t)
        else:
            if s[t - 1] <= threshold and s[t] > threshold:
                events.append(t)
    return events


def event_metrics(
    true_series: np.ndarray,
    pred_series: np.ndarray,
    threshold: float,
    tol: int,
    beta: float = 1.0,
    direction: str = "below",
):
    """
    Event-based evaluation with time tolerance τ for hypoglycemia detection.

    A predicted threshold crossing pc is counted as a True Positive (TP) if
    it falls within [tc − τ, tc + τ] of a true crossing tc. False Positives
    (FP) are predicted crossings with no matching true event; False Negatives
    (FN) are true crossings not matched by any prediction.

    F_β weighs recall β² times more than precision. β = 2 (F2) reflects the
    clinical priority of not missing hypoglycemic events, consistent with
    Hüni [8].

    Parameters

    true_series : np.ndarray
        Ground-truth glucose trajectory.
    pred_series : np.ndarray
        Predicted glucose trajectory.
    threshold : float
        Glucose threshold for event detection (70 mg/dL for hypoglycemia).
    tol : int
        Time tolerance τ in steps (e.g. tol=3 → ±15 min at 5-min sampling).
    beta : float
        β parameter for F_β score. Use 1.0 for F1, 2.0 for F2.
    direction : str
        Crossing direction passed to crossing_times().

    Returns

    dict with keys: tp, fp, fn, precision, recall, fbeta.

    References

    [7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
        Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).

    [8] Hüni, F. (2023). Predicting events of hypoglycemia. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).
    """
    true_events = crossing_times(true_series, threshold, direction=direction)
    pred_events = crossing_times(pred_series, threshold, direction=direction)

    matched_pred = set()
    tp = 0
    for te in true_events:
        for i, pe in enumerate(pred_events):
            if i in matched_pred:
                continue
            if (te - tol) <= pe <= (te + tol):
                matched_pred.add(i)
                tp += 1
                break

    fp = len(pred_events) - tp
    fn = len(true_events) - tp

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)

    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "fbeta": float(fbeta),
    }


def shift_1d(x: np.ndarray, k: int) -> np.ndarray:
    """
    Shift a 1D array by k steps, filling the vacated positions with NaN.

    k > 0: shift right (delay)   → first k positions become NaN.
    k < 0: shift left  (advance) → last -k positions become NaN.
    """
    x = np.asarray(x)
    y = x.astype(float).copy()

    if k > 0:
        y[:k] = np.nan
        y[k:] = x[:-k]
    elif k < 0:
        kk = -k
        y[-kk:] = np.nan
        y[:-kk] = x[kk:]
    return y


def best_lag_rmse(true: np.ndarray, pred: np.ndarray, max_lag: int = 24) -> int:
    """
    Find the lag k in [-max_lag, max_lag] that minimises RMSE(true, shift(pred, k)).

    Alternative to best_lag_crosscorr using RMSE minimisation instead of
    correlation maximisation.

    References

    [7] van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
        Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
        University of Bern, Faculty of Science (INF).
    """
    true = np.asarray(true).astype(float)
    pred = np.asarray(pred).astype(float)

    best_k = 0
    best_val = float("inf")

    for k in range(-max_lag, max_lag + 1):
        p = shift_1d(pred, k)
        mask = ~np.isnan(p)
        if mask.sum() < 10:
            continue
        err = true[mask] - p[mask]
        rmse_k = float(np.sqrt(np.mean(err * err)))
        if rmse_k < best_val:
            best_val = rmse_k
            best_k = k

    return best_k
