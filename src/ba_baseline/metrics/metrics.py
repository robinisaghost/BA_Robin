import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def best_lag_crosscorr(y_true: np.ndarray, y_pred: np.ndarray, max_lag: int) -> int:
    """
    Maximize corr(y_true[t], y_pred[t+lag]) for lag in [-max_lag, max_lag].
    Returns best lag (int).
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


def crossing_times(series: np.ndarray, threshold: float):
    s = np.asarray(series)
    events = []
    for t in range(1, len(s)):
        if s[t - 1] >= threshold and s[t] < threshold:
            events.append(t)
    return events


def event_metrics(
    true_series: np.ndarray,
    pred_series: np.ndarray,
    threshold: float,
    tol: int,
    beta: float = 1.0,
):
    true_events = crossing_times(true_series, threshold)
    pred_events = crossing_times(pred_series, threshold)

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
    Shift 1D array by k steps.
    k > 0: shift right (delay)  -> first k become NaN
    k < 0: shift left  (advance)-> last -k become NaN
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
    # k == 0: unchanged
    return y


def best_lag_rmse(true: np.ndarray, pred: np.ndarray, max_lag: int = 24) -> int:
    """
    Find lag k in [-max_lag, max_lag] that minimizes RMSE(true, shift_1d(pred, k)).
    This returns the *correction lag* to apply to pred.
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
