"""
Train patient-specific LSTM models using the Soft-DTW alignment loss.

This script is identical in structure to train_lstm_60min.py, with the sole
difference that the training and validation loss is replaced by soft_dtw
(gamma = 1.0) instead of standard MSE. Soft-DTW allows the model to tolerate
non-linear temporal distortions during training, addressing the time-shift
problem described in van den Hoek [7].

Evaluation is kept pointwise (RMSE, MAE) and event-based (precision, recall,
F1, F2) so that results are directly comparable to the MSE baseline.

Model
-----
LSTM [1]:
    Recurrent network with gating mechanism. Used as the baseline architecture
    for blood glucose forecasting, following Hüni [8] and the internal
    proposal of the Pattern Recognition Group [11].

Loss
----
Soft-DTW [12]:
    Differentiable relaxation of Dynamic Time Warping [13]. Replaces the hard
    minimum in the DTW dynamic programming recursion with a soft-minimum
    (log-sum-exp), allowing flexible non-linear alignment between prediction
    and ground truth during training.

References
----------
[1]  Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
     Neural Computation, 9(8), 1735-1780.
     https://doi.org/10.1162/NECO.1997.9.8.1735

[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[8]  Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
     short-term memory and graph attention network based approaches. Bachelor
     Thesis, University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
     Proposal. Internal unpublished manuscript.

[12] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
     function for time-series. In Proceedings of the 34th International
     Conference on Machine Learning (ICML 2017), vol. 70, pp. 894-903. PMLR.
     https://proceedings.mlr.press/v70/cuturi17a.html

[13] Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to find
     patterns in time series. In Proceedings of the AAAI Workshop on Knowledge
     Discovery in Databases (KDD 1994), pp. 359-370.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series
from ba_baseline.data.multi_patient_dataset import MultiPatientWindowDataset
from ba_baseline.models.lstm import LSTMForecaster
from ba_baseline.losses.soft_dtw_loss import soft_dtw
from ba_baseline.metrics.metrics import rmse, mae, event_metrics

GAMMA = 1.0  # Soft-DTW smoothing parameter [12]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_hstep_trace(model, series, lookback, horizon, device, mean, std, h_index, batch_size=2048):
    """Returns (y_true_h, y_pred_h) for a single patient series."""
    model.eval()
    n = len(series) - lookback - horizon
    if n <= 0:
        return None, None

    s_norm = ((series - mean) / (std + 1e-8)).astype(np.float32)
    xs = np.lib.stride_tricks.sliding_window_view(s_norm, lookback)[:n]
    ys = np.lib.stride_tricks.sliding_window_view(s_norm, horizon)[lookback:lookback + n]

    yhats = []
    for start in range(0, n, batch_size):
        xb = torch.tensor(xs[start:start + batch_size]).unsqueeze(-1).to(device)
        yhats.append(model(xb).cpu().numpy())
    yhat = np.concatenate(yhats, axis=0)

    yhat = yhat * (std + 1e-8) + mean
    ys = ys * (std + 1e-8) + mean
    return ys[:, h_index], yhat[:, h_index]


def train_patient(pid, train_s, val_s, lookback, horizon, device,
                  hidden_size=128, num_layers=1, lr=1e-3,
                  max_epochs=100, patience=10, batch_size=256):
    mean = float(train_s.mean())
    std = float(train_s.std())

    train_ds = MultiPatientWindowDataset(
        {pid: train_s}, [pid], lookback=lookback, horizon=horizon, mean=mean, std=std
    )
    val_ds = MultiPatientWindowDataset(
        {pid: val_s}, [pid], lookback=lookback, horizon=horizon, mean=mean, std=std
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LSTMForecaster(
        input_size=1, hidden_size=hidden_size, num_layers=num_layers,
        dropout=0.0, horizon=horizon
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = soft_dtw(model(x), y, gamma=GAMMA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += soft_dtw(model(x), y, gamma=GAMMA).item() * x.size(0)
                n += x.size(0)
        val_sdtw = val_loss / max(n, 1)

        if val_sdtw < best_val:
            best_val = val_sdtw
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, mean, std


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = load_patient_series("data/raw/all_cgm.csv")
    train_series, val_series, test_series = temporal_split_series(d, train_ratio=0.6, val_ratio=0.2)

    lookback = 24  # 2-hour context, consistent with baseline [7]
    horizon = 12
    h_index = 11  # 60-min ahead
    max_lag = 3   # kept for reference; not used in DTW loss
    HYPO_THRESH = 70.0
    EVENT_TOL = 3

    traces = {}
    rmses, maes, hypo_metrics_list, pids_done = [], [], [], []

    all_pids = sorted(d.keys())
    print(f"Training patient-specific LSTM (Soft-DTW, gamma={GAMMA}) for {len(all_pids)} patients...")

    for i, pid in enumerate(all_pids):
        train_s = train_series[pid]
        val_s = val_series[pid]
        test_s = test_series[pid]

        if (
            len(train_s) < lookback + horizon + 1
            or len(val_s) < lookback + horizon + 1
            or len(test_s) < lookback + horizon + 1
        ):
            print(f"  [{i+1}/{len(all_pids)}] patient {pid}: skipped (too short)")
            continue

        model, mean, std = train_patient(
            pid, train_s, val_s, lookback, horizon, device
        )

        y_true, y_pred = eval_hstep_trace(model, test_s, lookback, horizon, device, mean, std, h_index)
        if y_true is None:
            continue

        traces[pid] = (y_true, y_pred)
        r = rmse(y_true, y_pred)
        m = mae(y_true, y_pred)
        h = event_metrics(y_true, y_pred, threshold=HYPO_THRESH, tol=EVENT_TOL, direction="below")

        rmses.append(r)
        maes.append(m)
        hypo_metrics_list.append(h)
        pids_done.append(pid)
        print(f"  [{i+1}/{len(all_pids)}] patient {pid}: RMSE={r:.3f}  MAE={m:.3f}")

    os.makedirs("reports/results", exist_ok=True)

    np.savez(
        "reports/results/lstm_dtw_traces_all_patients.npz",
        **{f"{pid}_true": traces[pid][0] for pid in pids_done},
        **{f"{pid}_pred": traces[pid][1] for pid in pids_done},
    )

    with open("reports/results/lstm_dtw_per_patient_metrics_all.csv", "w", encoding="utf8") as f:
        f.write("patient_id,model,rmse,mae,hypo_precision,hypo_recall,hypo_f1\n")
        for pid, r, m, h in zip(pids_done, rmses, maes, hypo_metrics_list):
            f.write(f"{pid},lstm_dtw,{r:.6f},{m:.6f},{h['precision']:.6f},{h['recall']:.6f},{h['fbeta']:.6f}\n")

    summary = {
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "patient_ids": pids_done,
        "horizon_steps": horizon,
        "target_index": h_index,
        "target_minutes": 60,
        "model": "lstm_dtw",
        "hidden_size": 128,
        "num_layers": 1,
        "lookback": lookback,
        "loss": "soft_dtw",
        "gamma": GAMMA,
    }

    with open("reports/results/lstm_dtw_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LSTM Soft-DTW (test) ===")
    print(f"Patients: {len(pids_done)}")
    print(f"RMSE mean={float(np.mean(rmses)):.3f}  MAE mean={float(np.mean(maes)):.3f}")
    print("Saved to reports/results/")


if __name__ == "__main__":
    main()
