"""
Train patient-specific LSTM binary event classifiers for hypoglycemia detection.

Instead of forecasting the glucose trajectory and deriving event detections
post-hoc (as in the baseline and Objectives 1–2), this model is trained
directly to answer: "Will glucose fall below 70 mg/dL at any point in the
next 60 minutes?"

The label for each input window is 1 if the minimum of the next horizon=12
steps (60 min) is below the hypoglycemia threshold of 70 mg/dL, and 0
otherwise. BCEWithLogitsLoss with pos_weight is used to handle the class
imbalance caused by the rarity of hypoglycemia events.

Architecture: same LSTM backbone as the baseline (LSTMForecaster with
horizon=1), using the shared baseline hyperparameters (patient 85202).
Only the output interpretation changes: the single logit output is treated
as a binary classifier, not a point forecast.

This is the primary model for Objective 3 (event-centric evaluation).
Results are compared against the forecast-derived event detection from the
tau-sweep to quantify the benefit of direct event optimisation.

References
----------
[1]  Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
     Neural Computation, 9(8), 1735-1780.
     https://doi.org/10.1162/NECO.1997.9.8.1735

[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[8]  Huni, F. (2023). Predicting events of hypoglycemia: A comparison of
     long short-term memory and graph attention network based approaches.
     Bachelor Thesis, University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.
"""

import os
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series
from ba_baseline.models.lstm import LSTMForecaster

HYPO_THRESH = 70.0
CLF_THRESHOLD = 0.5   # sigmoid threshold for binary decision


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_binary_dataset(series, lookback, horizon, threshold, mean, std):
    """
    Build (X, y) tensors for binary event classification.

    X: (N, lookback, 1) normalised input windows
    y: (N,) binary labels — 1 if min(next horizon steps) < threshold
    """
    n = len(series) - lookback - horizon
    if n <= 0:
        return None, None, 0.0

    s_norm = ((series - mean) / (std + 1e-8)).astype(np.float32)
    xs = np.lib.stride_tricks.sliding_window_view(s_norm, lookback)[:n]

    # Labels computed on original scale
    ys = np.lib.stride_tricks.sliding_window_view(
        series.astype(np.float32), horizon
    )[lookback : lookback + n]
    labels = (ys.min(axis=1) < threshold).astype(np.float32)

    X = torch.tensor(xs).unsqueeze(-1)   # (N, lookback, 1)
    y = torch.tensor(labels)              # (N,)
    pos_rate = float(labels.mean())
    return X, y, pos_rate


def train_patient(
    pid,
    train_s,
    val_s,
    lookback,
    horizon,
    device,
    hidden_size=256,
    num_layers=2,
    dropout=0.15854918709610058,
    lr=0.0009611047754271736,
    max_epochs=100,
    patience=10,
    batch_size=128,
):
    mean = float(train_s.mean())
    std = float(train_s.std())

    X_tr, y_tr, pos_rate = make_binary_dataset(
        train_s, lookback, horizon, HYPO_THRESH, mean, std
    )
    X_val, y_val, _ = make_binary_dataset(
        val_s, lookback, horizon, HYPO_THRESH, mean, std
    )
    if X_tr is None or X_val is None:
        return None, None, None

    # pos_weight = n_neg / n_pos to counteract class imbalance
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr)) - n_pos
    if n_pos == 0:
        print(f"  Patient {pid} skipped: 0 positive events in training split")
        return None, None, None
    pos_weight = torch.tensor([n_neg / n_pos], device=device)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )

    # horizon=1: single logit output used as binary classifier
    model = LSTMForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=1,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x).squeeze(-1)   # (B,)
            loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logit = model(x).squeeze(-1)
                val_loss += criterion(logit, y).item() * x.size(0)
                n += x.size(0)
        vl = val_loss / max(n, 1)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, mean, std


@torch.no_grad()
def eval_patient(model, test_s, lookback, horizon, device, mean, std):
    """Returns (y_true_labels, y_pred_probs) for a single patient."""
    X, y, _ = make_binary_dataset(
        test_s, lookback, horizon, HYPO_THRESH, mean, std
    )
    if X is None:
        return None, None

    model.eval()
    all_logits = []
    bs = 2048
    for start in range(0, len(X), bs):
        xb = X[start : start + bs].to(device)
        all_logits.append(model(xb).squeeze(-1).cpu())
    logits = torch.cat(all_logits)
    probs = torch.sigmoid(logits).numpy()
    return y.numpy(), probs


def binary_metrics(y_true, y_prob, threshold=CLF_THRESHOLD):
    """Compute precision, recall, F1, F2 for binary predictions."""
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)

    def fbeta(p, r, beta):
        denom = beta ** 2 * p + r
        return (1 + beta ** 2) * p * r / denom if denom > 0 else 0.0

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall, 1.0),
        "f2": fbeta(precision, recall, 2.0),
    }


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = load_patient_series("data/raw/all_cgm.csv")
    train_series, val_series, test_series = temporal_split_series(
        d, train_ratio=0.6, val_ratio=0.2
    )

    lookback = 24
    horizon = 12   # 60 min future window for label computation

    # Baseline hyperparameters (Optuna, patient 85202, 50 trials, MSE loss)
    best_hp = {
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.15854918709610058,
        "lr": 0.0009611047754271736,
        "batch_size": 128,
    }
    print(f"Using baseline hyperparameters: {best_hp}")

    all_pids = sorted(d.keys())
    pids_done = []
    metrics_list = []
    traces = {}

    print(f"Training LSTM binary event classifier for {len(all_pids)} patients...")

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

        result = train_patient(
            pid, train_s, val_s, lookback, horizon, device, **best_hp
        )
        model, mean, std = result
        if model is None:
            continue

        y_true, y_prob = eval_patient(model, test_s, lookback, horizon, device, mean, std)
        if y_true is None:
            continue

        m = binary_metrics(y_true, y_prob)
        pids_done.append(pid)
        metrics_list.append(m)
        traces[pid] = (y_true, y_prob)

        print(
            f"  [{i+1}/{len(all_pids)}] patient {pid}: "
            f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  F2={m['f2']:.3f}  "
            f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
        )

    os.makedirs("reports/results", exist_ok=True)

    # Save traces (true labels + predicted probabilities)
    np.savez(
        "reports/results/lstm_event_traces_all_patients.npz",
        **{f"{pid}_true": traces[pid][0] for pid in pids_done},
        **{f"{pid}_pred": traces[pid][1] for pid in pids_done},
    )

    # Per-patient CSV
    with open(
        "reports/results/lstm_event_per_patient_metrics_all.csv", "w", encoding="utf8"
    ) as f:
        f.write("patient_id,model,tp,fp,fn,precision,recall,f1,f2\n")
        for pid, m in zip(pids_done, metrics_list):
            f.write(
                f"{pid},lstm_event,{m['tp']},{m['fp']},{m['fn']},"
                f"{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['f2']:.6f}\n"
            )

    # Summary JSON
    prec_mean = float(np.mean([m["precision"] for m in metrics_list]))
    rec_mean  = float(np.mean([m["recall"]    for m in metrics_list]))
    f1_mean   = float(np.mean([m["f1"]        for m in metrics_list]))
    f2_mean   = float(np.mean([m["f2"]        for m in metrics_list]))

    summary = {
        "precision_mean": prec_mean,
        "recall_mean":    rec_mean,
        "f1_mean":        f1_mean,
        "f2_mean":        f2_mean,
        "patient_ids":    pids_done,
        "model":          "lstm_event",
        "lookback":       lookback,
        "horizon":        horizon,
        "hypo_threshold": HYPO_THRESH,
        "clf_threshold":  CLF_THRESHOLD,
        "tuning_patient": "85202",
        "hyperparameters": best_hp,
    }
    with open("reports/results/lstm_event_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== LSTM Event Classifier (test) ===")
    print(f"Patients: {len(pids_done)}")
    print(
        f"Precision={prec_mean:.4f}  Recall={rec_mean:.4f}  "
        f"F1={f1_mean:.4f}  F2={f2_mean:.4f}"
    )
    print("Saved to reports/results/")


if __name__ == "__main__":
    main()
