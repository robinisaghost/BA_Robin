"""
Train patient-specific PatchTST binary event classifiers for hypoglycemia detection.

Mirror of train_lstm_event.py using the PatchTST backbone. See that module
for the full rationale and design decisions.

Architecture: same PatchTST backbone as the baseline, horizon=1, so the
single output token is treated as a binary logit. Baseline hyperparameters
are used (patient 85202, 50 Optuna trials, MSE loss).

References
----------
[2]  Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time
     series is worth 64 words: Long-term forecasting with transformers. In
     The Eleventh International Conference on Learning Representations
     (ICLR 2023). https://openreview.net/forum?id=Jbdc0vTOcol

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
from ba_baseline.models.patchtst import PatchTST

HYPO_THRESH = 70.0
CLF_THRESHOLD = 0.5


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_binary_dataset(series, lookback, horizon, threshold):
    """
    Build (X, y) tensors for binary event classification.

    X: (N, lookback, 1) raw mg/dL input windows (RevIN handles normalisation)
    y: (N,) binary labels — 1 if min(next horizon steps) < threshold
    """
    n = len(series) - lookback - horizon
    if n <= 0:
        return None, None

    s = series.astype(np.float32)
    xs = np.lib.stride_tricks.sliding_window_view(s, lookback)[:n]
    ys = np.lib.stride_tricks.sliding_window_view(s, horizon)[lookback : lookback + n]
    labels = (ys.min(axis=1) < threshold).astype(np.float32)

    X = torch.tensor(xs).unsqueeze(-1)   # (N, lookback, 1)
    y = torch.tensor(labels)              # (N,)
    return X, y


def train_patient(
    pid,
    train_s,
    val_s,
    lookback,
    horizon,
    device,
    d_model=64,
    n_heads=4,
    n_layers=3,
    dim_ff=256,
    dropout=0.025405743060852786,
    lr=0.00012116603187391654,
    max_epochs=100,
    patience=10,
    batch_size=128,
):
    X_tr, y_tr = make_binary_dataset(train_s, lookback, horizon, HYPO_THRESH)
    X_val, y_val = make_binary_dataset(val_s, lookback, horizon, HYPO_THRESH)
    if X_tr is None or X_val is None:
        return None

    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )

    # horizon=1: single output token used as binary logit
    model = PatchTST(
        lookback=lookback,
        horizon=1,
        patch_len=12,
        stride=6,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
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
    return model


@torch.no_grad()
def eval_patient(model, test_s, lookback, horizon, device):
    """Returns (y_true_labels, y_pred_probs) for a single patient."""
    X, y = make_binary_dataset(test_s, lookback, horizon, HYPO_THRESH)
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
    horizon = 12

    best_hp = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dim_ff": 256,
        "dropout": 0.025405743060852786,
        "lr": 0.00012116603187391654,
        "batch_size": 128,
    }
    print(f"Using baseline hyperparameters: {best_hp}")

    all_pids = sorted(d.keys())
    pids_done = []
    metrics_list = []
    traces = {}

    print(f"Training PatchTST binary event classifier for {len(all_pids)} patients...")

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

        model = train_patient(
            pid, train_s, val_s, lookback, horizon, device, **best_hp
        )
        if model is None:
            continue

        y_true, y_prob = eval_patient(model, test_s, lookback, horizon, device)
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

    np.savez(
        "reports/results/patchtst_event_traces_all_patients.npz",
        **{f"{pid}_true": traces[pid][0] for pid in pids_done},
        **{f"{pid}_pred": traces[pid][1] for pid in pids_done},
    )

    with open(
        "reports/results/patchtst_event_per_patient_metrics_all.csv", "w", encoding="utf8"
    ) as f:
        f.write("patient_id,model,tp,fp,fn,precision,recall,f1,f2\n")
        for pid, m in zip(pids_done, metrics_list):
            f.write(
                f"{pid},patchtst_event,{m['tp']},{m['fp']},{m['fn']},"
                f"{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['f2']:.6f}\n"
            )

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
        "model":          "patchtst_event",
        "lookback":       lookback,
        "horizon":        horizon,
        "hypo_threshold": HYPO_THRESH,
        "clf_threshold":  CLF_THRESHOLD,
        "tuning_patient": "85202",
        "hyperparameters": best_hp,
    }
    with open("reports/results/patchtst_event_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== PatchTST Event Classifier (test) ===")
    print(f"Patients: {len(pids_done)}")
    print(
        f"Precision={prec_mean:.4f}  Recall={rec_mean:.4f}  "
        f"F1={f1_mean:.4f}  F2={f2_mean:.4f}"
    )
    print("Saved to reports/results/")


if __name__ == "__main__":
    main()
