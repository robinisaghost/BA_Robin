import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import split_patients
from ba_baseline.data.multi_patient_dataset import MultiPatientWindowDataset
from ba_baseline.models.lstm import LSTMForecaster
from ba_baseline.metrics.metrics import rmse, mae, best_lag_crosscorr, event_metrics


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_1step_trace_per_patient(
    model, series_by_patient, patient_ids, lookback, horizon, device, mean, std
):
    """
    Build 1-step ahead trace for each patient:
    true[t] = s[lookback + t]
    pred[t] = model(s[t:t+lookback])[0]  (first horizon step)
    Returns dict pid -> (true_1, pred_1)
    """
    model.eval()
    out = {}
    for pid in patient_ids:
        s = series_by_patient[str(pid)]
        if len(s) < lookback + horizon + 1:
            continue

        xs = []
        ys = []
        # windows over patient
        max_start = len(s) - (lookback + horizon)
        for i in range(max_start):
            x = ((s[i : i + lookback] - mean) / (std + 1e-8)).astype(np.float32)
            y = (
                (s[i + lookback : i + lookback + horizon] - mean) / (std + 1e-8)
            ).astype(np.float32)
            xs.append(x)
            ys.append(y)

        xs = torch.tensor(np.stack(xs)).unsqueeze(-1).to(device)  # (N, lookback, 1)
        ys = np.stack(ys)  # (N, horizon)

        yhat = model(xs).detach().cpu().numpy()  # (N, horizon)

        # back to mg/dL for metrics/plots
        yhat = yhat * (std + 1e-8) + mean
        ys = ys * (std + 1e-8) + mean

        true_1 = ys[:, 0]
        pred_1 = yhat[:, 0]
        out[str(pid)] = (true_1, pred_1)
    return out


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    csv_path = "data/raw/all_cgm.csv"
    d = load_patient_series(csv_path)

    train_ids, val_ids, test_ids = split_patients(
        d.keys(), train_ratio=0.7, val_ratio=0.15, seed=42
    )
    # compute global mean/std from TRAIN patients only
    train_values = np.concatenate([d[pid] for pid in train_ids]).astype(np.float32)
    g_mean = float(train_values.mean())
    g_std = float(train_values.std())
    print("train mean/std:", g_mean, g_std)

    lookback = 72
    horizon = 12

    train_ds = MultiPatientWindowDataset(
        d, train_ids, lookback=lookback, horizon=horizon, mean=g_mean, std=g_std
    )
    val_ds = MultiPatientWindowDataset(
        d, val_ids, lookback=lookback, horizon=horizon, mean=g_mean, std=g_std
    )

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, drop_last=False, num_workers=0
    )

    model = LSTMForecaster(
        input_size=1, hidden_size=128, num_layers=2, dropout=0.1, horizon=horizon
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Smoke training: few epochs
    for epoch in range(1, 21):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_mse = total / max(n, 1)

        model.eval()
        total = 0.0
        n = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            total += float(loss.item()) * x.size(0)
            n += x.size(0)
        val_mse = total / max(n, 1)

        print(f"epoch {epoch} | train_mse={train_mse:.4f} | val_mse={val_mse:.4f}")

    traces = eval_1step_trace_per_patient(
        model, d, test_ids, lookback, horizon, device, g_mean, g_std
    )

    # aggregate metrics per patient
    lags = []
    rmses = []
    maes = []
    pr_list = []
    re_list = []
    f1_list = []
    f2_list = []

    for pid, (t, p) in traces.items():
        rmses.append(rmse(t, p))
        maes.append(mae(t, p))
        lags.append(best_lag_crosscorr(t, p, max_lag=24))

        m1 = event_metrics(t, p, threshold=70.0, tol=6, beta=1.0)
        m2 = event_metrics(t, p, threshold=70.0, tol=6, beta=2.0)
        pr_list.append(m1["precision"])
        re_list.append(m1["recall"])
        f1_list.append(m1["fbeta"])
        f2_list.append(m2["fbeta"])

    print("=== TEST 1-step (macro over patients) ===")
    print(
        f"RMSE mean={float(np.mean(rmses)):.3f} | MAE mean={float(np.mean(maes)):.3f}"
    )
    print(
        f"best_lag mean={float(np.mean(lags)):.2f} steps | median={float(np.median(lags)):.2f} steps"
    )
    print(
        f"Precision mean={float(np.mean(pr_list)):.3f} | Recall mean={float(np.mean(re_list)):.3f}"
    )
    print(
        f"F1 mean={float(np.mean(f1_list)):.3f} | F2 mean={float(np.mean(f2_list)):.3f}"
    )
    # --- save artifacts for plotting ---
    os.makedirs("reports/results", exist_ok=True)

    # Save per-patient lags and summary metrics
    summary = {
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "lag_mean_steps": float(np.mean(lags)),
        "lag_median_steps": float(np.median(lags)),
        "precision_mean": float(np.mean(pr_list)),
        "recall_mean": float(np.mean(re_list)),
        "f1_mean": float(np.mean(f1_list)),
        "f2_mean": float(np.mean(f2_list)),
        "test_patient_ids": list(traces.keys()),
    }

    with open("reports/results/lstm_1step_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    # Save lags per patient
    with open("reports/results/lstm_1step_lags.csv", "w", encoding="utf8") as f:
        f.write("patient_id,lag_steps\n")
        for pid, lag in zip(traces.keys(), lags):
            f.write(f"{pid},{lag}\n")

    # Save traces for a few example patients (for overlay plots)
    # Pick 3 patients deterministically
    example_pids = sorted(traces.keys())[:3]
    np.savez(
        "reports/results/lstm_1step_traces_examples.npz",
        **{f"{pid}_true": traces[pid][0] for pid in example_pids},
        **{f"{pid}_pred": traces[pid][1] for pid in example_pids},
    )
    print("Saved artifacts to reports/results/")


if __name__ == "__main__":
    main()
