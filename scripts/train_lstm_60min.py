import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import split_patients
from ba_baseline.data.multi_patient_dataset import MultiPatientWindowDataset
from ba_baseline.models.lstm import LSTMForecaster
from ba_baseline.metrics.metrics import rmse, mae


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_hstep_trace_per_patient(
    model,
    series_by_patient,
    patient_ids,
    lookback,
    horizon,
    device,
    mean,
    std,
    h_index,
    batch_size=2048,
):
    """
    Build h-step-ahead trace for each patient:
    true[t] = y_true[t, h_index]
    pred[t] = y_pred[t, h_index]
    For 60-min ahead with horizon=12 (5-min steps): h_index=11.
    """
    model.eval()
    out = {}
    for pid in patient_ids:
        s = series_by_patient[str(pid)]
        if len(s) < lookback + horizon + 1:
            continue

        N = len(s) - lookback - horizon
        s_norm = ((s - mean) / (std + 1e-8)).astype(np.float32)

        # Vectorized window creation (no Python loop)
        xs_np = np.lib.stride_tricks.sliding_window_view(s_norm, lookback)[:N]       # (N, lookback)
        ys_np = np.lib.stride_tricks.sliding_window_view(s_norm, horizon)[lookback:lookback + N]  # (N, horizon)

        # Batched inference to avoid large single tensors on CPU
        yhats = []
        for start in range(0, N, batch_size):
            xb = torch.tensor(xs_np[start : start + batch_size]).unsqueeze(-1).to(device)
            yhats.append(model(xb).cpu().numpy())
        yhat = np.concatenate(yhats, axis=0)  # (N, horizon) normalized

        # back to mg/dL
        yhat = yhat * (std + 1e-8) + mean
        ys = ys_np * (std + 1e-8) + mean

        out[str(pid)] = (ys[:, h_index], yhat[:, h_index])

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
        input_size=1, hidden_size=128, num_layers=1, dropout=0.0, horizon=horizon
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # training: few epochs
    for epoch in range(1, 16):
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
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)
                total += float(loss.item()) * x.size(0)
                n += x.size(0)
        val_mse = total / max(n, 1)

        print(f"epoch {epoch} | train_mse={train_mse:.4f} | val_mse={val_mse:.4f}")

    # --- 60-min ahead trace (horizon=12 => index 11) ---
    all_ids = sorted(d.keys())
    traces_60 = eval_hstep_trace_per_patient(
        model, d, all_ids, lookback, horizon, device, g_mean, g_std, h_index=11
    )

    rmses60 = []
    maes60 = []

    for pid, (t, p) in traces_60.items():
        rmses60.append(rmse(t, p))
        maes60.append(mae(t, p))

    print("=== LSTM 60-min ahead (macro over patients) ===")
    print(
        f"RMSE mean={float(np.mean(rmses60)):.3f} | MAE mean={float(np.mean(maes60)):.3f}"
    )

    os.makedirs("reports/results", exist_ok=True)

    # Save full per-patient 60-min metrics
    with open(
        "reports/results/lstm_60min_per_patient_metrics_all.csv", "w", encoding="utf8"
    ) as f:
        f.write("patient_id,model,rmse,mae\n")
        for pid, r, m in zip(traces_60.keys(), rmses60, maes60):
            f.write(f"{pid},lstm,{r:.6f},{m:.6f}\n")

    # Save full traces for all patients
    np.savez(
        "reports/results/lstm_60min_traces_all_patients.npz",
        **{f"{pid}_true": traces_60[pid][0] for pid in traces_60.keys()},
        **{f"{pid}_pred": traces_60[pid][1] for pid in traces_60.keys()},
    )

    print("Saved LSTM 60-min artifacts to reports/results/")

    summary60 = {
        "rmse_mean": float(np.mean(rmses60)),
        "mae_mean": float(np.mean(maes60)),
        "patient_ids": list(traces_60.keys()),
        "horizon_steps": horizon,
        "target_index": 11,
        "target_minutes": 60,
    }

    with open("reports/results/lstm_60min_summary.json", "w", encoding="utf8") as f:
        json.dump(summary60, f, indent=2)


if __name__ == "__main__":
    main()
