"""
Train patient-specific PatchTST models using the Soft-DTW alignment loss.

Identical pipeline to train_patchtst_60min.py — same Optuna tuning, same
architecture search space, same evaluation — with the sole difference that
the training and validation loss is soft_dtw (gamma=1.0) instead of standard
MSE. Models are evaluated at h_index=11 (60-min ahead) using RMSE, MAE, and
event-based metrics for hypoglycemia detection.

Model
-----
PatchTST [2]:
    Transformer-based model that segments the input into overlapping patches
    processed by a standard Transformer encoder. Used as an advanced baseline
    alongside LSTM, following the internal proposal of the Pattern Recognition
    Group [11].

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

[8]  Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of long
     short-term memory and graph attention network based approaches. Bachelor
     Thesis, University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[9]  Garcia-Tirado, J., Colmegna, P., Villard, O., Diaz, J. L.,
     Esquivel-Zuniga, R., Koravi, C. L. K., Barnett, C. L., Oliveri, M. C.,
     Fuller, M., Brown, S. A., DeBoer, M. D., & Breton, M. D. (2023).
     Assessment of meal anticipation for improving fully automated insulin
     delivery in adults with type 1 diabetes. Diabetes Care, 46(9), 1652–1658.
     https://doi.org/10.2337/dc23-0119

[10] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
     A next-generation hyperparameter optimization framework. In Proceedings
     of the 25th ACM SIGKDD International Conference on Knowledge Discovery
     & Data Mining (pp. 2623–2631).
     https://doi.org/10.1145/3292500.3330701

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
     Proposal. Internal unpublished manuscript.
"""

import os
import json
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series
from ba_baseline.data.multi_patient_dataset import MultiPatientWindowDataset
from ba_baseline.models.patchtst import PatchTST
from ba_baseline.metrics.metrics import rmse, mae, event_metrics, lag_adjusted_rmse
from ba_baseline.losses.soft_dtw_loss import soft_dtw

GAMMA = 1.0  # Soft-DTW smoothing parameter [12]


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_hstep_trace(
    model, series, lookback, horizon, device, h_index, batch_size=2048
):
    """Returns (y_true_h, y_pred_h) for a single patient series.

    RevIN inside PatchTST handles per-window normalisation; raw mg/dL values
    are passed directly and the model output is already in mg/dL.
    """
    model.eval()
    n = len(series) - lookback - horizon
    if n <= 0:
        return None, None

    s = series.astype(np.float32)
    xs = np.lib.stride_tricks.sliding_window_view(s, lookback)[:n]
    ys = np.lib.stride_tricks.sliding_window_view(s, horizon)[
        lookback : lookback + n
    ]

    yhats = []
    for start in range(0, n, batch_size):
        xb = torch.tensor(xs[start : start + batch_size]).unsqueeze(-1).to(device)
        yhats.append(model(xb).cpu().numpy())
    yhat = np.concatenate(yhats, axis=0)

    # Model output is already in mg/dL (RevIN denormalises internally).
    # Ground truth ys has shape (N, 12); h_index selects the 60-min step.
    return ys[:, h_index], yhat[:, h_index]


def train_patient(
    pid,
    train_s,
    val_s,
    lookback,
    horizon,
    h_index,
    device,
    patch_len=12,
    stride=6,
    d_model=128,
    n_heads=8,
    n_layers=3,
    dim_ff=256,
    dropout=0.1,
    lr=5e-4,
    max_epochs=100,
    patience=10,
    batch_size=256,
):
    # No global z-score normalisation: RevIN inside PatchTST handles per-window
    # instance normalisation on raw mg/dL data, as intended by Nie et al. [2].
    # Applying global z-score first would make RevIN a near-no-op and strip the
    # window-level mean/std context that RevIN is designed to preserve and reuse.
    train_ds = MultiPatientWindowDataset(
        {pid: train_s}, [pid], lookback=lookback, horizon=horizon
    )
    val_ds = MultiPatientWindowDataset(
        {pid: val_s}, [pid], lookback=lookback, horizon=horizon
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # horizon=12: full trajectory needed for Soft-DTW alignment loss.
    model = PatchTST(
        lookback=lookback,
        horizon=horizon,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ff=dim_ff,
        dropout=dropout,
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
        val_mse = val_loss / max(n, 1)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


def select_average_patient(train_series, val_series, lookback, horizon):
    """
    Select the patient with median training-set length as the tuning patient.

    Follows the average-patient approach of Hüni [8]: hyperparameter
    optimisation is performed on a single representative patient and the
    resulting configuration is applied to all patients.
    """
    eligible = [
        pid for pid in train_series
        if len(train_series[pid]) >= lookback + horizon + 1
        and len(val_series[pid]) >= lookback + horizon + 1
    ]
    sorted_pids = sorted(eligible, key=lambda p: len(train_series[p]))
    return sorted_pids[len(sorted_pids) // 2]


def optuna_tune(pid, train_s, val_s, lookback, horizon, h_index, device, n_trials=50):
    """
    Bayesian hyperparameter search (Optuna TPE sampler) on a single patient.

    Searches over learning rate, model width/depth, dropout, and batch size.
    Each trial uses a short training budget (max_epochs=30, patience=5) to
    keep tuning feasible on CPU.  The best configuration is then used with
    the full training budget for all patients.

    Follows the average-patient Bayesian optimisation approach of Hüni [8],
    adapted for PatchTST with Optuna [10] as the standard PyTorch-compatible
    hyperparameter framework.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        # n_heads must divide d_model; both 4 and 8 divide all three choices
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        hp = dict(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=trial.suggest_int("n_layers", 2, 4),
            dim_ff=trial.suggest_categorical("dim_ff", [128, 256, 512]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            max_epochs=30,
            patience=5,
        )
        model = train_patient(pid, train_s, val_s, lookback, horizon, h_index, device, **hp)
        y_true, y_pred = eval_hstep_trace(model, val_s, lookback, horizon, device, h_index)
        if y_true is None:
            raise optuna.exceptions.TrialPruned()
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params.copy()
    # Remove tuning-budget keys — full training uses its own max_epochs/patience
    best.pop("max_epochs", None)
    best.pop("patience", None)
    return best


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    d = load_patient_series("data/raw/all_cgm.csv")
    train_series, val_series, test_series = temporal_split_series(
        d, train_ratio=0.6, val_ratio=0.2
    )

    lookback = 24  # 2 hour context; consistent with Hüni [8] max window size (120 min)
    horizon = 12
    h_index = 11  # 60-min ahead
    HYPO_THRESH = 70.0
    EVENT_TOL = 3

    # Hyperparameter tuning on a single representative patient (Hüni 2023 approach)
    avg_pid = select_average_patient(train_series, val_series, lookback, horizon)
    print(f"Tuning hyperparameters on patient {avg_pid} (50 Optuna trials)...")
    best_hp = optuna_tune(
        avg_pid, train_series[avg_pid], val_series[avg_pid],
        lookback, horizon, h_index, device, n_trials=50,
    )
    print(f"Best hyperparameters: {best_hp}")

    traces = {}
    rmses, lag_rmses, maes, hypo_metrics_list, pids_done = [], [], [], [], []

    all_pids = sorted(d.keys())
    print(f"Training patient-specific PatchTST (Soft-DTW, gamma=1.0) for {len(all_pids)} patients...")

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
            pid, train_s, val_s, lookback, horizon, h_index, device, **best_hp
        )

        y_true, y_pred = eval_hstep_trace(
            model, test_s, lookback, horizon, device, h_index
        )
        if y_true is None:
            continue

        traces[pid] = (y_true, y_pred)
        r = rmse(y_true, y_pred)
        la_r = lag_adjusted_rmse(y_true, y_pred, max_lag=12)
        m = mae(y_true, y_pred)
        h = event_metrics(
            y_true, y_pred, threshold=HYPO_THRESH, tol=EVENT_TOL, direction="below"
        )

        rmses.append(r)
        lag_rmses.append(la_r)
        maes.append(m)
        hypo_metrics_list.append(h)
        pids_done.append(pid)
        print(f"  [{i+1}/{len(all_pids)}] patient {pid}: RMSE={r:.3f}  lag-RMSE={la_r:.3f}  MAE={m:.3f}")

    os.makedirs("reports/results", exist_ok=True)

    np.savez(
        "reports/results/patchtst_dtw_traces_all_patients.npz",
        **{f"{pid}_true": traces[pid][0] for pid in pids_done},
        **{f"{pid}_pred": traces[pid][1] for pid in pids_done},
    )

    with open(
        "reports/results/patchtst_dtw_per_patient_metrics_all.csv",
        "w",
        encoding="utf8",
    ) as f:
        f.write("patient_id,model,rmse,lag_rmse,mae,hypo_precision,hypo_recall,hypo_f1\n")
        for pid, r, la_r, m, h in zip(pids_done, rmses, lag_rmses, maes, hypo_metrics_list):
            f.write(
                f"{pid},patchtst_dtw,{r:.6f},{la_r:.6f},{m:.6f},{h['precision']:.6f},{h['recall']:.6f},{h['fbeta']:.6f}\n"
            )

    summary = {
        "rmse_mean": float(np.mean(rmses)),
        "lag_rmse_mean": float(np.mean(lag_rmses)),
        "mae_mean": float(np.mean(maes)),
        "patient_ids": pids_done,
        "horizon_steps": horizon,
        "target_index": h_index,
        "target_minutes": 60,
        "model": "patchtst_dtw",
        "gamma": GAMMA,
        "lookback": lookback,
        "tuning_patient": avg_pid,
        "hyperparameters": best_hp,
    }

    with open("reports/results/patchtst_dtw_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PatchTST Soft-DTW (test) ===")
    print(f"Patients: {len(pids_done)}")
    print(f"RMSE mean={float(np.mean(rmses)):.3f}  lag-RMSE mean={float(np.mean(lag_rmses)):.3f}  MAE mean={float(np.mean(maes)):.3f}")
    print("Saved to reports/results/")


if __name__ == "__main__":
    main()
