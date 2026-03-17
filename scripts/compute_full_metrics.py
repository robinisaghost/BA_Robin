"""
Compute full metrics for the professor meeting.

Metrics (as defined in the proposal):
  Forecasting:  RMSE, MAE
  Timeshift:    best_lag (cross-correlation, bounded to ±12 steps = ±60 min)
  Event:        precision, recall, F1 (beta=1), F2 (beta=2)
                hypoglycemia threshold = 70 mg/dL, time tolerance tau = 3 steps (15 min)

Output: reports/results/meeting_metrics_full.csv
        reports/results/meeting_metrics_summary.csv  (mean ± std per model)
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ba_baseline.metrics.metrics import rmse, mae, best_lag_crosscorr, event_metrics

HYPO_THRESH = 70.0
EVENT_TOL = 3       # 3 steps × 5 min = ±15 min  (tau in the proposal)
MAX_LAG = 12        # bounded window = horizon window = 60 min
BETA_F2 = 2.0


def compute_for_model(npz_path: str, model_name: str) -> pd.DataFrame:
    data = np.load(npz_path)

    # Collect unique patient IDs from keys like "85101_true"
    pids = sorted({k.replace("_true", "").replace("_pred", "") for k in data.files})

    rows = []
    for pid in pids:
        y_true = data[f"{pid}_true"]
        y_pred = data[f"{pid}_pred"]

        r = rmse(y_true, y_pred)
        m = mae(y_true, y_pred)

        lag = best_lag_crosscorr(y_true, y_pred, max_lag=MAX_LAG)

        h1 = event_metrics(y_true, y_pred, threshold=HYPO_THRESH,
                           tol=EVENT_TOL, beta=1.0, direction="below")
        h2 = event_metrics(y_true, y_pred, threshold=HYPO_THRESH,
                           tol=EVENT_TOL, beta=2.0, direction="below")

        rows.append({
            "patient_id": int(pid),
            "model": model_name,
            # Forecasting metrics
            "rmse": round(r, 4),
            "mae": round(m, 4),
            # Timeshift metric
            "best_lag_steps": lag,
            "best_lag_minutes": lag * 5,
            # Event metrics (tau = EVENT_TOL steps = 15 min)
            "hypo_tp": h1["tp"],
            "hypo_fp": h1["fp"],
            "hypo_fn": h1["fn"],
            "hypo_precision": round(h1["precision"], 4),
            "hypo_recall": round(h1["recall"], 4),
            "hypo_f1": round(h1["fbeta"], 4),
            "hypo_f2": round(h2["fbeta"], 4),
        })

    return pd.DataFrame(rows)


def main():
    out_dir = "reports/results"
    os.makedirs(out_dir, exist_ok=True)

    lstm_df = compute_for_model(
        "reports/results/lstm_60min_traces_all_patients.npz", "LSTM"
    )
    patchtst_df = compute_for_model(
        "reports/results/patchtst_60min_traces_all_patients.npz", "PatchTST"
    )

    models = [lstm_df, patchtst_df]

    lstm_bl_path = "reports/results/lstm_bounded_lag_traces_all_patients.npz"
    patchtst_bl_path = "reports/results/patchtst_bounded_lag_traces_all_patients.npz"
    if os.path.exists(lstm_bl_path):
        models.append(compute_for_model(lstm_bl_path, "LSTM_bounded_lag"))
    if os.path.exists(patchtst_bl_path):
        models.append(compute_for_model(patchtst_bl_path, "PatchTST_bounded_lag"))

    full_df = pd.concat(models, ignore_index=True)
    full_df = full_df.sort_values(["patient_id", "model"]).reset_index(drop=True)

    per_patient_path = os.path.join(out_dir, "meeting_metrics_full.csv")
    full_df.to_csv(per_patient_path, index=False)
    print(f"Saved per-patient metrics -> {per_patient_path}")

    # Summary table: mean ± std per model
    metric_cols = ["rmse", "mae", "best_lag_steps", "best_lag_minutes",
                   "hypo_precision", "hypo_recall", "hypo_f1", "hypo_f2"]

    summary_rows = []
    for model_name, grp in full_df.groupby("model"):
        row = {"model": model_name, "n_patients": len(grp)}
        for col in metric_cols:
            row[f"{col}_mean"] = round(grp[col].mean(), 4)
            row[f"{col}_std"] = round(grp[col].std(), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "meeting_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary          -> {summary_path}")

    # Pretty print to console
    print("\n=== Summary (mean ± std) ===")
    print(f"{'Metric':<25} {'LSTM':>20} {'PatchTST':>20}")
    print("-" * 67)
    for col in metric_cols:
        lstm_row = summary_df[summary_df["model"] == "LSTM"].iloc[0]
        ptst_row = summary_df[summary_df["model"] == "PatchTST"].iloc[0]
        lstm_str = f"{lstm_row[f'{col}_mean']:.4f} ± {lstm_row[f'{col}_std']:.4f}"
        ptst_str = f"{ptst_row[f'{col}_mean']:.4f} ± {ptst_row[f'{col}_std']:.4f}"
        print(f"{col:<25} {lstm_str:>20} {ptst_str:>20}")

    print(f"\nSettings: hypo_threshold={HYPO_THRESH} mg/dL | tau={EVENT_TOL} steps "
          f"({EVENT_TOL*5} min) | lag_window=±{MAX_LAG} steps (±{MAX_LAG*5} min)")


if __name__ == "__main__":
    main()
