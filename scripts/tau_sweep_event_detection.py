"""
Objective 3: tau-Sweep for Hypoglycemia Event Detection

Evaluates all trained models across a range of time tolerance values
tau ∈ {1, 2, 3, 4, 5, 6} steps (= 5, 10, 15, 20, 25, 30 minutes).

For each (model, tau) combination, computes Precision, Recall, F1, and F2
averaged across all patients, using threshold crossings on the predicted
glucose trajectory (threshold = 70 mg/dL, direction = below).

This addresses the core question of Objective 3: how robust is each model's
hypoglycemia event detection across different tolerance windows? A model
that degrades sharply as tau decreases is only useful when ample warning time
is accepted; a robust model maintains recall even at tight tolerances.

Output
------
reports/analysis/event_centric/tau_sweep_summary.csv
    One row per (model, tau). Columns: model, tau_steps, tau_minutes,
    precision, recall, f1, f2.

reports/analysis/event_centric/tau_sweep_per_patient.csv
    Per-patient breakdown for each (model, tau). Useful for identifying
    patients where event detection is consistently poor or good.

References
----------
[2]  Hüni, F. (2023). Predicting events of hypoglycemia: A comparison of
     long short-term memory and graph attention network based approaches.
     Bachelor Thesis, University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ba_baseline.metrics.metrics import event_metrics

# Configuration


THRESHOLD = 70.0  # mg/dL — hypoglycemia onset
TAU_VALUES = [1, 2, 3, 4, 5, 6]  # steps (× 5 min = 5–30 min)
STEP_MINUTES = 5

MODELS = {
    "lstm_baseline":       "reports/results/lstm_60min_traces_all_patients.npz",
    "patchtst_baseline":   "reports/results/patchtst_60min_traces_all_patients.npz",
    "lstm_bounded_lag":    "reports/results/lstm_bounded_lag_traces_all_patients.npz",
    "patchtst_bounded_lag":"reports/results/patchtst_bounded_lag_traces_all_patients.npz",
    "lstm_dtw":            "reports/results/lstm_dtw_traces_all_patients.npz",
    "patchtst_dtw":        "reports/results/patchtst_dtw_traces_all_patients.npz",
    "lstm_multistep":      "reports/results/lstm_multistep_traces_all_patients.npz",
    "patchtst_multistep":  "reports/results/patchtst_multistep_traces_all_patients.npz",
}

OUT_DIR = "reports/analysis/event_centric"


# Helpers


def load_traces(path: str) -> dict:
    """Load npz trace file, return {pid: (true, pred)} dict."""
    d = np.load(path)
    pids = sorted({k.replace("_true", "").replace("_pred", "") for k in d.keys()})
    return {pid: (d[f"{pid}_true"], d[f"{pid}_pred"]) for pid in pids}


def evaluate_model_tau(
    traces: dict, tau: int
) -> tuple[float, float, float, float, list]:
    """
    Compute mean Precision, Recall, F1, F2 across all patients for a given tau.
    Returns (precision, recall, f1, f2, per_patient_rows).
    """
    per_patient = []
    precisions, recalls, f1s, f2s = [], [], [], []

    for pid, (y_true, y_pred) in traces.items():
        m1 = event_metrics(y_true, y_pred, THRESHOLD, tau, beta=1.0)
        m2 = event_metrics(y_true, y_pred, THRESHOLD, tau, beta=2.0)
        precisions.append(m1["precision"])
        recalls.append(m1["recall"])
        f1s.append(m1["fbeta"])
        f2s.append(m2["fbeta"])
        per_patient.append(
            {
                "patient_id": pid,
                "precision": m1["precision"],
                "recall": m1["recall"],
                "f1": m1["fbeta"],
                "f2": m2["fbeta"],
                "tp": m1["tp"],
                "fp": m1["fp"],
                "fn": m1["fn"],
            }
        )

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(f1s)),
        float(np.mean(f2s)),
        per_patient,
    )


# Main


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []
    per_patient_rows = []

    for model_name, npz_path in MODELS.items():
        if not os.path.exists(npz_path):
            print(f"  SKIP {model_name} — file not found: {npz_path}")
            continue

        traces = load_traces(npz_path)
        print(f"\n{model_name} ({len(traces)} patients)")

        for tau in TAU_VALUES:
            tau_min = tau * STEP_MINUTES
            prec, rec, f1, f2, pp = evaluate_model_tau(traces, tau)

            print(
                f"  tau={tau_min:2d}min  precision={prec:.4f}  recall={rec:.4f}"
                f"  F1={f1:.4f}  F2={f2:.4f}"
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "tau_steps": tau,
                    "tau_minutes": tau_min,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "f2": round(f2, 4),
                }
            )

            for row in pp:
                per_patient_rows.append(
                    {
                        "model": model_name,
                        "tau_steps": tau,
                        "tau_minutes": tau_min,
                        **row,
                    }
                )

    # Write summary CSV
    summary_path = os.path.join(OUT_DIR, "tau_sweep_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tau_steps",
                "tau_minutes",
                "precision",
                "recall",
                "f1",
                "f2",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")

    # Write per-patient CSV
    pp_path = os.path.join(OUT_DIR, "tau_sweep_per_patient.csv")
    with open(pp_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tau_steps",
                "tau_minutes",
                "patient_id",
                "precision",
                "recall",
                "f1",
                "f2",
                "tp",
                "fp",
                "fn",
            ],
        )
        writer.writeheader()
        writer.writerows(per_patient_rows)
    print(f"Per-patient written to {pp_path}")


if __name__ == "__main__":
    main()
