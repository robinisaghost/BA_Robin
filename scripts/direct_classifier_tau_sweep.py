"""
Objective 3: tau-Sweep for the Direct Binary Event Classifiers

Analogous to tau_sweep_event_detection.py for the regression models, but for
the LSTM and PatchTST binary event classifiers that predict directly whether a
hypoglycemia event will occur in the next 60 minutes.

Whereas the regression tau-sweep matches *predicted crossing times* against
*true crossing times* with a positional tolerance tau, the direct classifier
emits a window-level binary decision ("event in next H steps").  Here, tau
controls the *required lead time* of that window:

  y_true_tau[i] = 1  if min(test_s[i+lookback : i+lookback+tau]) < 70
  y_pred[i]         = model output trained with tau_train = 12 (60 min)

As tau shrinks from 12 (60 min, training setting) down to 1 (5 min), we ask:
"How well does the 60-minute alarm translate to alarms about events that are
only tau minutes away?"  A perfect score at small tau means the model tends to
fire just before imminent events; degrading precision means many alarms warn of
events only late in the 60-minute window.

This is the companion metric to the lead-time analysis in
analyze_event_lead_time.py.

Output
------
reports/analysis/event_centric/direct_clf_tau_sweep_summary.csv
    One row per (model, tau). Columns: model, tau_steps, tau_minutes,
    precision, recall, f1, f2.

reports/analysis/event_centric/direct_clf_tau_sweep_per_patient.csv
    Per-patient breakdown for each (model, tau).

References
----------
[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[8]  Huni, F. (2023). Predicting events of hypoglycemia: A comparison of
     long short-term memory and graph attention network based approaches.
     Bachelor Thesis, University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series

HYPO_THRESH   = 70.0
LOOKBACK      = 24
TAU_TRAIN     = 12        # horizon used during training (60 min)
CLF_THRESHOLD = 0.5
STEP_MINUTES  = 5

TAU_VALUES = list(range(1, TAU_TRAIN + 1))   # 1..12 steps (5..60 min)

MODELS = {
    "lstm_event":     "reports/results/lstm_event_traces_all_patients.npz",
    "patchtst_event": "reports/results/patchtst_event_traces_all_patients.npz",
}

OUT_DIR = "reports/analysis/event_centric"


def fbeta(precision: float, recall: float, beta: float) -> float:
    denom = beta ** 2 * precision + recall
    return (1 + beta ** 2) * precision * recall / denom if denom > 0 else 0.0


def evaluate_tau(
    y_pred_binary: np.ndarray,
    test_s: np.ndarray,
    tau: int,
) -> dict:
    """
    Re-evaluate binary predictions against labels derived with window size tau.

    y_pred_binary : (N,) bool — fixed model output
    test_s        : full test series for the patient
    tau           : steps in [1, TAU_TRAIN] to check for a true event
    """
    tp = fp = fn = 0
    n = len(y_pred_binary)

    for i in range(n):
        start = i + LOOKBACK
        end   = start + tau
        if end > len(test_s):
            continue

        true_pos = float(test_s[start:end].min()) < HYPO_THRESH
        pred_pos = bool(y_pred_binary[i])

        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif not pred_pos and true_pos:
            fn += 1

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec,
        "recall":    rec,
        "f1":  fbeta(prec, rec, 1.0),
        "f2":  fbeta(prec, rec, 2.0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    d = load_patient_series("data/raw/all_cgm.csv")
    _, _, test_series = temporal_split_series(d, train_ratio=0.6, val_ratio=0.2)

    summary_rows     = []
    per_patient_rows = []

    for model_name, npz_path in MODELS.items():
        if not os.path.exists(npz_path):
            print(f"  SKIP {model_name} — file not found: {npz_path}")
            continue

        data = np.load(npz_path)
        pids = sorted(
            {k.replace("_true", "").replace("_pred", "") for k in data.keys()}
        )
        print(f"\n{model_name} ({len(pids)} patients)")

        for tau in TAU_VALUES:
            tau_min = tau * STEP_MINUTES
            precs, recs, f1s, f2s = [], [], [], []
            pp_rows = []

            for pid in pids:
                if pid not in test_series:
                    continue

                test_s        = test_series[pid]
                y_pred        = data[f"{pid}_pred"]
                y_pred_binary = (y_pred >= CLF_THRESHOLD)

                m = evaluate_tau(y_pred_binary, test_s, tau)
                precs.append(m["precision"])
                recs.append(m["recall"])
                f1s.append(m["f1"])
                f2s.append(m["f2"])
                pp_rows.append({"patient_id": pid, **m})

            prec = float(np.mean(precs))
            rec  = float(np.mean(recs))
            f1   = float(np.mean(f1s))
            f2   = float(np.mean(f2s))

            print(
                f"  tau={tau_min:2d}min  precision={prec:.4f}  recall={rec:.4f}"
                f"  F1={f1:.4f}  F2={f2:.4f}"
            )

            summary_rows.append(
                {
                    "model":       model_name,
                    "tau_steps":   tau,
                    "tau_minutes": tau_min,
                    "precision":   round(prec, 4),
                    "recall":      round(rec,  4),
                    "f1":          round(f1,   4),
                    "f2":          round(f2,   4),
                }
            )
            for row in pp_rows:
                per_patient_rows.append(
                    {
                        "model":       model_name,
                        "tau_steps":   tau,
                        "tau_minutes": tau_min,
                        **row,
                    }
                )

    # Summary CSV
    summary_path = os.path.join(OUT_DIR, "direct_clf_tau_sweep_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "tau_steps", "tau_minutes",
                "precision", "recall", "f1", "f2",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")

    # Per-patient CSV
    pp_path = os.path.join(OUT_DIR, "direct_clf_tau_sweep_per_patient.csv")
    with open(pp_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "tau_steps", "tau_minutes", "patient_id",
                "precision", "recall", "f1", "f2", "tp", "fp", "fn",
            ],
        )
        writer.writeheader()
        writer.writerows(per_patient_rows)
    print(f"Per-patient written to {pp_path}")


if __name__ == "__main__":
    main()
