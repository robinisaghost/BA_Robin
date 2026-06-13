"""
Objective 3: Lead-Time Analysis for Binary Event Classifiers

For every True Positive prediction made by the LSTM or PatchTST binary event
classifier, this script measures how many minutes into the 60-minute prediction
window the *first* actual hypoglycemia crossing occurs.

Background
----------
The binary classifiers are trained with a label y[i] = 1 if
min(test_s[i+lookback : i+lookback+horizon]) < 70 mg/dL.  A True Positive (TP)
at window index i means the model correctly predicted an event somewhere in
the next 60 minutes, but does NOT tell us *when* within those 60 minutes the
event happens.  An alarm that fires just as glucose is already below 70 is very
different from one that fires 50 minutes before onset.

This script answers: "Given a correct alarm, how far in advance does the model
actually warn the patient?"

Methodology
-----------
For each TP (y_true[i] = 1 AND y_pred[i] >= CLF_THRESHOLD):
  1. Locate the first step k in [0, horizon) where test_s[i+lookback+k] < 70.
  2. time_to_event = k * 5 minutes.

Aggregates: mean, median, and per-bucket count over {0, 5, 10, ..., 55} min.

Output
------
reports/analysis/event_centric/event_lead_time.json
    Per-model statistics: n_tp, mean_minutes_to_event, median_minutes_to_event,
    distribution_minutes (count per 5-min bucket).

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

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series

HYPO_THRESH = 70.0
LOOKBACK = 24
HORIZON = 12       # 60 min
CLF_THRESHOLD = 0.5
STEP_MINUTES = 5

MODELS = {
    "lstm_event":     "reports/results/lstm_event_traces_all_patients.npz",
    "patchtst_event": "reports/results/patchtst_event_traces_all_patients.npz",
}

OUT_DIR = "reports/analysis/event_centric"


def first_below_step(window: np.ndarray, threshold: float):
    """Return index of first step strictly below threshold, or None."""
    for k, v in enumerate(window):
        if v < threshold:
            return k
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    d = load_patient_series("data/raw/all_cgm.csv")
    _, _, test_series = temporal_split_series(d, train_ratio=0.6, val_ratio=0.2)

    results = {}

    for model_name, npz_path in MODELS.items():
        if not os.path.exists(npz_path):
            print(f"  SKIP {model_name} — file not found: {npz_path}")
            continue

        data = np.load(npz_path)
        pids = sorted(
            {k.replace("_true", "").replace("_pred", "") for k in data.keys()}
        )

        all_lead_times = []

        for pid in pids:
            if pid not in test_series:
                continue
            test_s = test_series[pid]
            y_true = data[f"{pid}_true"]
            y_pred = data[f"{pid}_pred"]

            n = len(y_true)
            for i in range(n):
                if y_true[i] < 0.5 or y_pred[i] < CLF_THRESHOLD:
                    continue  # only True Positives

                start = i + LOOKBACK
                end = start + HORIZON
                if end > len(test_s):
                    continue

                window = test_s[start:end]
                k = first_below_step(window, HYPO_THRESH)
                if k is not None:
                    all_lead_times.append(k * STEP_MINUTES)

        if not all_lead_times:
            print(f"  {model_name}: no True Positives found")
            results[model_name] = {"n_tp": 0}
            continue

        buckets = {
            str(t): int(sum(1 for x in all_lead_times if x == t))
            for t in range(0, HORIZON * STEP_MINUTES, STEP_MINUTES)
        }
        mean_lt   = float(np.mean(all_lead_times))
        median_lt = float(np.median(all_lead_times))

        results[model_name] = {
            "n_tp": len(all_lead_times),
            "mean_minutes_to_event":   round(mean_lt, 2),
            "median_minutes_to_event": round(median_lt, 2),
            "distribution_minutes": buckets,
        }

        print(
            f"  {model_name}: n_TP={len(all_lead_times)}, "
            f"mean={mean_lt:.1f} min, median={median_lt:.1f} min"
        )
        dist_str = "  ".join(f"{t}min:{buckets[str(t)]}" for t in range(0, 60, 5))
        print(f"    distribution: {dist_str}")

    out_path = os.path.join(OUT_DIR, "event_lead_time.json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
