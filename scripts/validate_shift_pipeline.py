"""
Pipeline Validation: Synthetic Shift Test

Verifies that the shift measurement (best_lag_rmse) and tolerant event
evaluation (event_metrics with tolerance tau) both behave correctly on
synthetic data where the true shift is known in advance.

Two tests are run:

Test 1 — Shift measurement.
  A real glucose series is used as y_true. Synthetic predictions are
  constructed by shifting y_true by known amounts (5, 15, 30, 45, 60 min).
  best_lag_rmse must recover the original shift within ±1 step for each case.

Test 2 — Tolerant event detection.
  A synthetic glucose series with known threshold crossings is constructed.
  A shifted prediction is created and event_metrics is called with tau = 3
  (15 min tolerance). The test checks whether TP, FP, FN counts match the
  expected values for shifts that fall inside and outside the tolerance window.

Both tests pass if all assertions hold. The script exits with code 0 on
success and prints the results to stdout.

This script validates the pipeline described in the thesis proposal
(Section 7, Weeks 4-6: protocol and pipeline phase).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ba_baseline.metrics.metrics import best_lag_rmse, event_metrics, shift_1d
from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import temporal_split_series

# Configuration
MAX_LAG = 12          # search window for lag recovery: +-60 min
TAU = 3               # event tolerance: 3 steps = 15 min
THRESHOLD = 70.0      # mg/dL
STEP_MIN = 5          # minutes per step
TOLERANCE_STEPS = 1   # allowed error in lag recovery (+-1 step = +-5 min)


def run_test1_lag_recovery():
    """
    Test 1: best_lag_rmse recovers a known synthetic shift.

    For each shift k_late in {1, 3, 6, 9, 12} steps, a synthetic prediction
    is created by shifting y_true right by k_late steps. best_lag_rmse should
    return a correction lag of approximately -k_late (advance the late
    prediction).
    """
    print("=" * 60)
    print("Test 1: Lag recovery on synthetic shifts")
    print("=" * 60)

    d = load_patient_series("data/raw/all_cgm.csv")
    _, _, test_series = temporal_split_series(d, train_ratio=0.6, val_ratio=0.2)
    y_true = test_series["85202"].astype(np.float32)

    shifts_steps = [1, 3, 6, 9, 12]
    all_passed = True

    print(f"{'Shift (min)':>12} {'Expected lag (min)':>19} {'Recovered lag (min)':>20} {'Status':>8}")
    print("-" * 63)

    for k_late in shifts_steps:
        y_pred = shift_1d(y_true, k_late)
        mask = ~np.isnan(y_pred)
        y_t = y_true[mask]
        y_p = y_pred[mask]

        recovered = best_lag_rmse(y_t, y_p, max_lag=MAX_LAG)
        expected = -k_late
        error = abs(recovered - expected)
        passed = error <= TOLERANCE_STEPS
        all_passed = all_passed and passed

        shift_min = k_late * STEP_MIN
        expected_min = expected * STEP_MIN
        recovered_min = recovered * STEP_MIN
        status = "PASS" if passed else "FAIL"
        print(
            f"{shift_min:>12d} {expected_min:>19d} {recovered_min:>20d} {status:>8}"
        )

    print()
    return all_passed


def run_test2_event_metrics():
    """
    Test 2: event_metrics gives correct TP/FP/FN for known shifts.

    A synthetic series with one known hypoglycemia event is constructed.
    Predictions are shifted by amounts both inside and outside tau = 3 steps.
    Expected results:
      shift <= tau steps: TP=1, FP=0, FN=0
      shift > tau steps:  TP=0, FP=1, FN=1
    """
    print("=" * 60)
    print("Test 2: Tolerant event detection with tau = {} steps ({} min)".format(
        TAU, TAU * STEP_MIN))
    print("=" * 60)

    # Build a synthetic series: glucose = 100 everywhere, drops below 70
    # at index 50 (a single event crossing)
    n = 200
    y_true = np.full(n, 100.0, dtype=np.float32)
    y_true[50:60] = 60.0   # drop below 70: crossing at index 50

    # Expected: one true event at index 50
    shifts_steps = [0, 1, 2, 3, 4, 6, 9]
    all_passed = True

    print(f"{'Shift (min)':>12} {'Expected TP':>12} {'Expected FP':>12} {'Expected FN':>12} {'Got TP':>8} {'Got FP':>8} {'Got FN':>8} {'Status':>8}")
    print("-" * 96)

    for k in shifts_steps:
        # Shift the prediction: the event appears k steps later in pred
        y_pred = shift_1d(y_true, k)
        y_pred_clean = np.where(np.isnan(y_pred), 100.0, y_pred)

        m = event_metrics(y_true, y_pred_clean, THRESHOLD, tol=TAU, beta=1.0)

        if k <= TAU:
            exp_tp, exp_fp, exp_fn = 1, 0, 0
        else:
            exp_tp, exp_fp, exp_fn = 0, 1, 1

        passed = (m["tp"] == exp_tp and m["fp"] == exp_fp and m["fn"] == exp_fn)
        all_passed = all_passed and passed

        shift_min = k * STEP_MIN
        status = "PASS" if passed else "FAIL"
        print(
            f"{shift_min:>12d} {exp_tp:>12d} {exp_fp:>12d} {exp_fn:>12d} "
            f"{m['tp']:>8d} {m['fp']:>8d} {m['fn']:>8d} {status:>8}"
        )

    print()
    return all_passed


def main():
    t1 = run_test1_lag_recovery()
    t2 = run_test2_event_metrics()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (lag recovery):        {'PASS' if t1 else 'FAIL'}")
    print(f"  Test 2 (event detection):     {'PASS' if t2 else 'FAIL'}")
    print()

    if t1 and t2:
        print("All tests passed. Pipeline validated.")
        sys.exit(0)
    else:
        print("One or more tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
