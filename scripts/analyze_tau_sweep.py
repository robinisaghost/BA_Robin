"""
Objective 3a: tau-sweep analysis for hypoglycemia event detection.

For each tolerance tau in {0, 1, ..., 12} steps, compute macro-averaged
Precision, Recall, F1, and F2 across all 36 patients using the baseline
LSTM and PatchTST predictions from the 60-min forecasting model.

Outputs:
    reports/results/tau_sweep_lstm.json
    reports/results/tau_sweep_patchtst.json
    reports/figures/tau_sweep/tau_sweep_comparison.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ba_baseline.metrics.metrics import event_metrics

THRESHOLD = 70.0
TAU_RANGE = list(range(0, 13))
TRACES_LSTM = Path("reports/results/lstm_60min_traces_all_patients.npz")
TRACES_PATCHTST = Path("reports/results/patchtst_60min_traces_all_patients.npz")
OUT_JSON_LSTM = Path("reports/results/tau_sweep_lstm.json")
OUT_JSON_PATCHTST = Path("reports/results/tau_sweep_patchtst.json")
OUT_FIGURE = Path("reports/figures/tau_sweep/tau_sweep_comparison.png")


def sweep_model(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    patient_ids = sorted(set(k.rsplit("_", 1)[0] for k in data.keys()))

    results = {}
    for tau in TAU_RANGE:
        precisions, recalls, f1s, f2s = [], [], [], []
        for pid in patient_ids:
            true_series = data[f"{pid}_true"].astype(np.float64)
            pred_series = data[f"{pid}_pred"].astype(np.float64)
            m1 = event_metrics(true_series, pred_series, THRESHOLD, tau, beta=1.0)
            m2 = event_metrics(true_series, pred_series, THRESHOLD, tau, beta=2.0)
            precisions.append(m1["precision"])
            recalls.append(m1["recall"])
            f1s.append(m1["fbeta"])
            f2s.append(m2["fbeta"])
        results[tau] = {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "f2": float(np.mean(f2s)),
        }
    return results


def save_json(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {path}")


def plot_comparison(lstm_results: dict, patchtst_results: dict, out_path: Path):
    taus = TAU_RANGE
    lstm_f1 = [lstm_results[t]["f1"] for t in taus]
    patchtst_f1 = [patchtst_results[t]["f1"] for t in taus]
    lstm_prec = [lstm_results[t]["precision"] for t in taus]
    patchtst_prec = [patchtst_results[t]["precision"] for t in taus]
    lstm_rec = [lstm_results[t]["recall"] for t in taus]
    patchtst_rec = [patchtst_results[t]["recall"] for t in taus]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, lstm_vals, ptst_vals, ylabel in zip(
        axes,
        [lstm_f1, lstm_prec, lstm_rec],
        [patchtst_f1, patchtst_prec, patchtst_rec],
        ["F1", "Precision", "Recall"],
    ):
        ax.plot(taus, lstm_vals, marker="o", label="LSTM", color="#1f77b4")
        ax.plot(taus, ptst_vals, marker="s", label="PatchTST", color="#ff7f0e")
        ax.set_xlabel("Tolerance $\\tau$ in 5 min steps")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. τ")
        ax.set_xticks(taus)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Running tau-sweep for LSTM baseline...")
    lstm_results = sweep_model(TRACES_LSTM)
    save_json(lstm_results, OUT_JSON_LSTM)

    print("Running tau-sweep for PatchTST baseline...")
    patchtst_results = sweep_model(TRACES_PATCHTST)
    save_json(patchtst_results, OUT_JSON_PATCHTST)

    print("Generating comparison plot...")
    plot_comparison(lstm_results, patchtst_results, OUT_FIGURE)

    print("\ntau-sweep results summary:")
    print(f"{'tau':>4}  {'LSTM F1':>8}  {'PTST F1':>8}  {'LSTM Rec':>9}  {'PTST Rec':>9}")
    for t in TAU_RANGE:
        print(
            f"{t:>4}  {lstm_results[t]['f1']:>8.4f}  {patchtst_results[t]['f1']:>8.4f}"
            f"  {lstm_results[t]['recall']:>9.4f}  {patchtst_results[t]['recall']:>9.4f}"
        )
    print("\nDone.")
