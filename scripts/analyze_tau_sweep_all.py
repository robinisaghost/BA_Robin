"""
Extended tau-sweep analysis for all 8 training configurations.

Outputs:
    reports/results/tau_sweep_all.json          -- all configs, all tau
    thesis/img/tau_sweep_comparison.png          -- Figure 4.5 (2-panel, all variants)
    thesis/img/tau_sweep_appendix_lstm.png       -- Appendix: LSTM F1/P/R curves
    thesis/img/tau_sweep_appendix_patchtst.png   -- Appendix: PatchTST F1/P/R curves
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ba_baseline.metrics.metrics import event_metrics

# seaborn-darkgrid theme (shared across all thesis data figures)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#EAEAF2",
    "axes.edgecolor": "white",
    "axes.linewidth": 0.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

THRESHOLD = 70.0
TAU_RANGE  = list(range(0, 13))
TAU_MINS   = [t * 5 for t in TAU_RANGE]   # in minutes

CONFIGS = [
    ("LSTM MSE",          "reports/results/lstm_60min_traces_all_patients.npz"),
    ("LSTM Bounded-Lag",  "reports/results/lstm_bounded_lag_traces_all_patients.npz"),
    ("LSTM Soft-DTW",     "reports/results/lstm_dtw_traces_all_patients.npz"),
    ("LSTM Multi-step",   "reports/results/lstm_multistep_traces_all_patients.npz"),
    ("PatchTST MSE",          "reports/results/patchtst_60min_traces_all_patients.npz"),
    ("PatchTST Bounded-Lag",  "reports/results/patchtst_bounded_lag_traces_all_patients.npz"),
    ("PatchTST Soft-DTW",     "reports/results/patchtst_dtw_traces_all_patients.npz"),
    ("PatchTST Multi-step",   "reports/results/patchtst_multistep_traces_all_patients.npz"),
]

IMG_DIR = Path("thesis/img")

STYLES = {
    "MSE":         dict(color="#1565C0", linestyle="-",  linewidth=1.6),
    "Bounded-Lag": dict(color="#E65100", linestyle="--", linewidth=1.6),
    "Soft-DTW":    dict(color="#00695C", linestyle="-.", linewidth=1.6),
    "Multi-step":  dict(color="#6A1B9A", linestyle=":",  linewidth=2.0),
}


def sweep_model(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    patient_ids = sorted(set(k.rsplit("_", 1)[0] for k in data.keys()))
    results = {}
    for tau in TAU_RANGE:
        precisions, recalls, f1s, f2s = [], [], [], []
        for pid in patient_ids:
            true_s = data[f"{pid}_true"].astype(np.float64)
            pred_s = data[f"{pid}_pred"].astype(np.float64)
            m1 = event_metrics(true_s, pred_s, THRESHOLD, tau, beta=1.0)
            m2 = event_metrics(true_s, pred_s, THRESHOLD, tau, beta=2.0)
            precisions.append(m1["precision"])
            recalls.append(m1["recall"])
            f1s.append(m1["fbeta"])
            f2s.append(m2["fbeta"])
        results[tau] = {
            "precision": float(np.mean(precisions)),
            "recall":    float(np.mean(recalls)),
            "f1":        float(np.mean(f1s)),
            "f2":        float(np.mean(f2s)),
        }
    return results


def make_figure45(all_results: dict):
    """Figure 4.5: 2-panel F1 tau sweep (LSTM left, PatchTST right), all 4 strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    for ax, arch in zip(axes, ["LSTM", "PatchTST"]):
        for suffix, style in STYLES.items():
            key = f"{arch} {suffix}"
            if key not in all_results:
                continue
            f1s = [all_results[key][t]["f1"] for t in TAU_RANGE]
            ax.plot(TAU_MINS, f1s, label=suffix, **style)
        ax.set_xlabel("Tolerance $\\tau$ (minutes)")
        ax.set_ylabel("F1")
        ax.set_title(arch)
        ax.set_xticks(TAU_MINS)
        ax.tick_params(axis='x', labelsize=7)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)
    fig.suptitle(r"Forecast-derived event detection: F1 vs. tolerance $\tau$",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "tau_sweep_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved tau_sweep_comparison.png")


def make_appendix_figure(all_results: dict, arch: str, fname: str):
    """Appendix: 1x3 panel (Precision, Recall, F2) for one architecture, all strategies.

    The figure size is kept close to the LaTeX text width (it is included at
    \\textwidth) so the fonts are not shrunk on the page; font sizes are set
    explicitly for comfortable print reading.
    """
    metrics = [("Precision", "precision"), ("Recall", "recall"), ("F2", "f2")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.9), sharey=False)

    for ax, (metric_label, metric_key) in zip(axes, metrics):
        for suffix, style in STYLES.items():
            key = f"{arch} {suffix}"
            if key not in all_results:
                continue
            vals = [all_results[key][t][metric_key] for t in TAU_RANGE]
            ax.plot(TAU_MINS, vals, label=suffix, **style)
        ax.set_xlabel("Tolerance $\\tau$ (min)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12)
        ax.set_xticks([0, 20, 40, 60])
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def print_latex_table(all_results: dict, arch: str, metric: str = "f1"):
    """Print a LaTeX table snippet for one architecture and one metric."""
    suffixes = ["MSE", "Bounded-Lag", "Soft-DTW", "Multi-step"]
    taus_to_show = [0, 2, 4, 6, 8, 10, 12]  # every 2 steps = every 10 min
    print(f"\n% {arch} — {metric.upper()} tau sweep")
    print(f"$\\tau$ (min) & " + " & ".join(f"{arch} {s}" for s in suffixes) + " \\\\")
    for t in taus_to_show:
        row = f"{TAU_MINS[t]}"
        for s in suffixes:
            key = f"{arch} {s}"
            val = all_results[key][t][metric] if key in all_results else float('nan')
            row += f" & {val:.3f}"
        row += " \\\\"
        print(row)


if __name__ == "__main__":
    all_results = {}
    for name, path in CONFIGS:
        print(f"Computing tau sweep for {name}...")
        all_results[name] = sweep_model(path)

    out = Path("reports/results/tau_sweep_all.json")
    with open(out, "w") as f:
        json.dump({k: {str(tau): v for tau, v in res.items()}
                   for k, res in all_results.items()}, f, indent=2)
    print(f"Saved {out}")

    make_figure45(all_results)
    make_appendix_figure(all_results, "LSTM",     "tau_sweep_appendix_lstm.png")
    make_appendix_figure(all_results, "PatchTST", "tau_sweep_appendix_patchtst.png")

    for arch in ["LSTM", "PatchTST"]:
        print_latex_table(all_results, arch, "f1")
    print("\nDone.")
