"""
Generate all figures needed for Chapter 4 and Appendix B of the thesis.

Outputs (saved to thesis/):
  lag_distribution.png         -- Section 4.1: per-patient best-lag histogram
  dtw_vs_mse_trace.png         -- Section 4.2: MSE vs Soft-DTW trajectory comparison
  multistep_vs_baseline_trace.png -- Section 4.3: baseline vs multi-step phase alignment
  lead_time_distribution.png   -- Section 4.4: lead-time histogram for event classifiers
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, "src")
from ba_baseline.metrics.metrics import best_lag_rmse

# ── shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Seaborn-darkgrid trajectory style, shared with the appendix patient panels.
# Variant -> colour is fixed across every trajectory figure (Okabe-Ito palette).
THRESHOLD_COLOR = "#C2185B"
VAR_COLOR = {
    "MSE (baseline)":         "#0072B2",
    "Single-step (baseline)": "#0072B2",
    "Prediction":             "#0072B2",
    "Bounded-Lag":            "#E69F00",
    "Soft-DTW":               "#009E73",
    "Multi-step":             "#CC79A7",
}


def style_trajectory_axis(ax, t):
    """Apply the seaborn-darkgrid look (grey panel, white major grid, no spines)
    and mark the 5-minute samples with small minor ticks on the time axis."""
    ax.set_facecolor("#EAEAF2")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="white", linewidth=1.0)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="major", length=0, colors="#555555")
    ax.tick_params(which="minor", axis="x", length=3, color="#9aa0b5")
    ax.set_xlim(t[0], t[-1])


HYPO_LINE  = 70.0   # mg/dL clinical threshold
OUT_DIR    = "thesis/img"
DEMO_PID   = "85202"  # tuning patient (trajectory overview)
HYPO_PID   = "85102"  # patient with early hypo event — best for zoomed comparison plots
N_WINDOW   = 120      # 10 h for zoomed trajectory plots
WIN_START  = 47       # start index for HYPO_PID: includes lead-up to hypo at step 87


# ═══════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════

def load_traces(path: str) -> dict:
    d = np.load(path)
    pids = sorted(set(k.split("_")[0] for k in d.keys()))
    return {pid: (d[f"{pid}_true"], d[f"{pid}_pred"]) for pid in pids}


def find_interesting_window(true: np.ndarray, length: int = N_WINDOW) -> int:
    """Return start index of a window that includes a descent toward hypoglycaemia."""
    # look for a window where min(true) < 85 mg/dL  (near-hypo or hypo)
    best_start = 0
    best_score = np.inf
    for start in range(0, len(true) - length, 10):
        window = true[start: start + length]
        mn = window.min()
        if mn < best_score:
            best_score = mn
            best_start = start
        if mn < HYPO_LINE:   # actual hypoglycaemia – take it and stop
            return start
    return best_start


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – lag_distribution.png
# ═══════════════════════════════════════════════════════════════════════════

def make_lag_distribution():
    print("Generating lag_distribution.png ...")
    lstm  = load_traces("reports/results/lstm_60min_traces_all_patients.npz")
    ptst  = load_traces("reports/results/patchtst_60min_traces_all_patients.npz")

    def compute_lags(traces):
        lags = []
        for pid, (t, p) in traces.items():
            lag = best_lag_rmse(t, p, max_lag=12)
            lags.append(abs(lag) * 5)
        return np.array(lags)

    lags_lstm = compute_lags(lstm)
    lags_ptst = compute_lags(ptst)

    labels = ["LSTM", "PatchTST"]
    data   = [lags_lstm, lags_ptst]
    colors = ["#4878CF", "#DD8452"]

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=""))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    for i, (lags, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(lags))
        ax.scatter(i + jitter, lags, color=color, s=18, alpha=0.85,
                   zorder=3, edgecolors="white", linewidths=0.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction delay $\\Delta^*$ (minutes)")
    ax.set_ylim(-2, 68)
    ax.set_title("Per-patient prediction delay $\\Delta^*$ across the cohort", pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lag_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  LSTM  mean={np.mean(lags_lstm):.1f} min  median={np.median(lags_lstm):.1f} min")
    print(f"  PatchTST mean={np.mean(lags_ptst):.1f} min  median={np.median(lags_ptst):.1f} min")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – dtw_vs_mse_trace.png
# ═══════════════════════════════════════════════════════════════════════════

def make_obj1_all_predictions():
    print("Generating obj1_all_predictions.png ...")
    lstm_mse = load_traces("reports/results/lstm_60min_traces_all_patients.npz")
    lstm_bl  = load_traces("reports/results/lstm_bounded_lag_traces_all_patients.npz")
    lstm_dtw = load_traces("reports/results/lstm_dtw_traces_all_patients.npz")
    ptst_mse = load_traces("reports/results/patchtst_60min_traces_all_patients.npz")
    ptst_bl  = load_traces("reports/results/patchtst_bounded_lag_traces_all_patients.npz")
    ptst_dtw = load_traces("reports/results/patchtst_dtw_traces_all_patients.npz")

    true_w = lstm_mse[HYPO_PID][0][WIN_START: WIN_START + N_WINDOW]
    t      = np.arange(N_WINDOW) * 5

    arch_preds = [
        ("LSTM", {
            "MSE (baseline)": lstm_mse[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Bounded-Lag":    lstm_bl [HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Soft-DTW":       lstm_dtw[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
        }),
        ("PatchTST", {
            "MSE (baseline)": ptst_mse[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Bounded-Lag":    ptst_bl [HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Soft-DTW":       ptst_dtw[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
        }),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=False, sharex=True)

    for ax, (arch, preds) in zip(axes, arch_preds):
        all_vals = [true_w] + list(preds.values())
        y_lo = min(v.min() for v in all_vals) - 8
        y_hi = max(v.max() for v in all_vals) + 12

        ax.axhline(HYPO_LINE, color=THRESHOLD_COLOR, linewidth=1.2,
                   linestyle=(0, (5, 3)), label="70 mg/dL threshold", zorder=2)
        ax.plot(t, true_w, color="black", linewidth=2.0, label="Ground truth", zorder=4)
        for label, pred in preds.items():
            ax.plot(t, pred, label=label, color=VAR_COLOR[label],
                    linewidth=1.4, zorder=3)

        ax.set_xlabel("Time (minutes, relative)")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(arch)
        style_trajectory_axis(ax, t)

    axes[0].set_ylabel("Blood glucose (mg/dL)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle("Objective 1: Offset-Aware Loss Functions", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/obj1_all_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – multistep_vs_baseline_trace.png
# ═══════════════════════════════════════════════════════════════════════════

def make_multistep_vs_baseline_trace():
    print("Generating multistep_vs_baseline_trace.png ...")
    lstm_base = load_traces("reports/results/lstm_60min_traces_all_patients.npz")
    lstm_ms   = load_traces("reports/results/lstm_multistep_traces_all_patients.npz")
    ptst_base = load_traces("reports/results/patchtst_60min_traces_all_patients.npz")
    ptst_ms   = load_traces("reports/results/patchtst_multistep_traces_all_patients.npz")

    true_w = lstm_base[HYPO_PID][0][WIN_START: WIN_START + N_WINDOW]
    t      = np.arange(N_WINDOW) * 5

    arch_preds = [
        ("LSTM", {
            "Single-step (baseline)": lstm_base[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Multi-step":             lstm_ms  [HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
        }),
        ("PatchTST", {
            "Single-step (baseline)": ptst_base[HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
            "Multi-step":             ptst_ms  [HYPO_PID][1][WIN_START: WIN_START + N_WINDOW],
        }),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=False, sharex=True)

    for ax, (arch, preds) in zip(axes, arch_preds):
        all_vals = [true_w] + list(preds.values())
        y_lo = min(v.min() for v in all_vals) - 8
        y_hi = max(v.max() for v in all_vals) + 12

        ax.axhline(HYPO_LINE, color=THRESHOLD_COLOR, linewidth=1.2,
                   linestyle=(0, (5, 3)), label="70 mg/dL threshold", zorder=2)
        ax.plot(t, true_w, color="black", linewidth=2.0, label="Ground truth", zorder=4)
        for label, pred in preds.items():
            ax.plot(t, pred, label=label, color=VAR_COLOR[label],
                    linewidth=1.4, zorder=3)

        ax.set_xlabel("Time (minutes, relative)")
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(arch)
        style_trajectory_axis(ax, t)

    axes[0].set_ylabel("Blood glucose (mg/dL)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle("Single-step vs. Multi-step supervision", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/multistep_vs_baseline_trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – lead_time_distribution.png
# ═══════════════════════════════════════════════════════════════════════════

def make_lead_time_distribution():
    print("Generating lead_time_distribution.png ...")
    with open("reports/analysis/event_centric/event_lead_time.json", "r") as f:
        data = json.load(f)

    minutes   = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    bar_width = 4.0
    offsets   = [-2.2, +2.2]
    colors    = ["#4878CF", "#DD8452"]
    models    = ["lstm_event", "patchtst_event"]
    labels    = ["LSTM", "PatchTST"]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))

    for i, (model, label, color, off) in enumerate(zip(models, labels, colors, offsets)):
        info   = data[model]
        counts = [info["distribution_minutes"][str(m)] for m in minutes]
        total  = info["n_tp"]
        pct    = [100 * c / total for c in counts]
        x_pos  = np.array(minutes) + off
        ax.bar(x_pos, pct, width=bar_width, color=color, alpha=0.85,
               label=f"{label}  (n={total:,},  med={info['median_minutes_to_event']:.0f} min)")

    ax.set_xlabel("Minutes until first hypoglycaemic step (lead time)")
    ax.set_ylabel("Percentage of true positives (%)")
    ax.set_xticks(minutes)
    ax.legend(loc="upper right")
    ax.set_title("Lead-time distribution of true-positive event detections", pad=6)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/lead_time_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 0 – patient_???.png  (baseline trajectory overview, Figure 4.1)
# ═══════════════════════════════════════════════════════════════════════════

def make_baseline_trajectory_comparison():
    print("Generating baseline trajectory comparison (Figure 4.1) ...")
    lstm = load_traces("reports/results/lstm_60min_traces_all_patients.npz")
    ptst = load_traces("reports/results/patchtst_60min_traces_all_patients.npz")

    # Use patient 85102 at WIN_START=47: confirmed hypo event around step 87
    pid   = HYPO_PID
    start = WIN_START

    true_w      = lstm[pid][0][start: start + N_WINDOW]
    pred_lstm_w = lstm[pid][1][start: start + N_WINDOW]
    pred_ptst_w = ptst[pid][1][start: start + N_WINDOW]
    t    = np.arange(N_WINDOW) * 5
    y_lo = min(true_w.min(), pred_lstm_w.min(), pred_ptst_w.min()) - 8

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True, sharex=True)

    for ax, pred, title in zip(axes,
                                [pred_lstm_w, pred_ptst_w],
                                ["LSTM", "PatchTST"]):
        ax.axhline(HYPO_LINE, color=THRESHOLD_COLOR, linewidth=1.2,
                   linestyle=(0, (5, 3)), label="70 mg/dL threshold", zorder=2)
        ax.plot(t, true_w, color="black", linewidth=2.0, label="Ground truth", zorder=4)
        ax.plot(t, pred, color=VAR_COLOR["Prediction"], linewidth=1.4,
                label="Prediction", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Time (minutes, relative)")
        ax.set_ylim(y_lo, max(true_w.max(), pred.max()) + 12)
        style_trajectory_axis(ax, t)

    axes[0].set_ylabel("Blood glucose (mg/dL)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), frameon=True)

    fig.suptitle("Baseline 60min prediction", y=1.02, fontsize=9)
    fig.tight_layout()
    fname = f"patient_{pid}.png"
    fig.savefig(f"{OUT_DIR}/{fname}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved as {fname} (patient={pid}, window start={start}, min_true={true_w.min():.1f} mg/dL)")


# ═══════════════════════════════════════════════════════════════════════════
# MEAN LAG – all 8 configurations (for table column)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mean_lag_all_configs():
    configs = [
        ("LSTM MSE",          "reports/results/lstm_60min_traces_all_patients.npz"),
        ("LSTM Bounded-Lag",  "reports/results/lstm_bounded_lag_traces_all_patients.npz"),
        ("LSTM Soft-DTW",     "reports/results/lstm_dtw_traces_all_patients.npz"),
        ("LSTM Multi-step",   "reports/results/lstm_multistep_traces_all_patients.npz"),
        ("PatchTST MSE",      "reports/results/patchtst_60min_traces_all_patients.npz"),
        ("PatchTST BL",       "reports/results/patchtst_bounded_lag_traces_all_patients.npz"),
        ("PatchTST Soft-DTW", "reports/results/patchtst_dtw_traces_all_patients.npz"),
        ("PatchTST Multi",    "reports/results/patchtst_multistep_traces_all_patients.npz"),
    ]
    print("\nMean Lag (minutes) for all configurations:")
    for name, path in configs:
        traces = load_traces(path)
        lags = [abs(best_lag_rmse(t, p, max_lag=12)) * 5 for _, (t, p) in traces.items()]
        print(f"  {name}: {np.mean(lags):.1f} min")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    make_baseline_trajectory_comparison()
    make_lag_distribution()
    make_obj1_all_predictions()
    make_multistep_vs_baseline_trace()
    make_lead_time_distribution()
    compute_mean_lag_all_configs()
    print("\nAll figures saved to", OUT_DIR)
