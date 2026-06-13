"""
Appendix figure: per-patient forecasting trajectories across all four training
variants, for both architectures.

Layout: 4 patients (rows) x 2 architectures (columns). Each panel overlays the
ground truth and all four forecasting variants (MSE, Bounded-Lag, Soft-DTW,
Multi-step) for one patient and one architecture, over a 10-hour test window.
A single shared legend is placed at the bottom of the page.

Colours follow the Okabe-Ito colourblind-safe palette and are fixed per variant
across every panel; each variant also has a distinct line style so the figure
remains readable in greyscale.

Data: pre-computed trace files in reports/results/ (no model retraining).
Output: thesis/img/appendix_patient_panels.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TRACES = "reports/results"
OUT = "thesis/img/appendix_patient_panels.png"

HYPO_THRESHOLD = 70.0
SAMPLING_MIN = 5
WIN = 120  # 120 steps x 5 min = 10-hour display window

# Patients chosen for distinct qualitative behaviour in the test window.
# (id, short descriptor)
PATIENTS = [
    ("85102", "Sharp single event"),
    ("85116", "Volatile, multiple events"),
    ("85215", "Gradual single descent"),
    ("85217", "No hypoglycaemia event"),
]

# Okabe-Ito colourblind-safe palette; ground truth in black.
# Each variant: fixed (colour, linestyle) used in every panel.
TRUE_STYLE = dict(color="#000000", linestyle="-", linewidth=2.2, zorder=5)
VARIANTS = [
    # label,            file-stem,        colour,     linestyle
    ("MSE (baseline)",  "60min",          "#0072B2",  "-"),
    ("Bounded-Lag",     "bounded_lag",    "#E69F00",  (0, (5, 1.5))),
    ("Soft-DTW",        "dtw",            "#009E73",  (0, (3, 1, 1, 1))),
    ("Multi-step",      "multistep",      "#CC79A7",  (0, (1, 1))),
]
ARCHS = [("LSTM", "lstm"), ("PatchTST", "patchtst")]
THRESHOLD_COLOR = "#777777"


def find_hypo_window(y_true, win_size=WIN):
    y = np.asarray(y_true)
    n = len(y)
    if n <= win_size:
        return 0, n
    crossings = [t for t in range(1, n)
                 if y[t - 1] >= HYPO_THRESHOLD and y[t] < HYPO_THRESHOLD]
    if crossings:
        best_score, best_t = -np.inf, crossings[len(crossings) // 2]
        for t in crossings:
            s = max(0, t - win_size // 2)
            e = min(n, s + win_size)
            score = float(np.std(y[s:e]))
            if score > best_score:
                best_score, best_t = score, t
        center = best_t
    else:
        best_mean, center = np.inf, win_size // 2
        for s in range(0, n - win_size, max(1, win_size // 4)):
            m = float(np.mean(y[s:s + win_size]))
            if m < best_mean:
                best_mean, center = m, s + win_size // 2
    start = max(0, center - win_size // 2)
    end = min(n, start + win_size)
    start = max(0, end - win_size)
    return start, end


def test_portion(arr):
    arr = np.asarray(arr)
    return arr[int(0.8 * len(arr)):]


def load_trace(arch_stem, var_stem, pid, key):
    path = os.path.join(TRACES, f"{arch_stem}_{var_stem}_traces_all_patients.npz")
    d = np.load(path)
    return np.asarray(d[f"{pid}_{key}"])


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })

    n_rows, n_cols = len(PATIENTS), len(ARCHS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 8.9))

    for r, (pid, descriptor) in enumerate(PATIENTS):
        # Window derived once from the patient's ground truth (LSTM-MSE file).
        gt_ref = test_portion(load_trace("lstm", "60min", pid, "true"))
        s, e = find_hypo_window(gt_ref)
        t_hours = (np.arange(e - s) * SAMPLING_MIN) / 60.0

        for c, (arch_label, arch_stem) in enumerate(ARCHS):
            ax = axes[r, c]
            gt = test_portion(load_trace(arch_stem, "60min", pid, "true"))
            ax.plot(t_hours, gt[s:e], label="Ground truth", **TRUE_STYLE)

            for var_label, var_stem, color, ls in VARIANTS:
                pred = test_portion(load_trace(arch_stem, var_stem, pid, "pred"))
                m = min(len(pred), e) - s
                ax.plot(t_hours[:m], pred[s:s + m], color=color,
                        linestyle=ls, linewidth=1.4, alpha=0.95)

            ax.axhline(HYPO_THRESHOLD, color=THRESHOLD_COLOR, linestyle=(0, (4, 3)),
                       linewidth=1.0, zorder=1)

            if r == 0:
                ax.set_title(arch_label, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Glucose [mg/dL]")
                ax.text(-0.30, 0.5, f"Patient {pid}\n{descriptor}",
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=8.5, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time [h]")
            ax.margins(x=0.01)

    # Shared legend at the bottom.
    handles = [Line2D([0], [0], **{k: v for k, v in TRUE_STYLE.items()
                                   if k in ("color", "linestyle", "linewidth")},
                      label="Ground truth")]
    handles += [Line2D([0], [0], color=c, linestyle=ls, linewidth=1.6, label=lab)
                for lab, _, c, ls in VARIANTS]
    handles.append(Line2D([0], [0], color=THRESHOLD_COLOR, linestyle=(0, (4, 3)),
                          linewidth=1.0, label="70 mg/dL threshold"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0.03, 0.06, 1.0, 1.0))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
