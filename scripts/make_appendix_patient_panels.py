"""
Appendix figures: per-patient forecasting trajectories across all four training
variants, for both architectures.

Layout: panels are stacked vertically at full text width so the 5-minute
sampling and the corners of each trajectory are clearly visible. Each panel
overlays the ground truth and all four forecasting variants (MSE, Bounded-Lag,
Soft-DTW, Multi-step) for one patient and one architecture over a 10-hour test
window. The eight panels (4 patients x 2 architectures) are split across two
one-page figures, with a shared legend on each page.

Colours follow the Okabe-Ito colourblind-safe palette and are fixed per variant
across every panel; each variant also has a distinct line style so the figures
remain readable in greyscale. Small markers mark every 5-minute sample.

Data: pre-computed trace files in reports/results/ (no model retraining).
Output: thesis/img/appendix_patient_panels_1.png and _2.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

TRACES = "reports/results"
OUT_TMPL = "thesis/img/appendix_patient_panels_{}.png"

HYPO_THRESHOLD = 70.0
SAMPLING_MIN = 5
WIN = 120  # 120 steps x 5 min = 10-hour display window

# Two pages, two patients each (LSTM + PatchTST panels per patient).
GROUPS = [["85102", "85116"], ["85215", "85217"]]

TRUE_STYLE = dict(color="#000000", linestyle="-", linewidth=1.8, zorder=5)
VARIANTS = [
    # label,            file-stem,      colour,     linestyle
    ("MSE (baseline)",  "60min",        "#0072B2",  "-"),
    ("Bounded-Lag",     "bounded_lag",  "#E69F00",  (0, (5, 1.5))),
    ("Soft-DTW",        "dtw",          "#009E73",  (0, (3, 1, 1, 1))),
    ("Multi-step",      "multistep",    "#CC79A7",  (0, (1, 1))),
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


def legend_handles():
    h = [Line2D([0], [0], color="#000000", linestyle="-", linewidth=1.8,
                marker="o", markersize=3, label="Ground truth")]
    h += [Line2D([0], [0], color=c, linestyle=ls, linewidth=1.5,
                 marker="o", markersize=3, label=lab)
          for lab, _, c, ls in VARIANTS]
    h.append(Line2D([0], [0], color=THRESHOLD_COLOR, linestyle=(0, (4, 3)),
                    linewidth=1.0, label="70 mg/dL threshold"))
    return h


def make_page(patients, out_path):
    # One panel per (patient, architecture), patient's two panels adjacent.
    panels = [(pid, arch_lab, arch_stem)
              for pid in patients for arch_lab, arch_stem in ARCHS]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(6.9, 7.9))

    for ax, (pid, arch_lab, arch_stem) in zip(axes, panels):
        gt_ref = test_portion(load_trace("lstm", "60min", pid, "true"))
        s, e = find_hypo_window(gt_ref)
        t = np.arange(e - s) * SAMPLING_MIN  # minutes within window

        gt = test_portion(load_trace(arch_stem, "60min", pid, "true"))
        ax.plot(t, gt[s:e], marker="o", markersize=2.4, markevery=1,
                label="Ground truth", **TRUE_STYLE)
        for _, var_stem, color, ls in VARIANTS:
            pred = test_portion(load_trace(arch_stem, var_stem, pid, "pred"))
            m = min(len(pred), e) - s
            ax.plot(t[:m], pred[s:s + m], color=color, linestyle=ls,
                    linewidth=1.3, marker="o", markersize=2.0, markevery=1,
                    alpha=0.95)

        ax.axhline(HYPO_THRESHOLD, color=THRESHOLD_COLOR, linestyle=(0, (4, 3)),
                   linewidth=1.0, zorder=1)
        ax.set_title(f"Patient {pid} — {arch_lab}", fontsize=10,
                     fontweight="bold", loc="left")
        ax.set_ylabel("Glucose [mg/dL]")
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(which="major", axis="x", color="0.85", linewidth=0.6)
        ax.grid(which="minor", axis="x", color="0.93", linewidth=0.4)
        ax.set_xlim(t[0], t[-1])
        ax.margins(y=0.08)

    axes[-1].set_xlabel("Time [min]  (5-minute samples)")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })
    for i, patients in enumerate(GROUPS, start=1):
        make_page(patients, OUT_TMPL.format(i))


if __name__ == "__main__":
    main()
