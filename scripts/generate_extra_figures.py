"""
Generate two additional figures for the thesis:
  class_imbalance.png      -- Chapter 2: positive vs negative window counts per patient
  chronological_split.png  -- Chapter 3: schematic of 60/20/20 temporal split
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, "src")

OUT_DIR = "thesis/img"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

HYPO = 70.0
LOOKBACK = 24
HORIZON  = 12


# ═══════════════════════════════════════════════════════════════════════════
# class_imbalance.png
# ═══════════════════════════════════════════════════════════════════════════

def make_class_imbalance():
    print("Generating class_imbalance.png ...")
    d = np.load("reports/results/lstm_60min_traces_all_patients.npz")
    pids = sorted(set(k.split("_")[0] for k in d.keys()))

    pos_counts, neg_counts = [], []
    for pid in pids:
        true = d[pid + "_true"]
        n_total = len(true) - HORIZON
        pos = 0
        for i in range(n_total):
            window_future = true[i: i + HORIZON]
            if window_future.min() < HYPO:
                pos += 1
        neg = n_total - pos
        pos_counts.append(pos)
        neg_counts.append(neg)

    pos_counts = np.array(pos_counts)
    neg_counts = np.array(neg_counts)
    total      = pos_counts + neg_counts
    pos_pct    = np.sort(100 * pos_counts / total)
    mean_pct   = pos_counts.sum() / total.sum() * 100

    # Single full-width panel: the per-patient positive rate as red bars, so
    # individual patients stay distinguishable and the (small) positive rates
    # are clearly visible. The aggregate counts go in the caption instead.
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    x = np.arange(len(pos_pct))
    ax.bar(x, pos_pct, width=0.8, color="#C44E52",
           label="Positive (hypoglycaemia) windows")
    ax.axhline(mean_pct, color="black", linewidth=1.2, linestyle="--",
               label=f"Cohort mean: {mean_pct:.1f}%")
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(pos_pct) - 0.3)
    ax.set_ylim(0, pos_pct.max() * 1.18)
    ax.set_ylabel("Positive windows (%)")
    ax.set_xlabel("Patients (each bar one patient, sorted by positive rate)")
    ax.set_title("Per-patient rate of positive (hypoglycaemia) windows")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/class_imbalance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ratio = neg_counts.sum() / pos_counts.sum()
    print(f"  Total positive: {pos_counts.sum():,} | Total negative: {neg_counts.sum():,} | Ratio: {ratio:.1f}:1")


# ═══════════════════════════════════════════════════════════════════════════
# chronological_split.png
# ═══════════════════════════════════════════════════════════════════════════

def make_chronological_split():
    print("Generating chronological_split.png ...")

    fig, ax = plt.subplots(figsize=(8.0, 2.4))

    segments = [
        (0.00, 0.60, "#4878CF", "Train (60%)"),
        (0.60, 0.20, "#55A868", "Validation (20%)"),
        (0.80, 0.20, "#C44E52", "Test (20%)"),
    ]

    bar_y  = 0.5
    bar_h  = 0.3
    n_rows = 4  # show 4 example patients

    for row in range(n_rows):
        y = bar_y + row * 0.7
        left = 0.0
        for start, width, color, label in segments:
            ax.barh(y, width, left=left, height=bar_h, color=color,
                    alpha=0.85, edgecolor="white", linewidth=0.5)
            # label only on the first row
            if row == 0:
                cx = left + width / 2
                ax.text(cx, y, label, ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += width
        ax.text(-0.02, y, f"Patient {row+1}", ha="right", va="center", fontsize=8)

    ax.set_xlim(-0.12, 1.05)
    ax.set_ylim(0.2, bar_y + n_rows * 0.7 + 0.1)
    ax.set_xlabel("Relative time (fraction of patient record)")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # Legend patches
    legend_patches = [mpatches.Patch(color=c, label=lbl, alpha=0.85)
                      for _, _, c, lbl in segments]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9, fontsize=8)

    ax.set_title("Chronological 60/20/20 split applied independently per patient",
                  pad=8, fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/chronological_split.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved.")


if __name__ == "__main__":
    make_class_imbalance()
    make_chronological_split()
    print(f"\nAll figures saved to {OUT_DIR}/")
