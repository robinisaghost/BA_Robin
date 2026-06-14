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
    pos_pct    = 100 * pos_counts / total

    # Sort by positive percentage
    order = np.argsort(pos_pct)
    pids_sorted = [pids[i] for i in order]
    pos_s = pos_pct[order]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8),
                              gridspec_kw={"width_ratios": [3, 1]})

    # Left panel: per-patient bar
    x = np.arange(len(pids_sorted))
    axes[0].bar(x, pos_s, color="#C44E52", alpha=0.8, label="Positive (hypo event)")
    axes[0].bar(x, 100 - pos_s, bottom=pos_s, color="#4878CF", alpha=0.5,
                label="Negative (no hypo event)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i + 1) for i in range(len(pids_sorted))], rotation=90, fontsize=7)
    axes[0].set_ylabel("Percentage of windows (%)")
    axes[0].set_xlabel("Patient (sorted by positive rate)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="upper left")
    axes[0].set_title("Class distribution per patient")
    axes[0].axhline(pos_counts.sum() / total.sum() * 100,
                     color="black", linewidth=1.0, linestyle="--",
                     label=f"Cohort mean: {pos_counts.sum()/total.sum()*100:.1f}%")
    axes[0].legend(loc="upper left", fontsize=7.5)

    # Right panel: cohort aggregate
    total_pos = pos_counts.sum()
    total_neg = neg_counts.sum()
    axes[1].bar(["Negative", "Positive"], [total_neg, total_pos],
                color=["#4878CF", "#C44E52"], alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("Number of windows")
    axes[1].set_title("Cohort aggregate")
    for i, (label, val) in enumerate(zip(["Negative", "Positive"], [total_neg, total_pos])):
        axes[1].text(i, val + 200, f"{val:,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Class imbalance in the T1DATA hypoglycaemia detection task",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/class_imbalance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ratio = total_neg / total_pos
    print(f"  Total positive: {total_pos:,} | Total negative: {total_neg:,} | Ratio: {ratio:.1f}:1")


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
