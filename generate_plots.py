"""
Figure 2.3: geometric interpretation of the three forecasting losses.

Produces a single combined figure (1x3 panels: MSE, Bounded-Lag, Soft-DTW) with
one shared legend, so the panels are as large as possible and no per-panel
legend overlaps the curves. Schematic style: light grey panel background (for
visual consistency with the data figures) but no gridlines, since the axes carry
no numeric values.

Output: thesis/img/loss_geometry.png  (included at \\textwidth in background.tex)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema

OUT = "thesis/img/loss_geometry.png"

COL_TRUE = "black"
COL_PRED = "#0072B2"   # Okabe-Ito blue, matching the trajectory figures
COL_LINK = "#E69F00"   # Okabe-Ito orange for the loss correspondences

plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def style(ax, title, ylabel=False):
    ax.set_title(title)
    ax.set_facecolor("#EAEAF2")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time $t$")
    if ylabel:
        ax.set_ylabel("Glucose")


fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))

# ── (a) MSE: same-time-step correspondences (vertical links) ────────────────
t = np.linspace(0, 4 * np.pi, 90)
shape_true = np.sin(t)
# prediction is the same shape but delayed (shifted to the right / too late),
# the characteristic behaviour of the MSE-trained models
shape_pred = np.sin(t - 1.8)

ax = axes[0]
peaks = argrelextrema(shape_true, np.greater, order=8)[0]
troughs = argrelextrema(shape_true, np.less, order=8)[0]
for idx in np.sort(np.concatenate([peaks, troughs])):
    ax.plot([t[idx], t[idx]], [shape_true[idx], shape_pred[idx]],
            color=COL_LINK, linestyle="--", linewidth=1.6, alpha=0.9)
ax.plot(t, shape_true, color=COL_TRUE, linewidth=2.0)
ax.plot(t, shape_pred, color=COL_PRED, linewidth=2.0)
style(ax, "(a) MSE", ylabel=True)

# ── (b) Bounded-Lag: nearest-feature correspondences (horizontal links) ─────
# Show one tolerated shift each in the upper (peak), middle (zero crossing) and
# lower (trough) part of the panel, so the dashed links are clearly separated
# and easy to read.
ax = axes[1]
ax.plot(t, shape_true, color=COL_TRUE, linewidth=2.0)
ax.plot(t, shape_pred, color=COL_PRED, linewidth=2.0)

pk_t = argrelextrema(shape_true, np.greater, order=8)[0]
pk_p = argrelextrema(shape_pred, np.greater, order=8)[0]
tr_t = argrelextrema(shape_true, np.less, order=8)[0]
tr_p = argrelextrema(shape_pred, np.less, order=8)[0]
# downward zero crossings (sign + -> -) for a clear mid-level feature
dc_t = np.where((shape_true[:-1] > 0) & (shape_true[1:] <= 0))[0]
dc_p = np.where((shape_pred[:-1] > 0) & (shape_pred[1:] <= 0))[0]


def delayed(i, cand):
    """Matching feature on the prediction, i.e. the nearest one to the right
    (the prediction lags, so its counterpart of a true feature comes later)."""
    right = cand[cand >= i]
    return right[0] if len(right) else cand[-1]


# connect each ground-truth feature to its delayed counterpart on the
# prediction, at the upper (peak), middle (zero crossing) and lower (trough)
# part of the panel so the tolerated lag is clearly separated and readable
links_b = [
    (pk_t[0], delayed(pk_t[0], pk_p)),   # upper
    (dc_t[0], delayed(dc_t[0], dc_p)),   # middle
    (tr_t[0], delayed(tr_t[0], tr_p)),   # lower
]
for a, b in links_b:
    yv = (shape_true[a] + shape_pred[b]) / 2
    ax.plot([t[a], t[b]], [yv, yv], color=COL_LINK, linestyle="--",
            linewidth=1.8, alpha=0.95, zorder=5)
style(ax, "(b) Bounded-Lag ($D{=}3$)")

# ── (c) Soft-DTW: optimal warping-path correspondences ──────────────────────
ax = axes[2]
n = 70
x = np.arange(n)
sig1 = 2.0 * np.sin(np.linspace(0, 4 * np.pi, n)) + 0.6
u = np.linspace(0, 1, n)
sig2 = 2.0 * np.sin((u + 0.28 * np.sin(np.pi * u)) * 4 * np.pi) - 0.6

dtw = np.full((n + 1, n + 1), np.inf)
dtw[0, 0] = 0.0
for i in range(1, n + 1):
    for j in range(1, n + 1):
        cost = (sig1[i - 1] - sig2[j - 1]) ** 2
        dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
path, i, j = [], n, n
while i > 0 and j > 0:
    path.append((i - 1, j - 1))
    best = min(dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1])
    if best == dtw[i - 1, j - 1]:
        i, j = i - 1, j - 1
    elif best == dtw[i - 1, j]:
        i -= 1
    else:
        j -= 1
for i1, i2 in path:
    ax.plot([x[i1], x[i2]], [sig1[i1], sig2[i2]], color=COL_LINK,
            linewidth=0.7, alpha=0.45)
ax.plot(x, sig1, color=COL_TRUE, linewidth=2.0)
ax.plot(x, sig2, color=COL_PRED, linewidth=2.0)
style(ax, "(c) Soft-DTW ($\\gamma{=}1.0$)")

# ── shared legend ───────────────────────────────────────────────────────────
handles = [
    Line2D([0], [0], color=COL_TRUE, linewidth=2.0, label="True ($y$)"),
    Line2D([0], [0], color=COL_PRED, linewidth=2.0, label="Predicted ($\\hat{y}$)"),
    Line2D([0], [0], color=COL_LINK, linewidth=1.6, linestyle="--",
           label="Loss correspondence"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT}")
