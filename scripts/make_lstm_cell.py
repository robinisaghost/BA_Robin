"""
Figure 2.1: simplified LSTM cell diagram.

Emphasises the additive cell-state "gradient highway" and the three gates with
their sigmoid/tanh activations, but omits the weight matrices (those appear in
the equations in the text). Designed to stay readable at a small size.

Output: thesis/img/lstm_cell.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif"})

SIG_F, SIG_E = "#F6B9B9", "#C0392B"
TAN_F, TAN_E = "#BBC9F2", "#3B5BA5"
GATE = "#C0392B"
LW = 1.7

fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.set_xlim(0, 15)
ax.set_ylim(-1.7, 9.6)
ax.axis("off")

ax.add_patch(FancyBboxPatch((1.0, 2.0), 12.4, 6.6,
             boxstyle="round,pad=0.1,rounding_size=0.45",
             facecolor="#DCEDED", edgecolor="#7FB3B3", linewidth=1.6, zorder=0))


def op(xy, sym, fs=16):
    ax.add_patch(Circle(xy, 0.42, facecolor="black", edgecolor="black", zorder=5))
    ax.text(xy[0], xy[1], sym, color="white", fontsize=fs, ha="center",
            va="center", zorder=6, fontweight="bold")


def act(xy, label, ff, ee, fs=13):
    ax.add_patch(Circle(xy, 0.58, facecolor=ff, edgecolor=ee, linewidth=1.8, zorder=5))
    ax.text(xy[0], xy[1], label, fontsize=fs, ha="center", va="center", zorder=6)


def arrow(p, q, color="black", lw=LW):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                 color=color, linewidth=lw, zorder=3, shrinkA=0, shrinkB=0))


def line(p, q, color="black", lw=LW):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=lw, zorder=3)


def dot(xy):
    ax.add_patch(Circle(xy, 0.09, color="black", zorder=5))


def glabel(x, y, s):
    ax.text(x, y, s, fontsize=11, color=GATE, ha="center", fontweight="bold")


y_hi, y_rail = 8.2, 2.6
x_f, x_i, x_c = 3.3, 5.5, 7.0
x_plus, x_xin, y_xin = 6.25, 6.25, 6.95
x_branch, x_xout, x_so = 9.3, 10.6, 10.6
y_act = 5.6

# ── cell-state highway ──────────────────────────────────────────────────────
ax.text(0.15, y_hi, r"$C_{t-1}$", fontsize=13, ha="left", va="center")
arrow((0.95, y_hi), (x_f - 0.42, y_hi))
op((x_f, y_hi), r"$\times$")
arrow((x_f + 0.42, y_hi), (x_plus - 0.42, y_hi))
op((x_plus, y_hi), "+")
arrow((x_plus + 0.42, y_hi), (x_branch - 0.05, y_hi))
dot((x_branch, y_hi))
arrow((x_branch, y_hi), (13.55, y_hi))
ax.text(13.7, y_hi, r"$C_t$", fontsize=13, ha="left", va="center")
ax.text(x_plus + 1.4, y_hi + 0.5, "Cell state", fontsize=11, color=GATE,
        ha="center", fontweight="bold")

# ── forget gate ─────────────────────────────────────────────────────────────
act((x_f, y_act), r"$\sigma$", SIG_F, SIG_E, fs=15)
arrow((x_f, y_act + 0.58), (x_f, y_hi - 0.42))
glabel(x_f, 4.35, "Forget")

# ── input gate ──────────────────────────────────────────────────────────────
act((x_i, y_act), r"$\sigma$", SIG_F, SIG_E, fs=15)
act((x_c, y_act), "tanh", TAN_F, TAN_E, fs=11)
op((x_xin, y_xin), r"$\times$")
arrow((x_i, y_act + 0.58), (x_xin - 0.30, y_xin - 0.30))
arrow((x_c, y_act + 0.58), (x_xin + 0.30, y_xin - 0.30))
arrow((x_xin, y_xin + 0.42), (x_plus, y_hi - 0.42))
glabel((x_i + x_c) / 2, 4.35, "Input")

# ── output gate: cell-state tanh and output sigma → ⊗ → h_t ────────────────
y_xout = 6.0
y_so = 4.4
act((x_branch, 6.75), "tanh", TAN_F, TAN_E, fs=11)
line((x_branch, y_hi), (x_branch, 6.75 + 0.58))
op((x_xout, y_xout), r"$\times$")
arrow((x_branch + 0.4, 6.75 - 0.25), (x_xout - 0.35, y_xout + 0.28))
act((x_so, y_so), r"$\sigma$", SIG_F, SIG_E, fs=15)
arrow((x_so, y_so + 0.58), (x_xout, y_xout - 0.42))
glabel(x_so + 1.1, y_so + 0.1, "Output")

# ── input rail: x_t and h_{t-1} feed every gate ────────────────────────────
ax.text(0.15, y_rail, r"$h_{t-1}$", fontsize=13, ha="left", va="center")
line((0.95, y_rail), (x_so, y_rail))
for gx, gy in ((x_f, y_act), (x_i, y_act), (x_c, y_act), (x_so, y_so)):
    dot((gx, y_rail))
    arrow((gx, y_rail), (gx, gy - 0.58))
arrow((x_f, 0.7), (x_f, y_rail))
ax.text(x_f, 0.35, r"$x_t$", fontsize=13, ha="center", va="top")

# ── h_t output and linear head ──────────────────────────────────────────────
x_drop = 12.1
arrow((x_xout + 0.42, y_xout), (13.55, y_xout))
ax.text(13.7, y_xout, r"$h_t$", fontsize=13, ha="left", va="center")
dot((x_drop, y_xout))
arrow((x_drop, y_xout), (x_drop, 0.55))
ax.add_patch(FancyBboxPatch((x_drop - 1.05, -0.55), 2.1, 1.05,
             boxstyle="round,pad=0.05,rounding_size=0.15",
             facecolor="#F6D7D7", edgecolor="#C0392B", linewidth=1.5, zorder=4))
ax.text(x_drop, -0.02, "Linear\nhead", fontsize=11, ha="center", va="center", zorder=5)
arrow((x_drop, -0.55), (x_drop, -1.3))
ax.text(x_drop + 0.25, -1.5, r"$\hat{y}\;\,(t+60\,\mathrm{min})$", fontsize=12,
        ha="left", va="center")

fig.tight_layout()
fig.savefig("thesis/img/lstm_cell.png", dpi=300, bbox_inches="tight")
print("wrote thesis/img/lstm_cell.png")
