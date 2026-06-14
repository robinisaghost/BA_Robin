"""
Figure 2.1: simplified LSTM cell diagram.

Emphasises the additive cell-state "gradient highway" and the three gates with
their sigmoid/tanh activations, but omits the weight matrices (those appear in
the equations in the text). Each gate is grouped by a dashed box and marked by
its initial (f/i/o); a legend below explains the symbols. Designed to stay
readable at a small print size.

Output: thesis/img/lstm_cell.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "serif"})

SIG_F, SIG_E = "#F6B9B9", "#C0392B"
TAN_F, TAN_E = "#BBC9F2", "#3B5BA5"
GATE = "#C0392B"
LW = 1.8

fig, ax = plt.subplots(figsize=(7.8, 6.0))
ax.set_xlim(-0.2, 14.2)
ax.set_ylim(-3.1, 10.4)
ax.axis("off")

ax.add_patch(FancyBboxPatch((1.9, 1.7), 10.8, 7.2,
             boxstyle="round,pad=0.1,rounding_size=0.4",
             facecolor="#DCEDED", edgecolor="#7FB3B3", linewidth=1.6, zorder=0))
ax.text(7.3, 9.7, "LSTM cell", fontsize=16, fontweight="bold", color="#4d4d4d",
        ha="center", va="center")


def op(xy, sym, fs=17):
    ax.add_patch(Circle(xy, 0.42, facecolor="black", edgecolor="black", zorder=6))
    ax.text(xy[0], xy[1], sym, color="white", fontsize=fs, ha="center",
            va="center", zorder=7, fontweight="bold")


def act(xy, label, ff, ee, fs=15, r=0.58):
    ax.add_patch(Circle(xy, r, facecolor=ff, edgecolor=ee, linewidth=1.9, zorder=6))
    ax.text(xy[0], xy[1], label, fontsize=fs, ha="center", va="center", zorder=7)


def arrow(p, q, color="black", lw=LW):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                 color=color, linewidth=lw, zorder=4, shrinkA=0, shrinkB=0))


def line(p, q, color="black", lw=LW):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=lw, zorder=4)


def dot(xy):
    ax.add_patch(Circle(xy, 0.10, color="black", zorder=6))


def gate_box(x0, y0, x1, y1, letter, lx, ly):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                 edgecolor=GATE, linewidth=1.5, linestyle=(0, (5, 3)), zorder=2))
    ax.text(lx, ly, letter, fontsize=16, color=GATE, ha="center",
            va="center", fontweight="bold", style="italic", zorder=8)


y_hi, y_rail, y_act = 8.1, 2.5, 5.5
x_f, x_i, x_c = 3.4, 5.5, 7.0
x_plus, x_xin, y_xin = 6.25, 6.25, 6.85
x_branch, x_xout, x_so = 9.4, 10.6, 10.6
y_xout, y_so = 5.9, 4.3

# ── cell-state highway ──────────────────────────────────────────────────────
ax.text(0.0, y_hi, r"$C_{t-1}$", fontsize=15, ha="left", va="center")
arrow((1.1, y_hi), (x_f - 0.42, y_hi))
op((x_f, y_hi), r"$\times$")
arrow((x_f + 0.42, y_hi), (x_plus - 0.42, y_hi))
op((x_plus, y_hi), "+")
arrow((x_plus + 0.42, y_hi), (x_branch - 0.05, y_hi))
dot((x_branch, y_hi))
arrow((x_branch, y_hi), (12.75, y_hi))
ax.text(12.9, y_hi, r"$C_t$", fontsize=15, ha="left", va="center")

# ── gates (dashed boxes + initials in a clear top-left corner) ──────────────
gate_box(2.55, 4.85, 4.25, 8.7, "f", 2.9, 7.05)
act((x_f, y_act), r"$\sigma$", SIG_F, SIG_E)
arrow((x_f, y_act + 0.58), (x_f, y_hi - 0.42))

gate_box(4.55, 4.85, 7.75, 7.55, "i", 4.9, 7.2)
act((x_i, y_act), r"$\sigma$", SIG_F, SIG_E)
act((x_c, y_act), "tanh", TAN_F, TAN_E, fs=12)
op((x_xin, y_xin), r"$\times$")
arrow((x_i, y_act + 0.58), (x_xin - 0.30, y_xin - 0.30))
arrow((x_c, y_act + 0.58), (x_xin + 0.30, y_xin - 0.30))
arrow((x_xin, y_xin + 0.42), (x_plus, y_hi - 0.42))

gate_box(8.6, 3.55, 11.5, 7.45, "o", 8.9, 7.0)
act((x_branch, 6.6), "tanh", TAN_F, TAN_E, fs=12)
line((x_branch, y_hi), (x_branch, 6.6 + 0.58))
op((x_xout, y_xout), r"$\times$")
arrow((x_branch + 0.40, 6.6 - 0.25), (x_xout - 0.35, y_xout + 0.28))
act((x_so, y_so), r"$\sigma$", SIG_F, SIG_E)
arrow((x_so, y_so + 0.58), (x_xout, y_xout - 0.42))

# ── input rail ──────────────────────────────────────────────────────────────
ax.text(0.0, y_rail, r"$h_{t-1}$", fontsize=15, ha="left", va="center")
line((1.1, y_rail), (x_so, y_rail))
for gx, gy in ((x_f, y_act), (x_i, y_act), (x_c, y_act), (x_so, y_so)):
    dot((gx, y_rail))
    arrow((gx, y_rail), (gx, gy - 0.58))
arrow((x_f, 0.55), (x_f, y_rail))
ax.text(x_f, 0.2, r"$x_t$", fontsize=15, ha="center", va="top")

# ── h_t output and linear head ──────────────────────────────────────────────
x_drop = 11.95
arrow((x_xout + 0.42, y_xout), (12.75, y_xout))
ax.text(12.9, y_xout, r"$h_t$", fontsize=15, ha="left", va="center")
dot((x_drop, y_xout))
arrow((x_drop, y_xout), (x_drop, 0.5))
ax.add_patch(FancyBboxPatch((x_drop - 1.05, -0.6), 2.1, 1.05,
             boxstyle="round,pad=0.05,rounding_size=0.15",
             facecolor="#F6D7D7", edgecolor="#C0392B", linewidth=1.6, zorder=5))
ax.text(x_drop, -0.07, "Linear\nhead", fontsize=13, ha="center", va="center", zorder=6)
arrow((x_drop, -0.6), (x_drop, -1.35))
ax.text(x_drop, -1.7, r"$\hat{y}\;\,(t+60\,\mathrm{min})$", fontsize=14,
        ha="center", va="top")

# ── legend ──────────────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.05, -2.95), 10.3, 1.9,
             boxstyle="round,pad=0.05,rounding_size=0.12",
             facecolor="white", edgecolor="#999999", linewidth=1.0, zorder=5))


def leg_act(xy, label, ff, ee, text, fs=11):
    ax.add_patch(Circle(xy, 0.30, facecolor=ff, edgecolor=ee, linewidth=1.5, zorder=6))
    ax.text(xy[0], xy[1], label, fontsize=fs - 2, ha="center", va="center", zorder=7)
    ax.text(xy[0] + 0.45, xy[1], text, fontsize=11, ha="left", va="center", zorder=6)


leg_act((0.55, -1.55), r"$\sigma$", SIG_F, SIG_E, "sigmoid")
leg_act((0.55, -2.45), "tanh", TAN_F, TAN_E, "hyperbolic tangent", fs=8)
ax.text(4.05, -1.55, r"$C_t$ : cell state", fontsize=11, ha="left", va="center", zorder=7)
ax.text(4.05, -2.45, r"$h_t$ : hidden state", fontsize=11, ha="left", va="center", zorder=7)
ax.text(7.5, -1.4, r"$f$ : forget gate", fontsize=11, ha="left", va="center", color=GATE, zorder=7)
ax.text(7.5, -2.0, r"$i$ : input gate", fontsize=11, ha="left", va="center", color=GATE, zorder=7)
ax.text(7.5, -2.6, r"$o$ : output gate", fontsize=11, ha="left", va="center", color=GATE, zorder=7)

fig.tight_layout()
fig.savefig("thesis/img/lstm_cell.png", dpi=300, bbox_inches="tight")
print("wrote thesis/img/lstm_cell.png")
