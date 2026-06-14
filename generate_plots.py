import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# 1. Daten generieren
t = np.linspace(0, 4 * np.pi, 90)

shape_true = np.sin(t)
shape_pred = np.sin(t * 0.85 - 0.8)

y_true = shape_true + 1.5
y_pred = shape_pred - 1.5


# 2. Plot Setup auf Englisch
def setup_plot(title):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t, y_true, color="black", linewidth=2.5, label="True ($y$)")
    ax.plot(t, y_pred, color="tab:blue", linewidth=2.5, label="Predicted ($\hat{y}$)")

    ax.set_title(title, pad=15, fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Glucose", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)

    return fig, ax


# === MSE Loss ===
# Kurven überlappend (kein vertikaler Offset) — zeigt grossen punktweisen Fehler
# trotz korrekter Form
y_mse_true = shape_true
y_mse_pred = shape_pred

fig, ax = plt.subplots(figsize=(3.0, 2.5))
ax.plot(t, y_mse_true, color="black",    linewidth=2.5, label="True ($y$)")
ax.plot(t, y_mse_pred, color="tab:blue", linewidth=2.5, label="Predicted ($\\hat{y}$)")
ax.set_title("MSE Loss", pad=15, fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time $t$", fontsize=12)
ax.set_ylabel("Glucose", fontsize=12)
ax.legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 1.15))

# Vertikale Linien an allen Peaks und Troughs der True-Kurve
peaks_true_mse   = argrelextrema(y_mse_true, np.greater, order=8)[0]
troughs_true_mse = argrelextrema(y_mse_true, np.less,    order=8)[0]
sample_indices   = np.sort(np.concatenate([peaks_true_mse, troughs_true_mse]))

for idx in sample_indices:
    ax.plot(
        [t[idx], t[idx]],
        [y_mse_true[idx], y_mse_pred[idx]],
        color="tab:orange", linestyle="--", linewidth=1.8, alpha=0.85,
    )

fig.tight_layout()
fig.savefig("thesis/img/loss_mse.png", dpi=300, bbox_inches="tight")
plt.close()
print("Erfolgreich generiert: loss_mse.png")


# === Bounded-Lag Loss ===
y_bl_true = shape_true
y_bl_pred = shape_pred

fig, ax = plt.subplots(figsize=(3.0, 2.5))
ax.plot(t, y_bl_true, color="black",    linewidth=2.5, label="True ($y$)")
ax.plot(t, y_bl_pred, color="tab:blue", linewidth=2.5, label="Predicted ($\\hat{y}$)")
ax.set_title("Bounded-Lag Loss", pad=15, fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time $t$", fontsize=12)
ax.set_ylabel("Glucose", fontsize=12)
ax.legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 1.15))

# Horizontale Linien: Peaks, Troughs und Nulldurchgänge der True-Kurve,
# gepaart mit dem zeitlich nächsten äquivalenten Punkt auf der Pred-Kurve.
peaks_true_bl   = argrelextrema(y_bl_true, np.greater, order=8)[0]
peaks_pred_bl   = argrelextrema(y_bl_pred, np.greater, order=8)[0]
troughs_true_bl = argrelextrema(y_bl_true, np.less,    order=8)[0]
troughs_pred_bl = argrelextrema(y_bl_pred, np.less,    order=8)[0]

# Nulldurchgänge (Vorzeichen-Wechsel) beider Kurven
zc_true_bl = np.where(np.diff(np.sign(y_bl_true)))[0]
zc_pred_bl = np.where(np.diff(np.sign(y_bl_pred)))[0]

line_kw = dict(color="tab:orange", linestyle="--", linewidth=1.8, alpha=0.85)

for pt, pp in zip(peaks_true_bl, peaks_pred_bl):
    y_val = (y_bl_true[pt] + y_bl_pred[pp]) / 2
    ax.plot([t[pt], t[pp]], [y_val, y_val], **line_kw)

for tt, tp in zip(troughs_true_bl, troughs_pred_bl):
    y_val = (y_bl_true[tt] + y_bl_pred[tp]) / 2
    ax.plot([t[tt], t[tp]], [y_val, y_val], **line_kw)

for zt, zp in zip(zc_true_bl, zc_pred_bl):
    y_val = (y_bl_true[zt] + y_bl_pred[zp]) / 2
    ax.plot([t[zt], t[zp]], [y_val, y_val], **line_kw)

fig.tight_layout()
fig.savefig("thesis/img/loss_bounded_lag.png", dpi=300, bbox_inches="tight")
plt.close()
print("Erfolgreich generiert: loss_bounded_lag.png")


fig, ax = setup_plot("Soft-DTW Loss")

# 3. DTW Pfad berechnen
n = len(t)
dtw_matrix = np.full((n + 1, n + 1), np.inf)
dtw_matrix[0, 0] = 0

for i in range(1, n + 1):
    for j in range(1, n + 1):
        cost = (shape_true[i - 1] - shape_pred[j - 1]) ** 2
        dtw_matrix[i, j] = cost + min(
            dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
        )

path = []
i, j = n, n
while i > 0 and j > 0:
    path.append((i - 1, j - 1))
    min_prev = min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i, j - 1])
    if min_prev == dtw_matrix[i - 1, j - 1]:
        i, j = i - 1, j - 1
    elif min_prev == dtw_matrix[i - 1, j]:
        i -= 1
    else:
        j -= 1

# 4. Linien zeichnen
for step in path:
    idx_true, idx_pred = step
    ax.plot(
        [t[idx_true], t[idx_pred]],
        [y_true[idx_true], y_pred[idx_pred]],
        color="tab:orange",
        linestyle="-",
        linewidth=0.8,
        alpha=0.6,
    )

# 5. Speichern
fig.tight_layout()
file_name = "loss_dtw_english.png"
fig.savefig(file_name, dpi=300, bbox_inches="tight")

print(f"Erfolgreich generiert: {file_name}")


# === DTW Alignment Illustration ===
n = 70
x = np.arange(n)

# Signal 1: 2 gleichmässige Zyklen (True)
sig1 = 2.0 * np.sin(np.linspace(0, 4 * np.pi, n)) + 0.6

# Signal 2: glatter Zeitwarp via u + a·sin(π·u), überall differenzierbar
# (maximale Ableitung 1 - 0.28π ≈ 0.12 > 0, also streng monoton)
u = np.linspace(0, 1, n)
t2_angle = (u + 0.28 * np.sin(np.pi * u)) * 4 * np.pi
sig2 = 2.0 * np.sin(t2_angle) - 0.6

fig, ax = plt.subplots(figsize=(3.0, 2.5))
ax.plot(x, sig1, color="black",    linewidth=2.5, label="True ($y$)")
ax.plot(x, sig2, color="tab:blue", linewidth=2.5, label="Predicted ($\\hat{y}$)")

ax.set_title("Time Series Aligned by DTW", pad=15, fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Time $t$", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)
ax.legend(loc="upper right", fontsize=11)

# DTW-Pfad berechnen
m = n
dtw_mat = np.full((m + 1, m + 1), np.inf)
dtw_mat[0, 0] = 0.0
for ii in range(1, m + 1):
    for jj in range(1, m + 1):
        cost = (sig1[ii - 1] - sig2[jj - 1]) ** 2
        dtw_mat[ii, jj] = cost + min(
            dtw_mat[ii - 1, jj - 1],
            dtw_mat[ii - 1, jj],
            dtw_mat[ii, jj - 1],
        )

path_align = []
ii, jj = m, m
while ii > 0 and jj > 0:
    path_align.append((ii - 1, jj - 1))
    best = min(dtw_mat[ii - 1, jj - 1], dtw_mat[ii - 1, jj], dtw_mat[ii, jj - 1])
    if best == dtw_mat[ii - 1, jj - 1]:
        ii, jj = ii - 1, jj - 1
    elif best == dtw_mat[ii - 1, jj]:
        ii -= 1
    else:
        jj -= 1

for idx1, idx2 in path_align:
    ax.plot(
        [x[idx1], x[idx2]],
        [sig1[idx1], sig2[idx2]],
        color="tab:orange",
        linewidth=0.7,
        alpha=0.5,
    )

fig.tight_layout()
file_name = "thesis/img/loss_dtw_alignment.png"
fig.savefig(file_name, dpi=300, bbox_inches="tight")
print(f"Erfolgreich generiert: {file_name}")
