"""
Plot generation for BA presentation - v3

Generates all figures used in the slides:
  1. timeshift_demo.png         — LSTM and PatchTST predictions vs. truth
  2. tau_sweep.png              — Hypo F1 vs. tolerance window tau (BASELINE models)
  3. pr_paradox.png             — Precision-Recall plot with F1 iso-curves
  4. confusion_example.png      — LSTM vs PatchTST classifier output probabilities
                                  (the "output saturation" / overconfidence finding)
  5. hypo_episode.png           — Illustration of a hypoglycemia event (motivation)
  6. confusion_counts.png       — TP / FP / FN bar chart (optional, kept for reference)

Usage
-----
    python make_plots.py
    python make_plots.py --traces reports/results --out figures \\
                         --patient 85106 --classifier_patient 85214

Inputs
------
For plots 1, 2 and 5 (forecasting traces):
    {traces}/lstm_60min_traces_all_patients.npz
    {traces}/patchtst_60min_traces_all_patients.npz

For plot 4 (classifier predictions):
    {traces}/lstm_event_traces_all_patients.npz
    {traces}/patchtst_event_traces_all_patients.npz

For plots 3 and 6 — values hardcoded from the per-patient CSVs.

Dependencies: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Color palette aligned with the UniBern Beamer theme
COLOR_TRUE = "#222222"
COLOR_LSTM = "#0050AA"
COLOR_PATCHTST = "#C8102E"
COLOR_HYPO = "#888888"
COLOR_GUIDE = "#999999"
COLOR_TRUE_EVENT = "#2E8B57"  # green for true hypo events

HYPO_THRESHOLD = 70.0
SAMPLING_MIN = 5  # 5-minute CGM sampling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_patients(npz):
    keys = npz.files
    pids = sorted({k.replace("_true", "").replace("_pred", "") for k in keys})
    return [p for p in pids if f"{p}_true" in keys and f"{p}_pred" in keys]


def find_hypo_window(y_true, win_size=288):
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n <= win_size:
        return 0, n

    crossings = []
    for t in range(1, n):
        if y_true[t - 1] >= HYPO_THRESHOLD and y_true[t] < HYPO_THRESHOLD:
            crossings.append(t)

    if crossings:
        best_score = -np.inf
        best_t = crossings[len(crossings) // 2]
        for t in crossings:
            s = max(0, t - win_size // 2)
            e = min(n, s + win_size)
            score = float(np.std(y_true[s:e]))
            if score > best_score:
                best_score = score
                best_t = t
        center = best_t
    else:
        best_mean = np.inf
        center = win_size // 2
        for s in range(0, n - win_size, win_size // 4):
            m = float(np.mean(y_true[s : s + win_size]))
            if m < best_mean:
                best_mean = m
                center = s + win_size // 2

    start = max(0, center - win_size // 2)
    end = min(n, start + win_size)
    start = max(0, end - win_size)
    return start, end


def time_axis(start, end):
    """Auto-switch between minutes and hours depending on window length."""
    n_steps = end - start
    duration_min = n_steps * SAMPLING_MIN
    if duration_min <= 240:
        return np.arange(start, end) * SAMPLING_MIN, "Time [min]"
    return (np.arange(start, end) * SAMPLING_MIN) / 60.0, "Time [hours]"


# ---------------------------------------------------------------------------
# Plot 1: Time-shift demonstration
# ---------------------------------------------------------------------------
def plot_timeshift_demo(
    lstm_npz_path, patchtst_npz_path, patient_id, out_path, win_size=288
):
    if not os.path.exists(lstm_npz_path) or not os.path.exists(patchtst_npz_path):
        print("  [skip] missing trace file(s)", file=sys.stderr)
        return False

    lstm_data = np.load(lstm_npz_path)
    pt_data = np.load(patchtst_npz_path)

    pid = str(patient_id)
    if f"{pid}_true" not in lstm_data.files:
        lstm_pids = list_patients(lstm_data)
        if not lstm_pids:
            print("  [error] no patients in LSTM traces", file=sys.stderr)
            return False
        pid = lstm_pids[0]
        print(f"  [info] patient {patient_id} not found - using {pid}")

    if f"{pid}_true" not in pt_data.files or f"{pid}_pred" not in pt_data.files:
        print(f"  [warn] patient {pid} missing in PatchTST traces", file=sys.stderr)
        return False

    y_true = np.asarray(lstm_data[f"{pid}_true"])
    y_lstm = np.asarray(lstm_data[f"{pid}_pred"])
    y_pt = np.asarray(pt_data[f"{pid}_pred"])

    n = min(len(y_true), len(y_lstm), len(y_pt))
    y_true, y_lstm, y_pt = y_true[:n], y_lstm[:n], y_pt[:n]

    # Restrict to test portion (last 20%) so we show unseen data
    test_start = int(0.8 * n)
    y_true = y_true[test_start:]
    y_lstm = y_lstm[test_start:]
    y_pt = y_pt[test_start:]

    start, end = find_hypo_window(y_true, win_size=win_size)
    t = np.arange(end - start) * SAMPLING_MIN  # relative time within window, always minutes
    xlabel = "Time [min]"

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    for ax, y_pred, label, color in [
        (axes[0], y_lstm[start:end], "LSTM", COLOR_LSTM),
        (axes[1], y_pt[start:end], "PatchTST", COLOR_PATCHTST),
    ]:
        ax.plot(t, y_true[start:end], color=COLOR_TRUE, linewidth=1.8, label="True")
        ax.plot(
            t,
            y_pred,
            color=color,
            linewidth=1.6,
            label=f"{label} prediction",
            alpha=0.9,
        )
        ax.axhline(
            HYPO_THRESHOLD,
            color=COLOR_HYPO,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Hypo threshold (70 mg/dL)",
        )
        ax.set_ylabel("Glucose [mg/dL]")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=color,
            va="top",
        )
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel(xlabel)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 2: tau-sweep (all model variants, forecast-derived)
# ---------------------------------------------------------------------------
def detect_crossings(series, threshold=HYPO_THRESHOLD):
    s = np.asarray(series)
    events = []
    for t in range(1, len(s)):
        if s[t - 1] >= threshold and s[t] < threshold:
            events.append(t)
    return events


def f1_for_tau(npz_data, tau):
    total_tp = total_fp = total_fn = 0
    for pid in list_patients(npz_data):
        y_true = npz_data[f"{pid}_true"]
        y_pred = npz_data[f"{pid}_pred"]
        true_events = detect_crossings(y_true)
        pred_events = detect_crossings(y_pred)
        matched = set()
        tp = 0
        for te in true_events:
            for i, pe in enumerate(pred_events):
                if i in matched:
                    continue
                if abs(pe - te) <= tau:
                    matched.add(i)
                    tp += 1
                    break
        fp = len(pred_events) - tp
        fn = len(true_events) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
    if total_tp == 0:
        return 0.0
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    return 2 * precision * recall / max(precision + recall, 1e-12)


# (label, lstm_file, patchtst_file, linestyle, marker)
_VARIANTS = [
    (
        "Baseline (MSE)",
        "lstm_60min_traces_all_patients.npz",
        "patchtst_60min_traces_all_patients.npz",
        "-",
        "o",
    ),
    (
        "DTW",
        "lstm_dtw_traces_all_patients.npz",
        "patchtst_dtw_traces_all_patients.npz",
        "--",
        "s",
    ),
    (
        "Bounded Lag",
        "lstm_bounded_lag_traces_all_patients.npz",
        "patchtst_bounded_lag_traces_all_patients.npz",
        ":",
        "^",
    ),
    (
        "Multistep",
        "lstm_multistep_traces_all_patients.npz",
        "patchtst_multistep_traces_all_patients.npz",
        "-.",
        "D",
    ),
]


def plot_tau_sweep(traces_dir, out_path, max_tau_steps=12):
    taus = list(range(0, max_tau_steps + 1))
    tau_min = [t * SAMPLING_MIN for t in taus]

    results = []
    for label, lstm_file, pt_file, ls, marker in _VARIANTS:
        lstm_path = os.path.join(traces_dir, lstm_file)
        pt_path = os.path.join(traces_dir, pt_file)
        if not os.path.exists(lstm_path) or not os.path.exists(pt_path):
            print(f"  [warn] skipping {label}: file missing", file=sys.stderr)
            continue
        print(f"  [info] computing F1 for {label}...")
        lstm_f1 = [f1_for_tau(np.load(lstm_path), t) for t in taus]
        pt_f1 = [f1_for_tau(np.load(pt_path), t) for t in taus]
        results.append((label, lstm_f1, pt_f1, ls, marker))

    if not results:
        print("  [skip] no variant data found", file=sys.stderr)
        return False

    all_f1 = [v for _, lf, pf, _, _ in results for v in lf + pf]
    y_max = max(max(all_f1) * 1.15, 0.6)

    # One distinct color per model variant (same in both subplots for easy comparison)
    _SWEEP_COLORS = ["#222222", "#0072B2", "#D55E00", "#009E73"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, arch_label, get_f1 in [
        (axes[0], "LSTM", lambda lf, _: lf),
        (axes[1], "PatchTST", lambda _, pf: pf),
    ]:
        for (label, lstm_f1, pt_f1, ls, marker), color in zip(results, _SWEEP_COLORS):
            ax.plot(
                tau_min,
                get_f1(lstm_f1, pt_f1),
                linestyle=ls,
                marker=marker,
                color=color,
                linewidth=2.0,
                markersize=6,
                label=label,
            )

        ax.axvline(15, color=COLOR_GUIDE, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            15.7,
            y_max * 0.50,
            "$\\tau$ = 15 min",
            fontsize=9,
            va="bottom",
            color="#555555",
        )

        ax.set_xlabel("Tolerance window $\\tau$ [min]")
        ax.set_title(
            arch_label, fontsize=13, fontweight="bold", color="black", loc="left"
        )
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 62)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Hypo F1 score (forecast-derived crossings)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 3: Precision-Recall paradox
# ---------------------------------------------------------------------------
def plot_pr_paradox(out_path):
    points = [
        ("LSTM forecast-derived", 0.010, 0.004, "o", "#7799C2", False),
        ("LSTM Direct Classifier", 0.237, 0.693, "o", COLOR_LSTM, True),
        ("PatchTST forecast-derived", 0.048, 0.089, "s", "#E5A2AE", False),
        ("PatchTST Direct Classifier", 0.054, 0.615, "s", COLOR_PATCHTST, True),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    R_grid = np.linspace(0.005, 0.999, 400)
    for f1_val in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        with np.errstate(divide="ignore", invalid="ignore"):
            P_grid = f1_val * R_grid / (2 * R_grid - f1_val)
        valid = (P_grid > 0) & (P_grid <= 1.0)
        ax.plot(
            R_grid[valid],
            P_grid[valid],
            color="lightgray",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
        for r in [0.5, 0.6, 0.4, 0.7, 0.3, 0.8]:
            p = f1_val * r / (2 * r - f1_val)
            if 0.03 < p < 0.73:
                ax.text(
                    r,
                    p + 0.012,
                    f"F1={f1_val}",
                    fontsize=8,
                    color="gray",
                    ha="center",
                    va="bottom",
                    zorder=2,
                )
                break

    from matplotlib.lines import Line2D

    legend_handles = []
    for label, p, r, marker, color, bold in points:
        ax.scatter(
            r,
            p,
            s=240 if bold else 130,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=1.3,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [], [],
                linestyle="none",
                marker=marker,
                markersize=8 if bold else 6,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=1.0,
                label=label,
                color=color,
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=9.5, framealpha=0.95)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 4: Classifier output predictions (NEW - overconfidence finding)
# ---------------------------------------------------------------------------
def plot_classifier_predictions(
    lstm_event_npz, patchtst_event_npz, patient_id, out_path
):
    """
    Show LSTM vs PatchTST direct classifier output probabilities over time
    for one patient. Reveals that PatchTST has learned extreme logits
    (output saturates at 0 / 1), while LSTM produces calibrated probabilities.
    """
    if not os.path.exists(lstm_event_npz) or not os.path.exists(patchtst_event_npz):
        print("  [skip] missing classifier trace file(s)", file=sys.stderr)
        return False

    lstm_data = np.load(lstm_event_npz)
    pt_data = np.load(patchtst_event_npz)

    pid = str(patient_id)
    if f"{pid}_pred" not in lstm_data.files:
        pids = list_patients(lstm_data)
        if not pids:
            print("  [error] no patients in classifier traces", file=sys.stderr)
            return False
        pid = pids[0]
        print(f"  [info] patient {patient_id} not found - using {pid}")

    if f"{pid}_pred" not in pt_data.files:
        print(
            f"  [warn] patient {pid} missing in PatchTST event traces", file=sys.stderr
        )
        return False

    lstm_true = np.asarray(lstm_data[f"{pid}_true"])
    lstm_prob = np.asarray(lstm_data[f"{pid}_pred"])
    pt_prob = np.asarray(pt_data[f"{pid}_pred"])

    n = min(len(lstm_true), len(lstm_prob), len(pt_prob))
    lstm_true, lstm_prob, pt_prob = lstm_true[:n], lstm_prob[:n], pt_prob[:n]

    # Only show the test portion (last 20%). If the chosen patient has no
    # true events in that window, switch to the patient with the most test events.
    test_start = int(0.8 * n)
    if not np.any(lstm_true[test_start:] > 0.5):
        best_pid, best_count = pid, 0
        for alt in list_patients(lstm_data):
            if f"{alt}_pred" not in pt_data.files:
                continue
            at = np.asarray(lstm_data[f"{alt}_true"])
            ats = int(0.8 * len(at))
            cnt = int(np.sum(at[ats:] > 0.5))
            if cnt > best_count:
                best_count, best_pid = cnt, alt
        if best_pid != pid:
            print(
                f"  [info] patient {pid} has no test events — using {best_pid} "
                f"({best_count} test events)"
            )
            pid = best_pid
            lstm_true = np.asarray(lstm_data[f"{pid}_true"])
            lstm_prob = np.asarray(lstm_data[f"{pid}_pred"])
            pt_prob = np.asarray(pt_data[f"{pid}_pred"])
            n = min(len(lstm_true), len(lstm_prob), len(pt_prob))
            lstm_true, lstm_prob, pt_prob = lstm_true[:n], lstm_prob[:n], pt_prob[:n]
            test_start = int(0.8 * n)

    # Slice to test portion then find a window centred on the event cluster
    test_true = lstm_true[test_start:]
    test_lstm = lstm_prob[test_start:]
    test_pt = pt_prob[test_start:]

    WIN = min(600, len(test_true))
    event_idx = np.where(test_true > 0.5)[0]
    if len(event_idx) > 0:
        center = int(event_idx[len(event_idx) // 2])
        w_start = max(0, center - WIN // 2)
        w_end = min(len(test_true), w_start + WIN)
        w_start = max(0, w_end - WIN)
    else:
        w_start, w_end = 0, min(WIN, len(test_true))

    lstm_true = test_true[w_start:w_end]
    lstm_prob = test_lstm[w_start:w_end]
    pt_prob = test_pt[w_start:w_end]
    n = len(lstm_true)

    t = np.arange(n) * SAMPLING_MIN  # in minutes (relative within test set)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    for ax, prob, label, color in [
        (axes[0], lstm_prob, "LSTM", COLOR_LSTM),
        (axes[1], pt_prob, "PatchTST", COLOR_PATCHTST),
    ]:
        # Shade where the true label is positive (hypo event coming)
        true_pos = lstm_true > 0.5
        if true_pos.any():
            in_run = False
            run_start = 0
            for i in range(n):
                if true_pos[i] and not in_run:
                    run_start = i
                    in_run = True
                elif not true_pos[i] and in_run:
                    ax.axvspan(
                        t[run_start], t[i], color=COLOR_TRUE_EVENT, alpha=0.45, zorder=1
                    )
                    in_run = False
            if in_run:
                ax.axvspan(
                    t[run_start], t[-1], color=COLOR_TRUE_EVENT, alpha=0.45, zorder=1
                )

        # Probability curve with shading underneath
        ax.fill_between(t, 0, prob, color=color, alpha=0.18, zorder=2)
        ax.plot(t, prob, color=color, linewidth=2.0, zorder=3)

        # Decision threshold
        ax.axhline(
            0.5, color="#666666", linestyle="--", linewidth=1.2, alpha=0.8, zorder=2
        )

        legend_handles = [
            Patch(
                facecolor=COLOR_TRUE_EVENT,
                alpha=0.3,
                edgecolor="none",
                label="True hypo event",
            ),
            Line2D(
                [], [], color=color, linewidth=2.0, label=f"{label} Direct Classifier"
            ),
            Line2D(
                [],
                [],
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                label="Decision threshold (0.5)",
            ),
        ]
        leg = ax.legend(
            handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95,
            title=label,
        )
        leg.get_title().set_fontsize(11)
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_color(color)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Event probability")
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Time [min]")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 5: Hypoglycemia episode illustration (motivation slide)
# ---------------------------------------------------------------------------
def plot_hypo_episode(lstm_npz_path, patient_id, out_path, win_size=72):
    """
    Illustrate what a hypoglycemia event is, using the true glucose trace
    of one patient. Shows a short window (~6h by default) containing a
    clear threshold crossing.
    """
    if not os.path.exists(lstm_npz_path):
        print(f"  [skip] missing {lstm_npz_path}", file=sys.stderr)
        return False

    data = np.load(lstm_npz_path)
    pid = str(patient_id)
    if f"{pid}_true" not in data.files:
        pids = list_patients(data)
        if not pids:
            print("  [error] no patients", file=sys.stderr)
            return False
        pid = pids[0]
        print(f"  [info] patient {patient_id} not found - using {pid}")

    y_true = np.asarray(data[f"{pid}_true"])

    n = len(y_true)
    best_start = 0
    best_score = -np.inf
    for start in range(0, n - win_size, max(1, win_size // 8)):
        window = y_true[start : start + win_size]
        if window.min() >= HYPO_THRESHOLD:
            continue
        n_below = int(np.sum(window < HYPO_THRESHOLD))
        if not (3 <= n_below <= win_size // 2):
            continue
        amp = float(window.max() - window.min())
        below_idx = np.where(window < HYPO_THRESHOLD)[0]
        center_bias = -abs(float(np.mean(below_idx)) - win_size * 0.55)
        score = amp + 0.3 * center_bias
        if score > best_score:
            best_score = score
            best_start = start

    start = best_start
    end = min(n, start + win_size)
    y = y_true[start:end]
    t = np.arange(end - start) * SAMPLING_MIN

    if (end - start) * SAMPLING_MIN > 240:
        t = t / 60.0
        xlabel = "Time [hours]"
    else:
        xlabel = "Time [min]"

    cross_idx = None
    for i in range(1, len(y)):
        if y[i - 1] >= HYPO_THRESHOLD and y[i] < HYPO_THRESHOLD:
            cross_idx = i
            break

    fig, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(t, y, color=COLOR_TRUE, linewidth=2.2, label="CGM glucose (true)")
    ax.axhline(
        HYPO_THRESHOLD,
        color="#C8102E",
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label="Hypo threshold (70 mg/dL)",
    )

    below_mask = y < HYPO_THRESHOLD
    ax.fill_between(
        t,
        HYPO_THRESHOLD,
        y,
        where=below_mask,
        color="#C8102E",
        alpha=0.18,
        label="Hypoglycemia episode",
    )

    if cross_idx is not None:
        tc = t[cross_idx]
        yc = HYPO_THRESHOLD
        ax.scatter(
            [tc],
            [yc],
            s=80,
            color="#C8102E",
            zorder=5,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.annotate(
            r"Threshold crossing $t_c$",
            xy=(tc, yc),
            xytext=(tc + (t[-1] - t[0]) * 0.06, yc + 18),
            fontsize=10,
            color="#C8102E",
            arrowprops=dict(arrowstyle="->", color="#C8102E", lw=1.0),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Glucose [mg/dL]")
    ax.set_title(f"(Patient {pid})", loc="left", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    y_lo = max(40, float(y.min()) - 10)
    y_hi = max(180, float(y.max()) + 10)
    ax.set_ylim(y_lo, y_hi)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# Plot 6: Confusion counts (kept for reference, not used in slides)
# ---------------------------------------------------------------------------
def plot_confusion_counts(out_path):
    counts = {
        "LSTM Direct Classifier": {"TP": 2610, "FP": 7981, "FN": 868},
        "PatchTST Direct Classifier": {"TP": 2243, "FP": 36794, "FN": 1235},
    }
    fa_ratio = {
        "LSTM Direct Classifier": 7981 / 2610,
        "PatchTST Direct Classifier": 36794 / 2243,
    }

    categories = ["TP", "FP", "FN"]
    x = np.arange(len(categories))
    width = 0.36

    lstm_vals = [counts["LSTM Direct Classifier"][c] for c in categories]
    pt_vals = [counts["PatchTST Direct Classifier"][c] for c in categories]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    bars1 = ax.bar(
        x - width / 2,
        lstm_vals,
        width,
        color=COLOR_LSTM,
        edgecolor="black",
        linewidth=0.8,
        label="LSTM Direct Classifier",
    )
    bars2 = ax.bar(
        x + width / 2,
        pt_vals,
        width,
        color=COLOR_PATCHTST,
        edgecolor="black",
        linewidth=0.8,
        label="PatchTST Direct Classifier",
    )

    for bars, vals in [(bars1, lstm_vals), (bars2, pt_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 1.06,
                f"{v:,}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Count (log scale)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    annot = (
        f"False alarms per true event caught:\n"
        f"   LSTM:     {fa_ratio['LSTM Direct Classifier']:.1f}\n"
        f"   PatchTST: {fa_ratio['PatchTST Direct Classifier']:.1f}"
    )
    ax.text(
        0.02,
        0.97,
        annot,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            linewidth=0.8,
        ),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [ok]   {out_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for the BA presentation."
    )
    parser.add_argument(
        "--traces",
        default="reports/results",
        help="directory containing the .npz trace files",
    )
    parser.add_argument(
        "--out", default="figures", help="output directory for the PNG plots"
    )
    parser.add_argument(
        "--patient",
        default="85106",
        help="patient ID for the time-shift demo and hypo-episode plots",
    )
    parser.add_argument(
        "--classifier_patient",
        default="85214",
        help="patient ID for the classifier-predictions plot",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=72,
        help="hypo-episode window length in steps (default 72 = 6h)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lstm_npz = os.path.join(args.traces, "lstm_60min_traces_all_patients.npz")
    pt_npz = os.path.join(args.traces, "patchtst_60min_traces_all_patients.npz")
    lstm_event_npz = os.path.join(args.traces, "lstm_event_traces_all_patients.npz")
    pt_event_npz = os.path.join(args.traces, "patchtst_event_traces_all_patients.npz")

    print("Generating plots:")
    plot_timeshift_demo(
        lstm_npz, pt_npz, args.patient, os.path.join(args.out, "timeshift_demo.png")
    )
    plot_tau_sweep(args.traces, os.path.join(args.out, "tau_sweep.png"))
    plot_pr_paradox(os.path.join(args.out, "pr_paradox.png"))
    plot_classifier_predictions(
        lstm_event_npz,
        pt_event_npz,
        args.classifier_patient,
        os.path.join(args.out, "confusion_example.png"),
    )
    plot_hypo_episode(
        lstm_npz,
        args.patient,
        os.path.join(args.out, "hypo_episode.png"),
        win_size=args.win_size,
    )
    plot_confusion_counts(os.path.join(args.out, "confusion_counts.png"))
    print("\nDone. Copy the PNGs into the same folder as your .tex file.")


if __name__ == "__main__":
    main()
