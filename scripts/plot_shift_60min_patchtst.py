import os
import json
import numpy as np
import matplotlib.pyplot as plt

from ba_baseline.metrics.metrics import best_lag_rmse, shift_1d


def shift_series(x: np.ndarray, lag: int) -> np.ndarray:
    """Return shifted copy for visualization (introduces NaNs)."""
    y = x.astype(float).copy()
    if lag > 0:
        y[:lag] = np.nan
        y[lag:] = x[:-lag]
    elif lag < 0:
        L = -lag
        y[-L:] = np.nan
        y[:-L] = x[L:]
    return y


def main():
    os.makedirs("reports/figures", exist_ok=True)

    # --- load summary ---
    with open("reports/results/patchtst_60min_summary.json", "r", encoding="utf8") as f:
        summary = json.load(f)

    # --- load lags ---
    lags = []
    pids = []
    with open("reports/results/patchtst_60min_lags.csv", "r", encoding="utf8") as f:
        next(f)
        for line in f:
            pid, lag = line.strip().split(",")
            pids.append(pid)
            lags.append(int(lag))
    lags = np.array(lags)

    # 1) Histogram of best lags
    plt.figure()
    bins = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
    plt.hist(lags, bins=bins)
    plt.title("PatchTST: best-lag distribution (1-step, per patient)")
    plt.xlabel("lag (steps)")
    plt.ylabel("number of patients")
    plt.tight_layout()
    plt.savefig("reports/figures/patchtst_lag_hist_60min.png", dpi=200)
    plt.close()

    # 2) Example overlay plots
    data = np.load("reports/results/patchtst_60min_traces_examples.npz")
    example_pids = sorted({k.split("_")[0] for k in data.keys()})

    for pid in example_pids:
        t = data[f"{pid}_true"]
        p = data[f"{pid}_pred"]

        # choose a clean window for plotting
        N = min(600, len(t))
        t = t[:N]
        p = p[:N]

        lag = best_lag_rmse(t, p, max_lag=24)  # lag is the correction to apply
        p_shift = shift_1d(p, lag)
        print(
            pid,
            "true std",
            float(np.std(t)),
            "pred std",
            float(np.std(p)),
            "pred min/max",
            float(np.min(p)),
            float(np.max(p)),
        )
        plt.figure()
        plt.plot(t, label="true", linewidth=2)
        plt.plot(
            p_shift, label=f"pred (shift-corrected, lag={lag})", linewidth=2, alpha=0.8
        )
        plt.plot(p, label="pred (raw)", linewidth=2, alpha=0.8, linestyle="--")

        plt.title(f"PatchTST overlay (1-step trace) | patient {pid}")
        plt.xlabel("time index (5-min steps, relative)")
        plt.ylabel("glucose")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"reports/figures/patchtst_overlay_patient_{pid}_60min.png", dpi=200
        )
        plt.close()

    # 3) Save a small text summary for easy copy into report
    with open("reports/figures/patchtst_60min_summary.txt", "w", encoding="utf8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("Saved figures to reports/figures/")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
