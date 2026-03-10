from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("reports/results")

in_path = RESULTS_DIR / "combined_60min_per_patient_metrics.csv"
out_path = RESULTS_DIR / "comparison_60min_per_patient.csv"

df = pd.read_csv(in_path)

pivot = df.pivot(index="patient_id", columns="model")
pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
pivot = pivot.reset_index()

pivot["rmse_diff"] = pivot["patchtst_rmse"] - pivot["lstm_rmse"]
pivot["mae_diff"] = pivot["patchtst_mae"] - pivot["lstm_mae"]

pivot = pivot[
    [
        "patient_id",
        "lstm_rmse",
        "patchtst_rmse",
        "rmse_diff",
        "lstm_mae",
        "patchtst_mae",
        "mae_diff",
    ]
].sort_values("patient_id")

pivot.to_csv(out_path, index=False)

print(f"Saved comparison table to {out_path}")
print(pivot)

print()
print("Summary:")
print("LSTM better on RMSE:", int((pivot["rmse_diff"] < 0).sum()))
print("PatchTST better on RMSE:", int((pivot["rmse_diff"] > 0).sum()))
print("Equal RMSE:", int((pivot["rmse_diff"] == 0).sum()))
print("LSTM better on MAE:    ", int((pivot["mae_diff"] < 0).sum()))
print("PatchTST better on MAE:", int((pivot["mae_diff"] > 0).sum()))
print("Equal MAE:             ", int((pivot["mae_diff"] == 0).sum()))
