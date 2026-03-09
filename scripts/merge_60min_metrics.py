from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("reports/results")

lstm_path = RESULTS_DIR / "lstm_60min_per_patient_metrics_all.csv"
patchtst_path = RESULTS_DIR / "patchtst_60min_per_patient_metrics_all.csv"
out_path = RESULTS_DIR / "combined_60min_per_patient_metrics.csv"

lstm = pd.read_csv(lstm_path)
patchtst = pd.read_csv(patchtst_path)

combined = pd.concat([lstm, patchtst], ignore_index=True)
combined = combined.sort_values(["patient_id", "model"]).reset_index(drop=True)

combined.to_csv(out_path, index=False)

print(f"Saved merged metrics to {out_path}")
print(combined)
