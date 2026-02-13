# BA_Robin_baseline

Baseline for CGM-based glucose forecasting and hypoglycemia event prediction.

## Models
- LSTM
- PatchTST

## Metrics
- Regression: RMSE, MAE
- Shift: best lag via cross-correlation
- Events: Precision, Recall, F1, F2 (hypoglycemia threshold crossing)

## Structure
- src/: code
- configs/: experiment configs
- notebooks/: exploration
- reports/figures/: thesis plots
- data/: excluded from git
