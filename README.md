# BA_Robin_baseline

Baseline for 60-minute CGM glucose forecasting with LSTM and PatchTST.

## Scope

This repository contains a compact baseline pipeline for 60-minute blood glucose prediction based on CGM time series.

The current focus is:
- 60-minute forecasting only
- patient-based data loading and splitting
- two models:
  - LSTM
  - PatchTST
- evaluation of forecast error and temporal shift

## Models

- **LSTM**: recurrent baseline model for 60-minute forecasting
- **PatchTST**: transformer-based time-series baseline for 60-minute forecasting

## Metrics

Implemented evaluation includes:
- **RMSE**
- **MAE**
- **best lag via cross-correlation**
- event-based metrics for hypoglycemia-related threshold behavior:
  - **Precision**
  - **Recall**
  - **F1**
  - **F2**

## Repository structure

```text
scripts/
  check_data.py
  check_split_and_dataset.py
  plot_shift_60min.py
  train_lstm_60min.py
  train_patchtst_60min.py

src/ba_baseline/
  data/
    multi_patient_dataset.py
    patient_loader.py
    split.py
  metrics/
    metrics.py
  models/
    lstm.py
    patchtst.py