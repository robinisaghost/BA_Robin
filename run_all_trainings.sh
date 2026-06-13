#!/bin/bash
# Run all alignment-loss trainings sequentially.
# Bounded-Lag (LSTM + PatchTST) → DTW (LSTM + PatchTST)
# Results land in reports/results/

set -e
PYTHON=".venv/Scripts/python"
export PYTHONPATH="src"
LOG_DIR="reports/logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Starting all alignment-loss trainings"
echo "$(date)"
echo "========================================"

# ── 1. Bounded-Lag ───────────────────────────────────────────────────────────
echo ""
echo "[1/4] LSTM bounded-lag  (branch: devBranch-offset-loss-bounded-lag)"
git checkout devBranch-offset-loss-bounded-lag
$PYTHON scripts/train_lstm_bounded_lag.py 2>&1 | tee "$LOG_DIR/lstm_bounded_lag.log"
echo "[1/4] DONE — $(date)"

echo ""
echo "[2/4] PatchTST bounded-lag  (branch: devBranch-offset-loss-bounded-lag)"
$PYTHON scripts/train_patchtst_bounded_lag.py 2>&1 | tee "$LOG_DIR/patchtst_bounded_lag.log"
echo "[2/4] DONE — $(date)"

# ── 2. DTW ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] LSTM DTW  (branch: devBranch-offset-loss-dtw)"
git checkout devBranch-offset-loss-dtw
$PYTHON scripts/train_lstm_dtw.py 2>&1 | tee "$LOG_DIR/lstm_dtw.log"
echo "[3/4] DONE — $(date)"

echo ""
echo "[4/4] PatchTST DTW  (branch: devBranch-offset-loss-dtw)"
$PYTHON scripts/train_patchtst_dtw.py 2>&1 | tee "$LOG_DIR/patchtst_dtw.log"
echo "[4/4] DONE — $(date)"

echo ""
echo "========================================"
echo "All trainings complete — $(date)"
echo "Results in reports/results/"
echo "Logs    in reports/logs/"
echo "========================================"
