from ba_baseline.data.patient_loader import load_patient_series
from ba_baseline.data.split import split_patients
from ba_baseline.data.multi_patient_dataset import MultiPatientWindowDataset

d = load_patient_series("data/raw/all_cgm.csv")
train_ids, val_ids, test_ids = split_patients(
    d.keys(), train_ratio=0.7, val_ratio=0.15, seed=42
)

print("train/val/test patients:", len(train_ids), len(val_ids), len(test_ids))
print("example train ids:", train_ids[:5])

lookback = 72  # 6h at 5-min steps (placeholder)
horizon = 12  # 60min

train_ds = MultiPatientWindowDataset(d, train_ids, lookback=lookback, horizon=horizon)
val_ds = MultiPatientWindowDataset(d, val_ids, lookback=lookback, horizon=horizon)

print("train samples:", len(train_ds))
print("val samples:", len(val_ds))

x, y = train_ds[0]
print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
