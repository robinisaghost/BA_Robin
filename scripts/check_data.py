from ba_baseline.data.patient_loader import load_patient_series
import numpy as np

d = load_patient_series("data/raw/all_cgm.csv")

print("patients:", len(d))

lengths = np.array(sorted(len(v) for v in d.values()))
print("min_len:", int(lengths.min()))
print("median_len:", int(np.median(lengths)))
print("max_len:", int(lengths.max()))

# quick sanity ranges
mins = [float(np.min(s)) for s in d.values()]
maxs = [float(np.max(s)) for s in d.values()]
print("global_min_glucose:", float(np.min(mins)))
print("global_max_glucose:", float(np.max(maxs)))

pid0 = next(iter(d))
print("example_patient:", pid0)
print("first_10_values:", d[pid0][:10])
