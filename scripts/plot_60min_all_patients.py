from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("reports/results")
FIG_DIR = Path("reports/figures/60min_compare_window")

LSTM_PATH = RESULTS_DIR / "lstm_60min_traces_all_patients.npz"
PATCHTST_PATH = RESULTS_DIR / "patchtst_60min_traces_all_patients.npz"


def extract_patient_ids(npz_file):
    ids = set()
    for key in npz_file.files:
        if key.endswith("_true"):
            ids.add(key[:-5])
        elif key.endswith("_pred"):
            ids.add(key[:-5])
    return sorted(ids)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lstm_data = np.load(LSTM_PATH)
    patchtst_data = np.load(PATCHTST_PATH)

    lstm_ids = set(extract_patient_ids(lstm_data))
    patchtst_ids = set(extract_patient_ids(patchtst_data))
    patient_ids = sorted(lstm_ids & patchtst_ids)

    print(f"Found {len(patient_ids)} common patients")

    window_len = 720
    start = 0

    for pid in patient_ids:
        true_key = f"{pid}_true"
        pred_key = f"{pid}_pred"

        if true_key not in lstm_data or pred_key not in lstm_data:
            print(f"Skipping patient {pid}: missing LSTM keys")
            continue
        if true_key not in patchtst_data or pred_key not in patchtst_data:
            print(f"Skipping patient {pid}: missing PatchTST keys")
            continue

        y_true_lstm = lstm_data[true_key]
        y_true_patchtst = patchtst_data[true_key]
        y_lstm = lstm_data[pred_key]
        y_patchtst = patchtst_data[pred_key]

        if len(y_true_lstm) != len(y_true_patchtst):
            print(f"Skipping patient {pid}: mismatched ground truth lengths")
            continue

        if not np.allclose(y_true_lstm, y_true_patchtst):
            print(f"Warning for patient {pid}: ground truth differs between files")

        y_true = y_true_lstm

        end = min(start + window_len, len(y_true))
        x = np.arange(start, end)

        plt.figure(figsize=(20, 5))
        plt.plot(x, y_true[start:end], label="Ground truth")
        plt.plot(x, y_lstm[start:end], label="LSTM")
        plt.plot(x, y_patchtst[start:end], label="PatchTST")

        plt.title(
            f"60-min prediction comparison - patient {pid} (window {start}:{end})"
        )
        plt.xlabel("Window index")
        plt.ylabel("Glucose (mg/dL)")
        plt.legend()
        plt.tight_layout()

        out_path = FIG_DIR / f"patient_{pid}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"Saved windowed plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
