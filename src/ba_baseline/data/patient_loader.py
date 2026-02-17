import numpy as np
import pandas as pd


def load_patient_series(
    csv_path: str,
    glucose_col: str = "glucose_level",
    patient_col: str = "patient_ID",
):
    """
    Returns: dict {patient_id (str): np.ndarray of glucose values in file order}
    Assumption: file order corresponds to time order per patient.
    """
    df = pd.read_csv(csv_path)

    if glucose_col not in df.columns or patient_col not in df.columns:
        raise ValueError(
            f"Expected columns {glucose_col}, {patient_col}, got {list(df.columns)}"
        )

    df = df[[glucose_col, patient_col]].copy()
    df[glucose_col] = pd.to_numeric(df[glucose_col], errors="coerce")
    df = df.dropna(subset=[glucose_col, patient_col])

    series_by_patient = {}
    for pid, g in df.groupby(patient_col, sort=False)[glucose_col]:
        series_by_patient[str(pid)] = g.to_numpy(dtype=np.float32)

    return series_by_patient
