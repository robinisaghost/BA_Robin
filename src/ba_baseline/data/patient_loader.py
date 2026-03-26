"""
Patient data loader for CGM glucose time series.

Loads continuous glucose monitoring (CGM) data from a CSV file and returns
a per-patient dictionary of glucose time series. The data originates from
the T1DATA dataset [9], a randomised crossover
clinical trial in which 35 individuals with type 1 diabetes used Dexcom G6
CGM devices recording glucose at 5-minute intervals over approximately 30 days.

References

[9] Garcia-Tirado, J., Colmegna, P., Villard, O., Diaz, J. L., Esquivel-Zuniga,
    R., Koravi, C. L. K., Barnett, C. L., Oliveri, M. C., Fuller, M.,
    Brown, S. A., DeBoer, M. D., & Breton, M. D. (2023). Assessment of meal
    anticipation for improving fully automated insulin delivery in adults with
    type 1 diabetes. Diabetes Care, 46(9), 1652–1658.
    https://doi.org/10.2337/dc23-0119
"""

import numpy as np
import pandas as pd


def load_patient_series(
    csv_path: str,
    glucose_col: str = "glucose_level",
    patient_col: str = "patient_ID",
):
    """
    Load per-patient CGM glucose time series from a CSV file.

    Parameters

    csv_path : str
        Path to the CSV file containing glucose readings and patient IDs.
    glucose_col : str
        Name of the column containing glucose values (mg/dL).
    patient_col : str
        Name of the column containing patient identifiers.

    Returns

    dict
        Mapping from patient ID (str) to a NumPy array of glucose values
        in file order, which is assumed to correspond to temporal order.

    References

    [9] Garcia-Tirado, J., Colmegna, P., Villard, O., Diaz, J. L.,
        Esquivel-Zuniga, R., Koravi, C. L. K., Barnett, C. L., Oliveri, M. C.,
        Fuller, M., Brown, S. A., DeBoer, M. D., & Breton, M. D. (2023).
        Assessment of meal anticipation for improving fully automated insulin
        delivery in adults with type 1 diabetes. Diabetes Care, 46(9),
        1652–1658. https://doi.org/10.2337/dc23-0119
    """
    df = pd.read_csv(csv_path)

    if glucose_col not in df.columns or patient_col not in df.columns:
        raise ValueError(
            f"Expected columns {glucose_col}, {patient_col}, and got {list(df.columns)}"
        )

    df = df[[glucose_col, patient_col]].copy()
    df[glucose_col] = pd.to_numeric(df[glucose_col], errors="coerce")
    df = df.dropna(subset=[patient_col])

    series_by_patient = {}
    for pid, g in df.groupby(patient_col, sort=False)[glucose_col]:
        # Impute missing values via linear interpolation, as specified in
        # the PRG Proposal [11]. Leading/trailing NaN that cannot
        # be interpolated are dropped.
        g = g.interpolate(method="linear")
        g = g.dropna()
        series_by_patient[str(pid)] = g.to_numpy(dtype=np.float32)

    return series_by_patient
