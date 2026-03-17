"""
Bounded-lag alignment loss for offset-aware training of glucose forecasting models.

Standard pointwise losses (e.g. MSE) penalise predictions that are correct in
shape but shifted in time. This module implements a loss that tolerates small
temporal misalignments within a bounded window D, so that a model is not
penalised for a prediction that is merely shifted by a few steps.

The approach and motivation are described in:
    van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose
    Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University
    of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.

The time-shift phenomenon that motivates this loss was observed in:
    Pattern Recognition Group, University of Bern. Glucose Prediction Proposal.
    Internal unpublished manuscript.

References
----------
van den Hoek (2026)
    Bachelor Thesis Proposal, University of Bern.
Pattern Recognition Group, University of Bern
    Internal glucose prediction proposal.
"""

import torch


def bounded_lag_mse(
    pred: torch.Tensor, true: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """
    MSE loss with best-lag alignment within ±max_lag steps.

    For each sample in the batch the lag k* in [-max_lag, max_lag] that
    minimises MSE on the overlapping slice is found. The loss is then computed
    at that alignment. Gradients flow through the MSE at k* (winner-takes-all),
    not through the argmin, which is consistent with structured-prediction
    practice for non-differentiable search steps.

    The alignment window D = max_lag should be chosen to cover clinically
    plausible timing offsets. For a 60-minute horizon (12 steps à 5 min),
    D = 3 (±15 min) is used as the default, following van den Hoek (2026).

    Parameters
    ----------
    pred : torch.Tensor
        Predicted trajectory of shape (batch, horizon).
    true : torch.Tensor
        Ground-truth trajectory of shape (batch, horizon).
    max_lag : int
        Maximum alignment shift D in both directions (in time steps).

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.

    References
    ----------
    van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based Glucose
    Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis, University
    of Bern, Faculty of Science (INF). Supervisor: PD Dr. Kaspar Riesen.
    """
    H = pred.shape[1]
    best_mse = None  # shape: (batch,)

    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            p_slice = pred[:, :H - k]   # pred aligned left
            t_slice = true[:, k:]       # true shifted right
        elif k < 0:
            p_slice = pred[:, -k:]      # pred shifted right
            t_slice = true[:, :H + k]   # true aligned left
        else:
            p_slice = pred
            t_slice = true

        mse_k = ((p_slice - t_slice) ** 2).mean(dim=1)  # (batch,)

        if best_mse is None:
            best_mse = mse_k
        else:
            # Gradient flows through whichever k achieved the minimum (winner-takes-all).
            best_mse = torch.minimum(best_mse, mse_k)

    return best_mse.mean()
