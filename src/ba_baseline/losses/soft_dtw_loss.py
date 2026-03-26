"""
Soft-DTW loss function for time-series alignment.

Dynamic Time Warping (DTW) measures the similarity between two sequences by
finding the optimal non-linear alignment path between them [13]. Unlike
standard MSE, DTW does not require point-wise correspondence: each step in
the prediction can be matched to a different step in the ground truth,
allowing the loss to be invariant to local temporal distortions.

Soft-DTW [12] replaces the hard minimum in the DTW dynamic programming
recursion with a differentiable soft-minimum (log-sum-exp), making the loss
function fully differentiable and suitable for gradient-based optimisation.

The smoothing parameter gamma controls the approximation:
  - gamma -> 0 recovers hard DTW (non-differentiable)
  - gamma -> inf approaches a sum over all alignment paths (MSE-like)
  - gamma = 1.0 is the standard value used in Cuturi & Blondel [12]

The objective is applied to this project following the thesis proposal by
van den Hoek [7].

References
----------
[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[12] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
     function for time-series. In Proceedings of the 34th International
     Conference on Machine Learning (ICML 2017), vol. 70, pp. 894-903. PMLR.
     https://proceedings.mlr.press/v70/cuturi17a.html

[13] Berndt, D. J., & Clifford, J. (1994). Using dynamic time warping to
     find patterns in time series. In Proceedings of the AAAI Workshop on
     Knowledge Discovery in Databases (KDD 1994), pp. 359-370.
"""

import torch


def soft_dtw(pred: torch.Tensor, true: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Soft-DTW loss between batched prediction and ground-truth sequences.

    For each sample in the batch, computes the soft-DTW distance between the
    predicted and true horizon sequences using dynamic programming with a
    differentiable soft-minimum. The batch mean is returned as a scalar loss.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted sequence of shape (B, H), where B is batch size and H is
        the forecast horizon.
    true : torch.Tensor
        Ground-truth sequence of shape (B, H).
    gamma : float
        Smoothing parameter gamma > 0. Default 1.0 following Cuturi &
        Blondel [12].

    Returns
    -------
    torch.Tensor
        Scalar loss (mean soft-DTW distance over the batch).

    References
    ----------
    [12] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
         function for time-series. ICML 2017, pp. 894-903.
    """
    B, H = pred.shape

    # Pairwise squared cost matrix: D[b, i, j] = (pred[b,i] - true[b,j])^2
    # Shape: (B, H, H)
    D = (pred.unsqueeze(2) - true.unsqueeze(1)) ** 2

    # DP table R[b, i, j] = soft-DTW value for subsequences pred[0..i-1]
    # and true[0..j-1].  Rows/cols 0 are sentinels initialised to +inf,
    # with R[:,0,0] = 0 as the base case.
    R = torch.full((B, H + 1, H + 1), float("inf"), device=pred.device, dtype=pred.dtype)
    R[:, 0, 0] = 0.0

    for i in range(1, H + 1):
        for j in range(1, H + 1):
            # Three predecessor cells (diagonal, top, left)
            r0 = R[:, i - 1, j - 1]  # diagonal
            r1 = R[:, i - 1, j]      # top
            r2 = R[:, i, j - 1]      # left

            # Soft-minimum via log-sum-exp:
            # softmin(a,b,c) = -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))
            stacked = torch.stack([-r0 / gamma, -r1 / gamma, -r2 / gamma], dim=1)
            softmin = -gamma * torch.logsumexp(stacked, dim=1)

            R[:, i, j] = D[:, i - 1, j - 1] + softmin

    # Return the mean soft-DTW distance over the batch
    return R[:, H, H].mean()
