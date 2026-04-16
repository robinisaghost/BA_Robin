"""
Bounded-lag alignment loss for offset-aware training of glucose forecasting models.

Standard pointwise losses (e.g. MSE) penalise predictions that are correct in
shape but shifted in time. This module implements a loss that tolerates small
temporal misalignments within a bounded window D, so that a model is not
penalised for a prediction that is merely shifted by a few steps.

The bounded-lag search is a discrete, winner-takes-all variant of
temporal-alignment losses. The closest prior work is DILATE [12], which
jointly optimises shape and temporal position. Soft-DTW [13] provides the
differentiable-DTW formulation that underlies such alignment objectives. The
±max_lag window is directly motivated by the Sakoe-Chiba band [14], which
constrains dynamic time warping to a diagonal strip of bounded width.

The objective is applied to this project following the thesis proposal by
van den Hoek [7]. The time-shift phenomenon that motivates this loss was
observed in the PRG Proposal [11].

Based on
---------
- Le Guen & Thome [12] — DILATE loss, which jointly optimises shape and
  temporal position; the bounded-lag loss is a simpler discrete variant.
  Reference implementation: https://github.com/vincent-leguen/DILATE
- Cuturi & Blondel [13] — Soft-DTW; informed the differentiable alignment
  framing. Reference: https://github.com/mblondel/soft-dtw
- Sakoe & Chiba [14] — Sakoe-Chiba band motivates the ±max_lag window.

References
----------
[7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
     Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
     University of Bern, Faculty of Science (INF).
     Supervisor: PD Dr. Kaspar Riesen.

[11] Pattern Recognition Group, University of Bern. Glucose Prediction
     Proposal. Internal unpublished manuscript.

[12] Le Guen, V., & Thome, N. (2019). Shape and time distortion loss for
     training deep time series forecasting models. In Advances in Neural
     Information Processing Systems 32 (NeurIPS 2019).
     https://proceedings.neurips.cc/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html

[13] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
     function for time-series. In Proceedings of the 34th International
     Conference on Machine Learning (ICML 2017), vol. 70, pp. 894-903. PMLR.
     https://proceedings.mlr.press/v70/cuturi17a.html

[14] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm
     optimization for spoken word recognition. IEEE Transactions on
     Acoustics, Speech, and Signal Processing, 26(1), 43-49.
     https://doi.org/10.1109/TASSP.1978.1163055
"""

import torch


def bounded_lag_mse(
    pred: torch.Tensor, true: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """
    MSE loss with best-lag alignment within ±max_lag steps.

    For each sample in the batch the lag k* in [-max_lag, max_lag] that
    minimises MSE is found. The loss is then computed at that alignment.
    Gradients flow through the MSE at k* (winner-takes-all), not through
    the argmin, which is consistent with structured-prediction practice for
    non-differentiable search steps.

    To avoid a slice-length bias, the ground-truth tensor must be extended
    by 2*max_lag steps relative to the prediction horizon.  For every shift k
    the comparison is always over exactly H = pred.shape[1] values:

        k = -max_lag : pred[0:H] vs true[0         : H       ]
        k =  0       : pred[0:H] vs true[max_lag   : max_lag+H]
        k = +max_lag : pred[0:H] vs true[2*max_lag : 2*max_lag+H]

    Without this extension, larger shifts would compare fewer values
    (H - |k| instead of H), artificially lowering the MSE and incentivising
    the model to produce shifted outputs.

    The alignment window D = max_lag is directly motivated by the
    Sakoe-Chiba band [14], which constrains DTW to a diagonal strip of
    bounded width. D should be chosen to cover clinically plausible timing
    offsets. For a 60-minute horizon (12 steps à 5 min), D = 3 (±15 min)
    is used as the default, following van den Hoek [7].

    Parameters
    ----------
    pred : torch.Tensor
        Predicted trajectory of shape (batch, H).
    true : torch.Tensor
        Extended ground-truth trajectory of shape (batch, H + 2*max_lag).
        Must be constructed from a target window that starts max_lag steps
        before and ends max_lag steps after the standard horizon window.
    max_lag : int
        Maximum alignment shift D in both directions (in time steps).

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.

    References
    ----------
    [7]  van den Hoek, R. (2026). Mitigating Time-Shift Errors in CGM-based
         Glucose Forecasting and Hypoglycemia Event Prediction. Bachelor Thesis,
         University of Bern, Faculty of Science (INF).
         Supervisor: PD Dr. Kaspar Riesen.

    [12] Le Guen, V., & Thome, N. (2019). Shape and time distortion loss for
         training deep time series forecasting models. NeurIPS 2019.

    [13] Cuturi, M., & Blondel, M. (2017). Soft-DTW: a differentiable loss
         function for time-series. ICML 2017, pp. 894-903.

    [14] Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm
         optimization for spoken word recognition. IEEE TASSP, 26(1), 43-49.
    """
    H = pred.shape[1]
    best_mse = None  # shape: (batch,)

    for k in range(-max_lag, max_lag + 1):
        # offset maps k ∈ [-max_lag, max_lag] to [0, 2*max_lag]
        # t_slice always has exactly H values regardless of k
        offset = k + max_lag
        t_slice = true[:, offset : offset + H]

        mse_k = ((pred - t_slice) ** 2).mean(dim=1)  # (batch,)

        if best_mse is None:
            best_mse = mse_k
        else:
            # Gradient flows through whichever k achieved the minimum (winner-takes-all).
            best_mse = torch.minimum(best_mse, mse_k)

    return best_mse.mean()
