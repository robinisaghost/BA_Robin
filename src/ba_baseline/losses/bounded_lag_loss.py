import torch


def bounded_lag_mse(
    pred: torch.Tensor, true: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """
    MSE loss with best-lag alignment within ±max_lag steps.

    For each sample in the batch, find the lag k* in [-max_lag, max_lag] that
    minimises MSE on the overlapping slice, then compute the loss at that
    alignment. Gradient flows through the MSE at k* (winner-takes-all),
    not through the argmin.

    Args:
        pred:     (batch, horizon) predicted trajectory
        true:     (batch, horizon) ground-truth trajectory
        max_lag:  maximum shift D in both directions

    Returns:
        Scalar loss tensor.
    """
    H = pred.shape[1]
    best_mse = None  # shape: (batch,)

    for k in range(-max_lag, max_lag + 1):
        if k > 0:
            p_slice = pred[:, : H - k]  # pred aligned left
            t_slice = true[:, k:]  # true shifted right
        elif k < 0:
            p_slice = pred[:, -k:]  # pred shifted right
            t_slice = true[:, : H + k]  # true aligned left
        else:
            p_slice = pred
            t_slice = true

        mse_k = ((p_slice - t_slice) ** 2).mean(dim=1)  # (batch,)

        if best_mse is None:
            best_mse = mse_k
        else:
            best_mse = torch.minimum(best_mse, mse_k)

    return best_mse.mean()
