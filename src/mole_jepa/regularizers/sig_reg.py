"""SIGReg: Sketched Isotropic Gaussian Regularization."""

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Applies a univariate test statistic to random projections of the input
    embeddings onto unit vectors sampled from the hypersphere, then averages
    the per-direction statistics. Formally:

        SIGReg_T(A, {z_n}) = (1 / |A|) * sum_{a in A} T({a^T z_n}_{n=1}^N)

    where A = {a_1, ..., a_M} are random unit-norm directions drawn from
    S^{K-1} and T is a univariate test statistic. A fresh set of directions
    is sampled on each forward pass.

    Args:
        test_statistic: A univariate test statistic module whose ``forward``
            accepts a 1-D tensor of shape `(n,)` and returns a scalar.
        n_directions: Number of random projection directions M.

    Notes:
        Definition from Balestriero & LeCun, "LeJEPA" (2025), Definition 2.
    """

    def __init__(
        self,
        test_statistic: nn.Module,
        n_directions: int = 128,
    ) -> None:
        super().__init__()
        self.test_statistic = test_statistic
        self.n_directions = n_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg loss for a batch of embeddings.

        Args:
            x: Tensor of shape `(n, d)` containing a batch of `n` embeddings
                of dimension `d`.

        Returns:
            Scalar regularization loss.
        """
        n, d = x.shape

        # Sample M random unit vectors on S^{d-1}: (M, d)
        directions = torch.randn(self.n_directions, d, device=x.device, dtype=x.dtype)
        directions = directions / directions.norm(dim=1, keepdim=True)

        # Project all embeddings onto each direction: (n, M)
        projections = x @ directions.T

        # Apply the test statistic to each direction's scalar projections and average.
        return torch.stack(
            [self.test_statistic(projections[:, m]) for m in range(self.n_directions)]
        ).mean()
