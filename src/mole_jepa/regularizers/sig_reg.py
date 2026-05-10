"""SIGReg: Sketched Isotropic Gaussian Regularization."""

import torch
from torch import nn


class SIGReg[TestT: nn.Module](nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Applies a univariate test statistic to random projections of the input
    embeddings onto unit vectors sampled from the hypersphere, then averages
    the per-direction statistics. Formally:

        SIGReg_T(A, {z_n}) = (1 / |A|) * sum_{a in A} T({a^T z_n}_{n=1}^N)

    where A = {a_1, ..., a_M} are random unit-norm directions drawn from
    S^{K-1} and T is a univariate test statistic. A fresh set of directions
    is sampled on each forward pass.

    Args:
        test_statistic: Univariate test statistic module. Must support batched
            input of shape `(n, M)` and return `(M,)`. All
            :class:`~mole_jepa.test_statistics.EppsPulley` subclasses satisfy
            this contract.
        n_directions: Number of random projection directions M.
        demean: If ``True``, subtract the batch mean from ``x`` before
            projecting. This removes the implicit zero-mean constraint of the
            isotropic Gaussian target, allowing each modality to sit at its
            own offset in the embedding space (preserving the "modality gap")
            while still enforcing isotropic covariance structure.

    Notes:
        Definition from Balestriero & LeCun, "LeJEPA" (2025), Definition 2.
    """

    def __init__(
        self,
        test_statistic: TestT,
        n_directions: int = 128,
        demean: bool = False,
    ) -> None:
        super().__init__()
        self.test_statistic = test_statistic
        self.n_directions = n_directions
        self.demean = demean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg loss for a batch of embeddings.

        Args:
            x: Tensor of shape `(n, d)` containing a batch of `n` embeddings
                of dimension `d`.

        Returns:
            Scalar regularization loss.
        """
        _, d = x.shape

        if self.demean:
            x = x - x.mean(dim=0, keepdim=True)

        # Sample M random unit vectors on S^{d-1}: (M, d)
        directions = torch.randn(self.n_directions, d, device=x.device, dtype=x.dtype)
        directions = directions / directions.norm(dim=1, keepdim=True)

        # Project all embeddings onto each direction: (n, M)
        projections = x @ directions.T

        # Evaluate all M directions in one batched call and average.
        return self.test_statistic(projections).mean()
