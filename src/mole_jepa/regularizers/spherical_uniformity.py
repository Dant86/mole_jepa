"""Wang & Isola spherical uniformity regularizer."""

import torch
import torch.nn.functional as F
from torch import nn


class SphericalUniformity(nn.Module):
    """Wang & Isola (2020) uniformity loss on the unit hypersphere.

    Measures how uniformly a set of embeddings is distributed on S^{d-1}.
    Formally:

        L_uniform(z) = log E_{i,j}[ exp(-2 * ||z̃_i - z̃_j||²) ]

    where z̃_i = z_i / ||z_i|| are the L2-normalised embeddings and the
    expectation is approximated by the empirical mean over all pairs in the
    batch (including self-pairs, which contribute 1 to the kernel and become
    negligible as n → ∞).

    For unit vectors the squared distance simplifies to
    ||z̃_i - z̃_j||² = 2 − 2 z̃_i · z̃_j, so the kernel is
    exp(4 z̃_i · z̃_j − 4), making the computation a single matmul.

    Theoretical guarantee: in the population limit this loss is uniquely
    minimised by the uniform distribution on S^{d-1}.  All moments of the
    Gaussian kernel are simultaneously minimised at isotropy
    (Σ = (1/d)I), which characterises the uniform distribution.

    Reference:
        Wang & Isola, "Understanding Contrastive Representation Learning
        through Alignment and Uniformity on the Hypersphere", ICML 2020.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the uniformity loss for a batch of embeddings.

        Args:
            z: Tensor of shape ``(n, d)`` containing raw (unnormalised)
               embeddings.  Normalisation to S^{d-1} is applied internally.

        Returns:
            Scalar uniformity loss.
        """
        z_norm = F.normalize(z, dim=-1)  # (n, d) — project to sphere
        gram = z_norm @ z_norm.T  # (n, n) — cosine similarities
        sq_dists = 2.0 - 2.0 * gram  # ||z̃_i - z̃_j||²
        return sq_dists.mul(-2.0).exp().mean().log()
