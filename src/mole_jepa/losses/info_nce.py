"""InfoNCE contrastive loss."""

import torch
from torch import nn
from torch.nn import functional


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE (NT-Xent) contrastive loss.

    Given two batches of embeddings where the i-th sample in each batch forms
    a positive pair, this loss encourages aligned pairs to be close and
    misaligned pairs to be far apart in embedding space.

    Both inputs are L2-normalised before similarity is computed. The loss is
    the average of the two cross-entropy terms (image-to-text and
    text-to-image):

        L = (CE(S, labels) + CE(Sᵀ, labels)) / 2

    where ``S_{ij} = z1_i · z2_j / τ`` and ``labels = [0, 1, …, B-1]``.

    Args:
        temperature: Softmax temperature τ. Smaller values sharpen the
            distribution. Defaults to ``0.07``.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric InfoNCE loss for a batch of pairs.

        Args:
            z1: First set of embeddings of shape `(B, d)`.
            z2: Second set of embeddings of shape `(B, d)`.

        Returns:
            Scalar loss.
        """
        # L2-normalise along the embedding dimension.
        u = functional.normalize(z1, dim=-1)  # (B, d)
        v = functional.normalize(z2, dim=-1)  # (B, d)

        # Pairwise cosine similarities scaled by temperature: (B, B)
        logits = u @ v.T / self.temperature

        # Positive pairs lie on the diagonal.
        labels = torch.arange(z1.size(0), device=z1.device)

        loss = (
            functional.cross_entropy(logits, labels)
            + functional.cross_entropy(logits.T, labels)
        ) / 2

        return loss
