"""MLP predictor for MoLeJEPA."""

import torch
from torch import nn


class Predictor(nn.Module):
    """MLP that predicts text embeddings from image embeddings.

    Each intermediate layer is followed by GELU activation and LayerNorm.
    The final layer is a plain linear projection with no activation.

    Args:
        embed_dim: Input and output dimension.
        hidden_dim: Hidden layer width.
        n_layers: Total number of linear layers. Must be at least 2.

    Raises:
        ValueError: If `n_layers` is less than 2.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")

        layers: list[nn.Module] = [
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        ]
        for _ in range(n_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ]
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """Predict text embeddings from image embeddings.

        Args:
            z_v: Image embeddings of shape `(B, embed_dim)`.

        Returns:
            Predicted text embeddings of shape `(B, embed_dim)`.
        """
        return self.net(z_v)
