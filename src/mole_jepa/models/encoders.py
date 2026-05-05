"""Image and text encoders for MoLeJEPA."""

import torch
import transformers
from torch import nn


class ImageEncoder(nn.Module):
    """ViT-based image encoder with a learned projection head.

    Extracts the CLS token from the final hidden state and projects it to
    the shared embedding space.

    Args:
        model_name: HuggingFace model identifier for the ViT.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.vit = transformers.AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.vit.config.hidden_size, embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            pixel_values: Float tensor of shape `(B, C, H, W)`.

        Returns:
            Embedding tensor of shape `(B, embed_dim)`.
        """
        cls = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0]
        return self.projection(cls)


class TextEncoder(nn.Module):
    """Language model-based text encoder with a learned projection head.

    Mean-pools over non-padding token embeddings before projecting to the
    shared embedding space.

    Args:
        model_name: HuggingFace model identifier for the language model.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.lm = transformers.AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.lm.config.hidden_size, embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of token sequences.

        Args:
            input_ids: Integer tensor of shape `(B, T)`.
            attention_mask: Binary mask of shape `(B, T)`, 1 for real tokens.

        Returns:
            Embedding tensor of shape `(B, embed_dim)`.
        """
        hidden = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, T, H)
        pooled = _mean_pool(hidden, attention_mask)  # (B, H)
        return self.projection(pooled)


def _mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool token embeddings weighted by the attention mask.

    Args:
        hidden_states: Token embeddings of shape `(B, T, H)`.
        attention_mask: Binary mask of shape `(B, T)`.

    Returns:
        Pooled tensor of shape `(B, H)`.
    """
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
