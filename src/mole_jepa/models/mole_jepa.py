"""Top-level MoLeJEPA model."""

import dataclasses

import torch
from torch import nn

from mole_jepa.models import encoders
from mole_jepa.models import predictor as predictor_module


@dataclasses.dataclass
class MoLeJEPAOutput:
    """Encoder and predictor outputs from a MoLeJEPA forward pass.

    Attributes:
        z_v: Image embeddings ``f(x)`` of shape ``(B, d)``.
        z_hat_t: Predicted text embeddings ``g(f(x))`` of shape ``(B, d)``.
        z_t: Text embeddings ``h(y)`` of shape ``(B, d)``.
    """

    z_v: torch.Tensor
    z_hat_t: torch.Tensor
    z_t: torch.Tensor


class MoLeJEPA(nn.Module):
    """MoLeJEPA: multimodal JEPA encoder–predictor.

    Encodes images with `f` and text with `h`, then predicts text embeddings
    from image embeddings via predictor `g`. Loss computation is intentionally
    decoupled; use a loss module such as
    :class:`~mole_jepa.losses.JEPALoss` or
    :class:`~mole_jepa.losses.InfoNCELoss` on the returned output.

    Args:
        image_encoder: Encoder `f` mapping images to `z_v ∈ ℝ^d`.
        text_encoder: Encoder `h` mapping text to `z_t ∈ ℝ^d`.
        predictor: Module `g` mapping `z_v` to `ẑ_t`.
    """

    def __init__(
        self,
        image_encoder: encoders.ImageEncoder,
        text_encoder: encoders.TextEncoder,
        predictor: predictor_module.Predictor,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.predictor = predictor

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> MoLeJEPAOutput:
        """Encode inputs and predict text embeddings.

        Args:
            pixel_values: Float tensor of shape `(B, C, H, W)`.
            input_ids: Token ID tensor of shape `(B, T)`.
            attention_mask: Padding mask of shape `(B, T)`.

        Returns:
            MoLeJEPAOutput containing ``z_v``, ``z_hat_t``, and ``z_t``.
        """
        z_v = self.image_encoder(pixel_values)  # (B, d)
        z_t = self.text_encoder(input_ids, attention_mask)  # (B, d)
        z_hat_t = self.predictor(z_v)  # (B, d)

        return MoLeJEPAOutput(z_v=z_v, z_hat_t=z_hat_t, z_t=z_t)
