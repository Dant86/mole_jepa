"""Top-level MoLeJEPA model."""

import dataclasses

import torch
from torch import nn
from torch.nn import functional

from mole_jepa.models import encoders
from mole_jepa.models import predictor as predictor_module


@dataclasses.dataclass
class MoLeJEPAOutput:
    """Named outputs from a MoLeJEPA forward pass.

    Attributes:
        loss: Total training loss: ``loss_mse + reg_weight * loss_reg``.
        loss_mse: MSE between predicted and actual text embeddings.
        loss_reg: SIGReg applied to both encoder outputs.
    """

    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_reg: torch.Tensor


class MoLeJEPA(nn.Module):
    """MoLeJEPA: multimodal JEPA with SIGReg regularization.

    Encodes images with `f` and text with `h`, predicts text embeddings
    from image embeddings via predictor `g`, and applies SIGReg to both
    encoder outputs to prevent representational collapse. All components
    are trained jointly.

    The total loss is:

        L = L_MSE(g(f(x)), h(y)) + reg_weight * (SIGReg(f(x)) + SIGReg(h(y)))

    Args:
        image_encoder: Encoder `f` mapping images to `z_v ∈ ℝ^d`.
        text_encoder: Encoder `h` mapping text to `z_t ∈ ℝ^d`.
        predictor: Module `g` mapping `z_v` to `ẑ_t`.
        regularizer: SIGReg instance applied to both `z_v` and `z_t`.
        reg_weight: Weighting factor λ scaling the regularization loss.
    """

    def __init__(
        self,
        image_encoder: encoders.ImageEncoder,
        text_encoder: encoders.TextEncoder,
        predictor: predictor_module.Predictor,
        regularizer: nn.Module,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.predictor = predictor
        self.regularizer = regularizer
        self.reg_weight = reg_weight

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> MoLeJEPAOutput:
        """Run a forward pass and compute training losses.

        Args:
            pixel_values: Float tensor of shape `(B, C, H, W)`.
            input_ids: Token ID tensor of shape `(B, T)`.
            attention_mask: Padding mask of shape `(B, T)`.

        Returns:
            MoLeJEPAOutput containing total loss and its components.
        """
        z_v = self.image_encoder(pixel_values)               # (B, d)
        z_t = self.text_encoder(input_ids, attention_mask)   # (B, d)
        z_hat_t = self.predictor(z_v)                        # (B, d)

        loss_mse = functional.mse_loss(z_hat_t, z_t)
        loss_reg = self.regularizer(z_v) + self.regularizer(z_t)

        return MoLeJEPAOutput(
            loss=loss_mse + self.reg_weight * loss_reg,
            loss_mse=loss_mse,
            loss_reg=loss_reg,
        )
