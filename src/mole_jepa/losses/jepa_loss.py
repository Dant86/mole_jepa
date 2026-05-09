"""JEPA prediction loss with SIGReg regularization."""

import dataclasses

import torch
from torch import nn
from torch.nn import functional

from mole_jepa import models


@dataclasses.dataclass
class JEPALossOutput:
    """Decomposed outputs from a JEPALoss forward pass.

    Attributes:
        loss: Total loss:
            ``lam * loss_mse + (1 - lam) * (loss_reg_image + loss_reg_text)``.
        loss_mse: MSE between ``z_hat_t`` and ``z_t``.
        loss_reg_image: SIGReg penalty on the image embeddings ``z_v``.
        loss_reg_text: SIGReg penalty on the text embeddings ``z_t``.
    """

    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_reg_image: torch.Tensor
    loss_reg_text: torch.Tensor


class JEPALoss(nn.Module):
    """MSE prediction loss with SIGReg regularization.

    Computes (following LeJEPA, eq. 4):

        L = λ · MSE(ẑ_t, z_t) + (1 - λ) · (SIGReg(z_v) + SIGReg(z_t))

    λ ∈ [0, 1] trades off reconstruction against regularization. The LeJEPA
    paper recommends λ = 0.05, heavily weighting SIGReg to prevent collapse.

    Args:
        regularizer: SIGReg instance applied independently to ``z_v`` and
            ``z_t``.
        lam: Reconstruction weight λ; ``1 - lam`` scales the regularization
            term. Default 0.05 per the LeJEPA recommendation.
    """

    def __init__(
        self,
        regularizer: nn.Module,
        lam: float = 0.05,
    ) -> None:
        super().__init__()
        self.regularizer = regularizer
        self.lam = lam

    def forward(self, output: models.MoLeJEPAOutput) -> JEPALossOutput:
        """Compute the JEPA loss for a batch of model outputs.

        Args:
            output: Output from a :class:`~mole_jepa.models.MoLeJEPA` forward
                pass, containing ``z_v``, ``z_hat_t``, and ``z_t``.

        Returns:
            JEPALossOutput with the total loss and its components.
        """
        loss_mse = functional.mse_loss(output.z_hat_t, output.z_t)
        loss_reg_image = self.regularizer(output.z_v)
        loss_reg_text = self.regularizer(output.z_t)

        loss_reg = loss_reg_image + loss_reg_text
        return JEPALossOutput(
            loss=self.lam * loss_mse + (1 - self.lam) * loss_reg,
            loss_mse=loss_mse,
            loss_reg_image=loss_reg_image,
            loss_reg_text=loss_reg_text,
        )
