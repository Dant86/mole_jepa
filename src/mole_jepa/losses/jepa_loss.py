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
        loss: Total loss: ``loss_mse + reg_weight * loss_reg``.
        loss_mse: MSE between ``z_hat_t`` and ``z_t``.
        loss_reg: SIGReg summed over the image and text embeddings.
    """

    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_reg: torch.Tensor


class JEPALoss(nn.Module):
    """MSE prediction loss with SIGReg regularization.

    Computes:

        L = MSE(ẑ_t, z_t) + reg_weight * (SIGReg(z_v) + SIGReg(z_t))

    Args:
        regularizer: SIGReg instance applied independently to ``z_v`` and
            ``z_t``.
        reg_weight: Weighting factor λ scaling the regularization term.
    """

    def __init__(
        self,
        regularizer: nn.Module,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.regularizer = regularizer
        self.reg_weight = reg_weight

    def forward(self, output: models.MoLeJEPAOutput) -> JEPALossOutput:
        """Compute the JEPA loss for a batch of model outputs.

        Args:
            output: Output from a :class:`~mole_jepa.models.MoLeJEPA` forward
                pass, containing ``z_v``, ``z_hat_t``, and ``z_t``.

        Returns:
            JEPALossOutput with the total loss and its components.
        """
        loss_mse = functional.mse_loss(output.z_hat_t, output.z_t)
        loss_reg = self.regularizer(output.z_v) + self.regularizer(output.z_t)

        return JEPALossOutput(
            loss=loss_mse + self.reg_weight * loss_reg,
            loss_mse=loss_mse,
            loss_reg=loss_reg,
        )
