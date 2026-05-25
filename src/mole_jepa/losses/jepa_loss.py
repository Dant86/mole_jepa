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
            ``lam_image * loss_reg_image + lam_text * loss_reg_text + loss_mse``.
        loss_mse: MSE between ``z_hat_t`` and ``z_t``.
        loss_reg_image: SIGReg penalty on the image embeddings ``z_v``.
        loss_reg_text: SIGReg penalty on the text embeddings ``z_t``.
    """

    loss: torch.Tensor
    loss_mse: torch.Tensor
    loss_reg_image: torch.Tensor
    loss_reg_text: torch.Tensor


class JEPALoss(nn.Module):
    """MSE prediction loss with per-modality SIGReg regularization.

    Computes:

        L = λ_image · SIGReg(z_v) + λ_text · SIGReg(z_t) + MSE(ẑ_t, z_t)

    The MSE weight is implicitly 1; ``lam_image`` and ``lam_text`` independently
    scale the regularization pressure on each modality.  This decoupling allows
    pushing hard on text isotropy (high ``lam_text``) without inflating the
    image regularization or shrinking the prediction signal.

    Either regularization term can be disabled independently via
    ``regularize_z_i`` and ``regularize_z_t``. With a frozen image encoder,
    setting ``regularize_z_i=False`` is recommended: the image projection is
    well-behaved without explicit regularization, so adding it only creates a
    competing objective.

    Args:
        regularizer: SIGReg instance applied to encoder outputs.
        lam_image: Regularization weight for image embeddings ``z_v``.
            Default 0.05.
        lam_text: Regularization weight for text embeddings ``z_t``.
            Default 0.05. Increase (e.g. 0.3–0.5) if text embeddings exhibit
            directional collapse (high inter-sample cosine similarity).
        regularize_z_i: If ``True`` (default), apply SIGReg to the image
            embeddings ``z_v``. Set to ``False`` when the image encoder is
            frozen.
        regularize_z_t: If ``True`` (default), apply SIGReg to the text
            embeddings ``z_t``. If ``False``, ``z_t`` serves as an
            unregularized prediction target.
    """

    def __init__(
        self,
        regularizer: nn.Module,
        lam_image: float = 0.05,
        lam_text: float = 0.05,
        regularize_z_i: bool = True,
        regularize_z_t: bool = True,
    ) -> None:
        super().__init__()
        self.regularizer = regularizer
        self.lam_image = lam_image
        self.lam_text = lam_text
        self.regularize_z_i = regularize_z_i
        self.regularize_z_t = regularize_z_t

    def forward(self, output: models.MoLeJEPAOutput) -> JEPALossOutput:
        """Compute the JEPA loss for a batch of model outputs.

        Args:
            output: Output from a :class:`~mole_jepa.models.MoLeJEPA` forward
                pass, containing ``z_v``, ``z_hat_t``, and ``z_t``.

        Returns:
            JEPALossOutput with the total loss and its components.
            ``loss_reg_image`` is zero when ``regularize_z_i=False``;
            ``loss_reg_text`` is zero when ``regularize_z_t=False``.
        """
        loss_mse = functional.mse_loss(output.z_hat_t, output.z_t)
        zero = torch.zeros([], device=output.z_v.device, dtype=output.z_v.dtype)
        loss_reg_image = self.regularizer(output.z_v) if self.regularize_z_i else zero
        loss_reg_text = self.regularizer(output.z_t) if self.regularize_z_t else zero

        loss = (
            self.lam_image * loss_reg_image + self.lam_text * loss_reg_text + loss_mse
        )
        return JEPALossOutput(
            loss=loss,
            loss_mse=loss_mse,
            loss_reg_image=loss_reg_image,
            loss_reg_text=loss_reg_text,
        )
