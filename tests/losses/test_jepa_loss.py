"""Unit tests for JEPALoss."""

import pytest
import torch

from mole_jepa import losses, regularizers, test_statistics
from mole_jepa.models import mole_jepa as mole_jepa_module

_N = 64
_D = 16
_SEED = 42
_LAM_IMAGE = 0.05  # matches the JEPALoss default
_LAM_TEXT = 0.05  # matches the JEPALoss default


def _make_output(
    z_v: torch.Tensor | None = None,
    z_hat_t: torch.Tensor | None = None,
    z_t: torch.Tensor | None = None,
) -> mole_jepa_module.MoLeJEPAOutput:
    torch.manual_seed(_SEED)
    z_v = z_v if z_v is not None else torch.randn(_N, _D)
    z_t = z_t if z_t is not None else torch.randn(_N, _D)
    z_hat_t = z_hat_t if z_hat_t is not None else torch.randn(_N, _D)
    return mole_jepa_module.MoLeJEPAOutput(z_v=z_v, z_hat_t=z_hat_t, z_t=z_t)


@pytest.fixture
def loss_fn() -> losses.JEPALoss:
    reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
    return losses.JEPALoss(regularizer=reg, lam_image=_LAM_IMAGE, lam_text=_LAM_TEXT)


class TestJEPALoss:
    def test_output_type(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        assert isinstance(result, losses.JEPALossOutput)

    def test_all_terms_are_scalar(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        assert result.loss.shape == torch.Size([])
        assert result.loss_mse.shape == torch.Size([])
        assert result.loss_reg_image.shape == torch.Size([])
        assert result.loss_reg_text.shape == torch.Size([])

    def test_perfect_prediction_zeroes_mse(self, loss_fn: losses.JEPALoss) -> None:
        z_t = torch.randn(_N, _D)
        result = loss_fn(_make_output(z_hat_t=z_t.clone(), z_t=z_t))
        assert result.loss_mse.item() == pytest.approx(0.0, abs=1e-6)

    def test_lam_zero_loss_equals_mse(self) -> None:
        """When lam_image=lam_text=0 the regularization terms vanish and loss == MSE."""
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        loss_fn = losses.JEPALoss(regularizer=reg, lam_image=0.0, lam_text=0.0)
        result = loss_fn(_make_output())
        assert result.loss.item() == pytest.approx(result.loss_mse.item())

    def test_loss_decomposition(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        expected = (
            _LAM_IMAGE * result.loss_reg_image.item()
            + _LAM_TEXT * result.loss_reg_text.item()
            + result.loss_mse.item()
        )
        assert result.loss.item() == pytest.approx(expected, rel=1e-5)

    def test_gradient_flows(self, loss_fn: losses.JEPALoss) -> None:
        z_v = torch.randn(_N, _D, requires_grad=True)
        z_hat_t = torch.randn(_N, _D, requires_grad=True)
        z_t = torch.randn(_N, _D, requires_grad=True)
        output = mole_jepa_module.MoLeJEPAOutput(z_v=z_v, z_hat_t=z_hat_t, z_t=z_t)
        loss_fn(output).loss.backward()
        assert z_v.grad is not None
        assert z_hat_t.grad is not None
        assert z_t.grad is not None

    def test_regularize_z_i_false_zeroes_image_reg(self) -> None:
        """regularize_z_i=False sets loss_reg_image to zero."""
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        loss_fn = losses.JEPALoss(
            regularizer=reg,
            lam_image=_LAM_IMAGE,
            lam_text=_LAM_TEXT,
            regularize_z_i=False,
        )
        result = loss_fn(_make_output())
        assert result.loss_reg_image.item() == pytest.approx(0.0)
        assert result.loss_reg_text.item() > 0.0

    def test_regularize_z_t_false_zeroes_text_reg(self) -> None:
        """regularize_z_t=False sets loss_reg_text to zero."""
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        loss_fn = losses.JEPALoss(
            regularizer=reg,
            lam_image=_LAM_IMAGE,
            lam_text=_LAM_TEXT,
            regularize_z_t=False,
        )
        result = loss_fn(_make_output())
        assert result.loss_reg_text.item() == pytest.approx(0.0)
        assert result.loss_reg_image.item() > 0.0

    def test_regularize_both_false_loss_equals_mse(self) -> None:
        """With both reg flags off, total loss == MSE."""
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        loss_fn = losses.JEPALoss(
            regularizer=reg,
            lam_image=_LAM_IMAGE,
            lam_text=_LAM_TEXT,
            regularize_z_i=False,
            regularize_z_t=False,
        )
        result = loss_fn(_make_output())
        assert result.loss.item() == pytest.approx(result.loss_mse.item(), rel=1e-5)


class TestCosineLoss:
    """Tests for cosine_loss=True (the 1 - cos(pred, target) prediction term)."""

    def _loss_fn(
        self, regularize_z_i: bool = True, regularize_z_t: bool = True
    ) -> losses.JEPALoss:
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        return losses.JEPALoss(
            regularizer=reg,
            lam_image=_LAM_IMAGE,
            lam_text=_LAM_TEXT,
            cosine_loss=True,
            regularize_z_i=regularize_z_i,
            regularize_z_t=regularize_z_t,
        )

    def test_parallel_vectors_give_zero_loss(self) -> None:
        z_t = torch.randn(_N, _D)
        result = self._loss_fn()(_make_output(z_hat_t=z_t.clone(), z_t=z_t))
        assert result.loss_mse.item() == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors_give_loss_one(self) -> None:
        z_t = torch.zeros(1, 2)
        z_t[0, 0] = 1.0
        z_hat_t = torch.zeros(1, 2)
        z_hat_t[0, 1] = 1.0
        output = mole_jepa_module.MoLeJEPAOutput(
            z_v=torch.randn(1, 2), z_hat_t=z_hat_t, z_t=z_t
        )
        result = self._loss_fn()(output)
        assert result.loss_mse.item() == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_give_loss_two(self) -> None:
        z_t = torch.zeros(1, 2)
        z_t[0, 0] = 1.0
        z_hat_t = torch.zeros(1, 2)
        z_hat_t[0, 0] = -1.0
        output = mole_jepa_module.MoLeJEPAOutput(
            z_v=torch.randn(1, 2), z_hat_t=z_hat_t, z_t=z_t
        )
        result = self._loss_fn()(output)
        assert result.loss_mse.item() == pytest.approx(2.0, abs=1e-6)

    def test_scale_invariant_unlike_mse(self) -> None:
        """Cosine loss ignores magnitude; MSE on the same inputs would not."""
        z_t = torch.randn(_N, _D)
        z_hat_t = z_t * 5.0  # same direction, different magnitude
        output = _make_output(z_hat_t=z_hat_t, z_t=z_t)
        result = self._loss_fn()(output)
        assert result.loss_mse.item() == pytest.approx(0.0, abs=1e-6)

    def test_applies_to_reverse_direction_too(self) -> None:
        z_v = torch.randn(_N, _D)
        output = mole_jepa_module.MoLeJEPAOutput(
            z_v=z_v,
            z_hat_t=torch.randn(_N, _D),
            z_t=torch.randn(_N, _D),
            z_hat_v=z_v.clone(),
        )
        result = self._loss_fn()(output)
        assert result.loss_mse_reverse.item() == pytest.approx(0.0, abs=1e-6)

    def test_differs_from_mse_for_same_inputs(self) -> None:
        output = _make_output()
        cosine_result = self._loss_fn()(output)
        mse_loss_fn = losses.JEPALoss(
            regularizer=regularizers.SIGReg(test_statistics.epps_pulley("gaussian")),
            lam_image=_LAM_IMAGE,
            lam_text=_LAM_TEXT,
            cosine_loss=False,
        )
        mse_result = mse_loss_fn(output)
        assert cosine_result.loss_mse.item() != pytest.approx(
            mse_result.loss_mse.item()
        )
