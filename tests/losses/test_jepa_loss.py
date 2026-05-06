"""Unit tests for JEPALoss."""

import pytest
import torch

from mole_jepa import losses, regularizers, test_statistics
from mole_jepa.models import mole_jepa as mole_jepa_module

_N = 64
_D = 16
_SEED = 42


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
    return losses.JEPALoss(regularizer=reg, reg_weight=1.0)


class TestJEPALoss:
    def test_output_type(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        assert isinstance(result, losses.JEPALossOutput)

    def test_all_terms_are_scalar(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        assert result.loss.shape == torch.Size([])
        assert result.loss_mse.shape == torch.Size([])
        assert result.loss_reg.shape == torch.Size([])

    def test_perfect_prediction_zeroes_mse(self, loss_fn: losses.JEPALoss) -> None:
        z_t = torch.randn(_N, _D)
        result = loss_fn(_make_output(z_hat_t=z_t.clone(), z_t=z_t))
        assert result.loss_mse.item() == pytest.approx(0.0, abs=1e-6)

    def test_reg_weight_zero_loss_equals_mse(self) -> None:
        reg = regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))
        loss_fn = losses.JEPALoss(regularizer=reg, reg_weight=0.0)
        result = loss_fn(_make_output())
        assert result.loss.item() == pytest.approx(result.loss_mse.item())

    def test_loss_decomposition(self, loss_fn: losses.JEPALoss) -> None:
        result = loss_fn(_make_output())
        expected = result.loss_mse.item() + 1.0 * result.loss_reg.item()
        assert result.loss.item() == pytest.approx(expected, rel=1e-5)

    def test_gradient_flows(self, loss_fn: losses.JEPALoss) -> None:
        z_v = torch.randn(_N, _D, requires_grad=True)
        z_hat_t = torch.randn(_N, _D, requires_grad=True)
        z_t = torch.randn(_N, _D, requires_grad=True)
        output = mole_jepa_module.MoLeJEPAOutput(
            z_v=z_v, z_hat_t=z_hat_t, z_t=z_t
        )
        loss_fn(output).loss.backward()
        assert z_v.grad is not None
        assert z_hat_t.grad is not None
        assert z_t.grad is not None
