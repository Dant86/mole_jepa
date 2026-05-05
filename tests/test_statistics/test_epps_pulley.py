"""Unit tests for the Epps-Pulley test_statistics."""

import math

import pytest
import torch

from mole_jepa import test_statistics

_N_SAMPLES = 5000
_SEED = 42


@pytest.fixture
def normal_samples() -> torch.Tensor:
    torch.manual_seed(_SEED)
    return torch.randn(_N_SAMPLES)


@pytest.fixture
def laplace_samples() -> torch.Tensor:
    torch.manual_seed(_SEED)
    return torch.distributions.Laplace(0.0, 1.0).sample((_N_SAMPLES,))


class TestEppsPulleyGaussian:
    def test_output_is_scalar(self, normal_samples: torch.Tensor) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        assert reg(normal_samples).shape == torch.Size([])

    def test_output_is_nonnegative(self, normal_samples: torch.Tensor) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        assert reg(normal_samples).item() >= 0.0

    def test_characteristic_fn_values(self) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        t = torch.tensor([0.0, 1.0, 2.0])
        expected = torch.tensor([math.exp(0.0), math.exp(-0.5), math.exp(-2.0)])
        torch.testing.assert_close(reg.characteristic_fn(t), expected)

    def test_statistic_small_for_normal_samples(
        self, normal_samples: torch.Tensor
    ) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        assert reg(normal_samples).item() < 0.05

    def test_discriminates_laplace(
        self, normal_samples: torch.Tensor, laplace_samples: torch.Tensor
    ) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        assert reg(normal_samples).item() < reg(laplace_samples).item()

    def test_gradient_flows(self, normal_samples: torch.Tensor) -> None:
        x = normal_samples.requires_grad_(True)
        reg = test_statistics.EppsPulleyGaussian()
        reg(x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_buffers_registered(self) -> None:
        reg = test_statistics.EppsPulleyGaussian()
        buffers = dict(reg.named_buffers())
        assert "t" in buffers
        assert "weights" in buffers


class TestEppsPulleyLaplace:
    def test_output_is_scalar(self, laplace_samples: torch.Tensor) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        assert reg(laplace_samples).shape == torch.Size([])

    def test_output_is_nonnegative(self, laplace_samples: torch.Tensor) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        assert reg(laplace_samples).item() >= 0.0

    def test_characteristic_fn_values(self) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        t = torch.tensor([0.0, 1.0, 2.0])
        expected = torch.tensor([1.0, 0.5, 1.0 / 5.0])
        torch.testing.assert_close(reg.characteristic_fn(t), expected)

    def test_statistic_small_for_laplace_samples(
        self, laplace_samples: torch.Tensor
    ) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        assert reg(laplace_samples).item() < 0.05

    def test_discriminates_normal(
        self, normal_samples: torch.Tensor, laplace_samples: torch.Tensor
    ) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        assert reg(laplace_samples).item() < reg(normal_samples).item()

    def test_gradient_flows(self, laplace_samples: torch.Tensor) -> None:
        x = laplace_samples.requires_grad_(True)
        reg = test_statistics.EppsPulleyLaplace()
        reg(x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_buffers_registered(self) -> None:
        reg = test_statistics.EppsPulleyLaplace()
        buffers = dict(reg.named_buffers())
        assert "t" in buffers
        assert "weights" in buffers


class TestEppsPulleyFactory:
    def test_gaussian_returns_correct_type(self) -> None:
        reg = test_statistics.epps_pulley("gaussian")
        assert isinstance(reg, test_statistics.EppsPulleyGaussian)

    def test_laplace_returns_correct_type(self) -> None:
        reg = test_statistics.epps_pulley("laplace")
        assert isinstance(reg, test_statistics.EppsPulleyLaplace)

    def test_unknown_distribution_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown distribution"):
            test_statistics.epps_pulley("cauchy")  # type: ignore[arg-type]

    def test_forwards_kwargs(self) -> None:
        reg = test_statistics.epps_pulley(
            "gaussian", t_max=5.0, n_integration_points=33
        )
        assert reg.t[-1].item() == pytest.approx(5.0)
        assert reg.t.shape == torch.Size([33])
