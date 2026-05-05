"""Unit tests for SIGReg."""

import pytest
import torch

from mole_jepa import regularizers, test_statistics

_N_SAMPLES = 2000
_DIM = 16
_SEED = 42


@pytest.fixture
def gaussian_reg() -> regularizers.SIGReg:
    return regularizers.SIGReg(test_statistics.epps_pulley("gaussian"))


@pytest.fixture
def gaussian_embeddings() -> torch.Tensor:
    torch.manual_seed(_SEED)
    return torch.randn(_N_SAMPLES, _DIM)


@pytest.fixture
def laplace_embeddings() -> torch.Tensor:
    torch.manual_seed(_SEED)
    return torch.distributions.Laplace(0.0, 1.0).sample((_N_SAMPLES, _DIM))


class TestSIGReg:
    def test_output_is_scalar(
        self, gaussian_reg: regularizers.SIGReg, gaussian_embeddings: torch.Tensor
    ) -> None:
        assert gaussian_reg(gaussian_embeddings).shape == torch.Size([])

    def test_output_is_nonnegative(
        self, gaussian_reg: regularizers.SIGReg, gaussian_embeddings: torch.Tensor
    ) -> None:
        assert gaussian_reg(gaussian_embeddings).item() >= 0.0

    def test_gradient_flows(
        self, gaussian_reg: regularizers.SIGReg, gaussian_embeddings: torch.Tensor
    ) -> None:
        x = gaussian_embeddings.requires_grad_(True)
        gaussian_reg(x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_small_loss_for_gaussian_embeddings(
        self, gaussian_reg: regularizers.SIGReg, gaussian_embeddings: torch.Tensor
    ) -> None:
        # Projections of N(0, I) onto any unit vector are N(0, 1), so the
        # Gaussian EP statistic should be small across all directions.
        torch.manual_seed(_SEED)
        assert gaussian_reg(gaussian_embeddings).item() < 0.05

    def test_larger_loss_for_mismatched_distribution(
        self,
        gaussian_reg: regularizers.SIGReg,
        gaussian_embeddings: torch.Tensor,
        laplace_embeddings: torch.Tensor,
    ) -> None:
        # Use low d so the CLT hasn't yet washed out the non-Gaussianity of
        # the projected marginals.
        torch.manual_seed(_SEED)
        loss_gaussian = gaussian_reg(gaussian_embeddings)
        torch.manual_seed(_SEED)
        loss_laplace = gaussian_reg(laplace_embeddings)
        assert loss_gaussian.item() < loss_laplace.item()

    def test_directions_resampled_each_forward_pass(
        self, gaussian_reg: regularizers.SIGReg, gaussian_embeddings: torch.Tensor
    ) -> None:
        loss_a = gaussian_reg(gaussian_embeddings)
        loss_b = gaussian_reg(gaussian_embeddings)
        assert loss_a.item() != loss_b.item()

    def test_n_directions_one(self, gaussian_embeddings: torch.Tensor) -> None:
        reg = regularizers.SIGReg(
            test_statistics.epps_pulley("gaussian"), n_directions=1
        )
        assert reg(gaussian_embeddings).shape == torch.Size([])

    def test_n_directions_large(self, gaussian_embeddings: torch.Tensor) -> None:
        reg = regularizers.SIGReg(
            test_statistics.epps_pulley("gaussian"), n_directions=512
        )
        assert reg(gaussian_embeddings).shape == torch.Size([])
