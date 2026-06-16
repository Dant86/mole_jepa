"""Unit tests for SphericalUniformity."""

import math

import pytest
import torch

from mole_jepa import regularizers

_N_SAMPLES = 512
_DIM = 16
_SEED = 42


@pytest.fixture
def uniformity() -> regularizers.SphericalUniformity:
    return regularizers.SphericalUniformity()


@pytest.fixture
def random_embeddings() -> torch.Tensor:
    torch.manual_seed(_SEED)
    return torch.randn(_N_SAMPLES, _DIM)


class TestSphericalUniformity:
    def test_output_is_scalar(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        assert uniformity(random_embeddings).shape == torch.Size([])

    def test_output_is_nonpositive(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        # log E[exp(-2||z_i - z_j||^2)] <= 0 since the kernel is in (0, 1].
        assert uniformity(random_embeddings).item() <= 0.0

    def test_gradient_flows(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        x = random_embeddings.clone().requires_grad_(True)
        uniformity(x).backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_clustered_embeddings_give_loss_near_zero(
        self, uniformity: regularizers.SphericalUniformity
    ) -> None:
        # All points identical (after normalisation) -> kernel == 1 for every
        # pair -> mean == 1 -> log(1) == 0, the maximum (least negative) value.
        z = torch.ones(_N_SAMPLES, _DIM)
        assert uniformity(z).item() == pytest.approx(0.0, abs=1e-5)

    def test_spread_embeddings_give_more_negative_loss_than_clustered(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        clustered = torch.ones(_N_SAMPLES, _DIM)
        loss_clustered = uniformity(clustered).item()
        loss_spread = uniformity(random_embeddings).item()
        assert loss_spread < loss_clustered

    def test_scale_invariant(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        # Per-row rescaling shouldn't change the loss: F.normalize divides
        # out each row's own magnitude before the kernel is computed.
        scales = torch.linspace(0.1, 10.0, _N_SAMPLES).unsqueeze(-1)
        scaled = random_embeddings * scales
        torch.manual_seed(0)
        loss_original = uniformity(random_embeddings).item()
        loss_scaled = uniformity(scaled).item()
        assert loss_original == pytest.approx(loss_scaled, abs=1e-5)

    def test_permutation_invariant(
        self,
        uniformity: regularizers.SphericalUniformity,
        random_embeddings: torch.Tensor,
    ) -> None:
        perm = torch.randperm(_N_SAMPLES)
        loss_original = uniformity(random_embeddings).item()
        loss_permuted = uniformity(random_embeddings[perm]).item()
        assert loss_original == pytest.approx(loss_permuted, abs=1e-6)

    def test_matches_closed_form_for_orthogonal_pair(
        self, uniformity: regularizers.SphericalUniformity
    ) -> None:
        # Two orthogonal unit vectors: kernel(self,self) = 1, kernel(i,j) =
        # exp(-2 * 2) = exp(-4) for i != j.  Mean over the 2x2 Gram of pairs:
        # (1 + e^-4 + e^-4 + 1) / 4 = (1 + e^-4) / 2.
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        expected = math.log((1.0 + math.exp(-4.0)) / 2.0)
        assert uniformity(z).item() == pytest.approx(expected, abs=1e-6)
