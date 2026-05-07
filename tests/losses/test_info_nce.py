"""Unit tests for InfoNCELoss."""

import pytest
import torch
from torch.nn import functional

from mole_jepa import losses

_B = 16
_D = 32
_SEED = 0


@pytest.fixture
def loss_fn() -> losses.InfoNCELoss:
    return losses.InfoNCELoss(temperature=0.07)


class TestInfoNCELoss:
    def test_output_is_scalar(self, loss_fn: losses.InfoNCELoss) -> None:
        z1 = torch.randn(_B, _D)
        z2 = torch.randn(_B, _D)
        assert loss_fn(z1, z2).shape == torch.Size([])

    def test_output_is_nonnegative(self, loss_fn: losses.InfoNCELoss) -> None:
        torch.manual_seed(_SEED)
        z1 = torch.randn(_B, _D)
        z2 = torch.randn(_B, _D)
        assert loss_fn(z1, z2).item() >= 0.0

    def test_single_pair_is_zero(self, loss_fn: losses.InfoNCELoss) -> None:
        # With B=1 the softmax denominator has only one term, so CE = 0
        # regardless of the embeddings.
        z1 = torch.randn(1, _D)
        z2 = torch.randn(1, _D)
        assert loss_fn(z1, z2).item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self, loss_fn: losses.InfoNCELoss) -> None:
        torch.manual_seed(_SEED)
        z1 = torch.randn(_B, _D)
        z2 = torch.randn(_B, _D)
        assert loss_fn(z1, z2).item() == pytest.approx(loss_fn(z2, z1).item())

    def test_aligned_pairs_lower_than_random(self, loss_fn: losses.InfoNCELoss) -> None:
        # When z1 == z2 (perfectly aligned), each positive pair has maximum
        # similarity; loss should be lower than for random, uncorrelated pairs.
        torch.manual_seed(_SEED)
        z = functional.normalize(torch.randn(_B, _D), dim=-1)
        z_random = functional.normalize(torch.randn(_B, _D), dim=-1)
        loss_aligned = loss_fn(z, z.clone())
        loss_random = loss_fn(z, z_random)
        assert loss_aligned.item() < loss_random.item()

    def test_lower_temperature_sharpens_aligned_loss(self) -> None:
        # For perfectly aligned pairs, a lower temperature concentrates the
        # softmax on the correct entry, yielding a lower loss.
        torch.manual_seed(_SEED)
        z = functional.normalize(torch.randn(_B, _D), dim=-1)
        loss_sharp = losses.InfoNCELoss(temperature=0.07)(z, z.clone())
        loss_soft = losses.InfoNCELoss(temperature=1.0)(z, z.clone())
        assert loss_sharp.item() < loss_soft.item()

    def test_gradient_flows(self, loss_fn: losses.InfoNCELoss) -> None:
        z1 = torch.randn(_B, _D, requires_grad=True)
        z2 = torch.randn(_B, _D, requires_grad=True)
        loss_fn(z1, z2).backward()
        assert z1.grad is not None
        assert z2.grad is not None
        assert torch.isfinite(z1.grad).all()
        assert torch.isfinite(z2.grad).all()
