"""Integration tests: full pipeline with tiny synthetic data.

These tests run without a GPU, without downloading any HuggingFace models,
and without a real dataset. They use lightweight fake backbones that return
random tensors in the correct shapes so that every other component -- the
projection heads, predictor, loss modules, and backward pass -- executes
against real tensors.

They specifically catch:
- Zero or NaN loss (the training loop bug that prompted their creation)
- Shape mismatches between components
- Gradient flow failures (no parameters updated)
"""

import types
import unittest.mock

import pytest
import torch

from mole_jepa import config as config_module
from mole_jepa.losses import InfoNCELoss

_HIDDEN = 64  # hidden size of the fake HF backbones
_EMBED = 32  # embed_dim for the test model
_B = 4  # batch size
_T = 8  # sequence length


# ── fake HF backbones ─────────────────────────────────────────────────────────


class _FakeViT(torch.nn.Module):
    """Minimal ViT stand-in: accepts pixel_values, returns random last_hidden_state."""

    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=_HIDDEN)

    def forward(self, pixel_values: torch.Tensor, **_: object) -> object:
        B = pixel_values.shape[0]
        return types.SimpleNamespace(last_hidden_state=torch.randn(B, 197, _HIDDEN))


class _FakeBERT(torch.nn.Module):
    """Minimal BERT stand-in: accepts input_ids + attention_mask, returns random last_hidden_state."""  # noqa: E501

    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=_HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> object:
        B, T = input_ids.shape
        return types.SimpleNamespace(last_hidden_state=torch.randn(B, T, _HIDDEN))


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(params=["jepa", "contrastive"])
def model_and_loss(
    request: pytest.FixtureRequest,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Tiny MoLeJEPA + loss built with mocked HF backbones.

    Parametrised over both loss modes so every test runs for JEPA and InfoNCE.
    """
    cfg = config_module.ModelConfig(
        embed_dim=_EMBED,
        predictor_hidden_dim=64,
        predictor_n_layers=2,
        contrastive=(request.param == "contrastive"),
    )
    with unittest.mock.patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=[_FakeViT(), _FakeBERT()],
    ):
        return config_module.build(cfg)


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(_B, 3, 224, 224),
        torch.randint(0, 1000, (_B, _T)),
        torch.ones(_B, _T, dtype=torch.long),
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _forward_loss(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    pixel_values, input_ids, attention_mask = batch
    output = model(pixel_values, input_ids, attention_mask)
    if isinstance(loss_fn, InfoNCELoss):
        return loss_fn(output.z_v, output.z_t)
    return loss_fn(output).loss


# ── tests ─────────────────────────────────────────────────────────────────────


class TestPipelineSmoke:
    def test_loss_is_finite_and_nonzero(
        self,
        model_and_loss: tuple[torch.nn.Module, torch.nn.Module],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        model, loss_fn = model_and_loss
        loss = _forward_loss(model, loss_fn, batch)
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        assert loss.item() != 0.0, "loss is exactly zero"

    def test_backward_populates_gradients(
        self,
        model_and_loss: tuple[torch.nn.Module, torch.nn.Module],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        model, loss_fn = model_and_loss
        loss = _forward_loss(model, loss_fn, batch)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients were populated after backward()"

    def test_optimizer_step_updates_parameters(
        self,
        model_and_loss: tuple[torch.nn.Module, torch.nn.Module],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        model, loss_fn = model_and_loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        params_before = [p.clone().detach() for p in model.parameters()]

        optimizer.zero_grad()
        loss = _forward_loss(model, loss_fn, batch)
        loss.backward()
        optimizer.step()

        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, model.parameters())
        ), "no parameters were updated after optimizer step"
