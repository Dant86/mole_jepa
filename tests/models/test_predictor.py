"""Unit tests for Predictor."""

import pytest
import torch

from mole_jepa.models import predictor as predictor_module

_EMBED_DIM = 16
_HIDDEN_DIM = 32


@pytest.fixture
def predictor() -> predictor_module.Predictor:
    return predictor_module.Predictor(embed_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM)


class TestPredictor:
    def test_output_shape(self, predictor: predictor_module.Predictor) -> None:
        x = torch.randn(8, _EMBED_DIM)
        assert predictor(x).shape == (8, _EMBED_DIM)

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_output_shape_various_batch_sizes(
        self, predictor: predictor_module.Predictor, batch_size: int
    ) -> None:
        x = torch.randn(batch_size, _EMBED_DIM)
        assert predictor(x).shape == (batch_size, _EMBED_DIM)

    def test_n_layers_2_no_intermediate(self) -> None:
        # n_layers=2 means one hidden layer with no loop body — exercises that
        # code path (the loop runs zero times).
        pred = predictor_module.Predictor(
            embed_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM, n_layers=2
        )
        assert pred(torch.randn(4, _EMBED_DIM)).shape == (4, _EMBED_DIM)

    def test_n_layers_4_intermediate_layers(self) -> None:
        # n_layers=4 adds two intermediate layers, exercising the loop body.
        pred = predictor_module.Predictor(
            embed_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM, n_layers=4
        )
        assert pred(torch.randn(4, _EMBED_DIM)).shape == (4, _EMBED_DIM)

    def test_n_layers_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n_layers"):
            predictor_module.Predictor(
                embed_dim=_EMBED_DIM, hidden_dim=_HIDDEN_DIM, n_layers=1
            )

    def test_gradient_flows(self, predictor: predictor_module.Predictor) -> None:
        x = torch.randn(4, _EMBED_DIM)
        predictor(x).sum().backward()
        for param in predictor.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
