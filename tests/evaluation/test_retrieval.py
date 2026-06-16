"""Unit tests for retrieval metrics."""

import pytest
import torch

from mole_jepa import evaluation

_N = 10
_D = 20
_CPI = 5  # captions per image


def _perfect_embeddings(
    n: int = _N, d: int = _D, cpi: int = _CPI
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return image and text embeddings with perfect ground-truth alignment.

    Each image embedding is a unit vector; each image's captions share that
    same unit vector. The similarity matrix has maximum values on the diagonal
    blocks, so any sensible retrieval system should achieve R@K = 1.0.
    """
    image_embs = torch.zeros(n, d)
    image_embs[torch.arange(n), torch.arange(n) % d] = 1.0
    text_embs = image_embs.repeat_interleave(cpi, dim=0)
    return image_embs, text_embs


class TestRetrievalMetrics:
    def test_returns_retrieval_result(self) -> None:
        image_embs, text_embs = _perfect_embeddings()
        result = evaluation.retrieval_metrics(image_embs, text_embs, _CPI)
        assert isinstance(result, evaluation.RetrievalResult)

    def test_perfect_alignment_gives_recall_one(self) -> None:
        image_embs, text_embs = _perfect_embeddings()
        result = evaluation.retrieval_metrics(image_embs, text_embs, _CPI)
        assert result.i2t_r1 == pytest.approx(1.0)
        assert result.i2t_r5 == pytest.approx(1.0)
        assert result.i2t_r10 == pytest.approx(1.0)
        assert result.t2i_r1 == pytest.approx(1.0)
        assert result.t2i_r5 == pytest.approx(1.0)
        assert result.t2i_r10 == pytest.approx(1.0)

    def test_all_scores_between_zero_and_one(self) -> None:
        torch.manual_seed(0)
        image_embs = torch.randn(_N, _D)
        text_embs = torch.randn(_N * _CPI, _D)
        result = evaluation.retrieval_metrics(image_embs, text_embs, _CPI)
        for score in (
            result.i2t_r1,
            result.i2t_r5,
            result.i2t_r10,
            result.t2i_r1,
            result.t2i_r5,
            result.t2i_r10,
        ):
            assert 0.0 <= score <= 1.0

    def test_recall_monotone_in_k(self) -> None:
        # R@5 >= R@1 and R@10 >= R@5 for both directions.
        torch.manual_seed(0)
        image_embs = torch.randn(_N, _D)
        text_embs = torch.randn(_N * _CPI, _D)
        result = evaluation.retrieval_metrics(image_embs, text_embs, _CPI)
        assert result.i2t_r5 >= result.i2t_r1
        assert result.i2t_r10 >= result.i2t_r5
        assert result.t2i_r5 >= result.t2i_r1
        assert result.t2i_r10 >= result.t2i_r5

    def test_perfect_beats_random(self) -> None:
        torch.manual_seed(0)
        image_embs, text_embs = _perfect_embeddings(n=20)
        perfect = evaluation.retrieval_metrics(image_embs, text_embs, _CPI)

        random_text = torch.randn(20 * _CPI, _D)
        random = evaluation.retrieval_metrics(image_embs, random_text, _CPI)

        assert perfect.i2t_r1 > random.i2t_r1
        assert perfect.t2i_r1 > random.t2i_r1


class TestT2IDirect:
    """Tests for the reverse-predictor t2i path (t2i_image_embs/t2i_text_embs)."""

    def test_perfect_t2i_embeddings_give_recall_one(self) -> None:
        # i2t uses mismatched embeddings (so i2t alone would not be perfect),
        # but the dedicated t2i_image_embs/t2i_text_embs pair is perfectly
        # aligned -- t2i should be driven entirely by that pair.
        image_embs, text_embs = _perfect_embeddings()
        torch.manual_seed(1)
        mismatched_image_embs = torch.randn(_N, _D)

        result = evaluation.retrieval_metrics(
            mismatched_image_embs,
            text_embs,
            _CPI,
            t2i_image_embs=image_embs,
            t2i_text_embs=text_embs,
        )
        assert result.t2i_r1 == pytest.approx(1.0)
        assert result.t2i_r5 == pytest.approx(1.0)
        assert result.t2i_r10 == pytest.approx(1.0)

    def test_t2i_direct_overrides_transposed_fallback(self) -> None:
        # Default (no override) t2i falls back to transposing the i2t sim
        # matrix, which is poor when image_embs/text_embs are mismatched.
        # Supplying a well-aligned t2i pair should do strictly better.
        torch.manual_seed(2)
        mismatched_image_embs = torch.randn(_N, _D)
        mismatched_text_embs = torch.randn(_N * _CPI, _D)
        perfect_image_embs, perfect_text_embs = _perfect_embeddings()

        fallback = evaluation.retrieval_metrics(
            mismatched_image_embs, mismatched_text_embs, _CPI
        )
        direct = evaluation.retrieval_metrics(
            mismatched_image_embs,
            mismatched_text_embs,
            _CPI,
            t2i_image_embs=perfect_image_embs,
            t2i_text_embs=perfect_text_embs,
        )
        assert direct.t2i_r1 >= fallback.t2i_r1
        assert direct.t2i_r1 == pytest.approx(1.0)

    def test_only_one_of_the_pair_falls_back_to_transposed(self) -> None:
        # Passing only one of t2i_image_embs/t2i_text_embs should not engage
        # the direct path -- both must be provided.
        image_embs, text_embs = _perfect_embeddings()
        result = evaluation.retrieval_metrics(
            image_embs, text_embs, _CPI, t2i_image_embs=image_embs
        )
        assert result.t2i_r1 == pytest.approx(1.0)  # falls back, still perfect here
