"""Image-text retrieval metrics."""

import dataclasses

import torch
from torch.nn import functional


@dataclasses.dataclass
class RetrievalResult:
    """Recall@K scores for image-text retrieval in both directions.

    Attributes:
        i2t_r1: Image-to-text Recall@1.
        i2t_r5: Image-to-text Recall@5.
        i2t_r10: Image-to-text Recall@10.
        t2i_r1: Text-to-image Recall@1.
        t2i_r5: Text-to-image Recall@5.
        t2i_r10: Text-to-image Recall@10.
    """

    i2t_r1: float
    i2t_r5: float
    i2t_r10: float
    t2i_r1: float
    t2i_r5: float
    t2i_r10: float


def retrieval_metrics(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    captions_per_image: int = 5,
    *,
    t2i_image_embs: torch.Tensor | None = None,
    t2i_text_embs: torch.Tensor | None = None,
) -> RetrievalResult:
    """Compute R@1, R@5, and R@10 for image-to-text and text-to-image retrieval.

    Both embedding matrices are L2-normalised before similarity is computed.
    Captions must be in image-grouped order: captions for image *i* must
    occupy positions ``[i * captions_per_image, (i + 1) * captions_per_image)``.

    For JEPA models with a reverse predictor, pass ``t2i_image_embs`` (raw
    ``z_v``) and ``t2i_text_embs`` (``ẑ_v = g_φ(z_t)``) to use the reverse
    predictor for t2i retrieval.  When omitted, t2i falls back to transposing
    the i2t similarity matrix.

    Args:
        image_embs: Image embeddings of shape ``(N, d)``.
        text_embs: Caption embeddings of shape ``(N * captions_per_image, d)``.
        captions_per_image: Number of captions per image.
        t2i_image_embs: Optional raw ``z_v`` of shape ``(N, d)`` — t2i gallery.
        t2i_text_embs: Optional ``ẑ_v = g_φ(z_t)`` of shape
            ``(N * captions_per_image, d)`` — t2i queries.

    Returns:
        A :class:`RetrievalResult` containing all six Recall@K scores.
    """
    image_embs = functional.normalize(image_embs, dim=-1)
    text_embs = functional.normalize(text_embs, dim=-1)

    sim = image_embs @ text_embs.T  # (N, N * captions_per_image)
    i2t = _i2t_recalls(sim, captions_per_image)

    if t2i_image_embs is not None and t2i_text_embs is not None:
        t2i_img = functional.normalize(t2i_image_embs, dim=-1)
        t2i_txt = functional.normalize(t2i_text_embs, dim=-1)
        t2i_sim = t2i_txt @ t2i_img.T  # (N * captions_per_image, N)
        t2i = _t2i_recalls_direct(t2i_sim, captions_per_image)
    else:
        t2i = _t2i_recalls(sim, captions_per_image)

    return RetrievalResult(
        i2t_r1=i2t[0],
        i2t_r5=i2t[1],
        i2t_r10=i2t[2],
        t2i_r1=t2i[0],
        t2i_r5=t2i[1],
        t2i_r10=t2i[2],
    )


def _i2t_recalls(
    sim: torch.Tensor,
    captions_per_image: int,
) -> tuple[float, float, float]:
    """Compute image-to-text R@1, R@5, and R@10.

    Args:
        sim: Cosine similarity matrix of shape ``(N, N * captions_per_image)``.
        captions_per_image: Number of captions per image.

    Returns:
        A ``(R@1, R@5, R@10)`` tuple.
    """
    n = sim.shape[0]
    ranks = sim.argsort(dim=-1, descending=True)

    # Positive captions for image i occupy [i*cpi, (i+1)*cpi).
    pos_start = torch.arange(n, device=sim.device).unsqueeze(1) * captions_per_image

    recalls = []
    for k in (1, 5, 10):
        top_k = ranks[:, :k]
        in_range = (top_k >= pos_start) & (top_k < pos_start + captions_per_image)
        recalls.append(in_range.any(dim=-1).float().mean().item())

    return recalls[0], recalls[1], recalls[2]


def _t2i_recalls(
    sim: torch.Tensor,
    captions_per_image: int,
) -> tuple[float, float, float]:
    """Compute text-to-image R@1, R@5, and R@10.

    Args:
        sim: Cosine similarity matrix of shape ``(N, N * captions_per_image)``.
        captions_per_image: Number of captions per image.

    Returns:
        A ``(R@1, R@5, R@10)`` tuple.
    """
    n_caps = sim.shape[1]
    ranks = sim.T.argsort(dim=-1, descending=True)  # (N*cpi, N)

    # Positive image for caption j is j // cpi.
    pos_image = (
        torch.arange(n_caps, device=sim.device).unsqueeze(1) // captions_per_image
    )

    recalls = []
    for k in (1, 5, 10):
        top_k = ranks[:, :k]
        in_top_k = (top_k == pos_image).any(dim=-1).float().mean().item()
        recalls.append(in_top_k)

    return recalls[0], recalls[1], recalls[2]


def _t2i_recalls_direct(
    t2i_sim: torch.Tensor,
    captions_per_image: int,
) -> tuple[float, float, float]:
    """Compute t2i R@1/5/10 from a precomputed (N_caps, N_images) sim matrix.

    Used when the reverse predictor provides the t2i query embeddings directly,
    rather than transposing the i2t matrix.

    Args:
        t2i_sim: Similarity matrix of shape ``(N * captions_per_image, N)``
            where entry ``[j, i]`` is the similarity between text query j and
            image i.
        captions_per_image: Number of captions per image.

    Returns:
        A ``(R@1, R@5, R@10)`` tuple.
    """
    n_caps = t2i_sim.shape[0]
    ranks = t2i_sim.argsort(dim=-1, descending=True)  # (N*cpi, N)

    # Positive image for caption j is j // cpi.
    pos_image = (
        torch.arange(n_caps, device=t2i_sim.device).unsqueeze(1) // captions_per_image
    )

    recalls = []
    for k in (1, 5, 10):
        top_k = ranks[:, :k]
        in_top_k = (top_k == pos_image).any(dim=-1).float().mean().item()
        recalls.append(in_top_k)

    return recalls[0], recalls[1], recalls[2]
