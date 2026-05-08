"""COCO dataset for image-text retrieval evaluation."""

import collections.abc
from typing import Any

import torch
import torch.utils.data

from mole_jepa import config as config_module
from mole_jepa.data import transforms

_CAPTIONS_PER_IMAGE = 5


class COCODataset:
    """COCO dataset for image-text retrieval benchmarking.

    Pre-processes all images and captions at construction time and stores
    them as :class:`torch.utils.data.TensorDataset` instances. Captions are
    kept in image-grouped order so that the captions for image *i* occupy
    positions ``[i * captions_per_image, (i + 1) * captions_per_image)``;
    this ordering is required by
    :func:`~mole_jepa.evaluation.retrieval.retrieval_metrics`.

    Use :meth:`image_loader` and :meth:`caption_loader` to obtain
    DataLoaders for the two encoder passes during evaluation.

    Args:
        hf_dataset: HuggingFace dataset with ``"image"`` (PIL image) and
            ``caption_field`` (list of strings) columns.
        config: Data configuration supplying processor and tokenizer settings.
        captions_per_image: Number of captions to retain per image.
        caption_field: Name of the captions column in ``hf_dataset``.
    """

    def __init__(
        self,
        hf_dataset: collections.abc.Iterable[dict[str, Any]],
        config: config_module.DataConfig,
        captions_per_image: int = _CAPTIONS_PER_IMAGE,
        caption_field: str = "captions",
    ) -> None:
        image_transform = transforms.build_image_transform(
            config.image_processor_model_name
        )
        tokenize = transforms.build_tokenizer(
            config.tokenizer_model_name, config.max_seq_length
        )
        self._captions_per_image = captions_per_image

        pixel_values: list[torch.Tensor] = []
        input_ids: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []

        for example in hf_dataset:
            pixel_values.append(image_transform(example["image"]))
            for caption in example[caption_field][:captions_per_image]:
                ids, mask = tokenize(caption)
                input_ids.append(ids)
                attention_masks.append(mask)

        self._images = torch.utils.data.TensorDataset(torch.stack(pixel_values))
        self._captions = torch.utils.data.TensorDataset(
            torch.stack(input_ids),
            torch.stack(attention_masks),
        )

    @property
    def n_images(self) -> int:
        """Number of unique images in the dataset."""
        return len(self._images)

    @property
    def captions_per_image(self) -> int:
        """Number of captions per image."""
        return self._captions_per_image

    def image_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader over the unique images.

        Args:
            batch_size: Number of images per batch.
            num_workers: Number of DataLoader worker processes.

        Returns:
            A DataLoader yielding ``(pixel_values,)`` tuples.
        """
        return torch.utils.data.DataLoader(
            self._images,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

    def caption_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader over all captions in image-grouped order.

        Args:
            batch_size: Number of captions per batch.
            num_workers: Number of DataLoader worker processes.

        Returns:
            A DataLoader yielding ``(input_ids, attention_mask)`` tuples.
        """
        return torch.utils.data.DataLoader(
            self._captions,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
