"""COCO dataset for image-text retrieval evaluation."""

import collections.abc
from typing import Any

import torch
import torch.utils.data

from mole_jepa import config as config_module
from mole_jepa.data import transforms

_CAPTIONS_PER_IMAGE = 5


class COCODataset:
    """Dataset wrapper for image-text retrieval benchmarking.

    Supports two row layouts:

    **Grouped** (``flat=False``, default)
        One row per image; the caption column contains a list of strings.
        Compatible with HuggingFace COCO-style datasets.

    **Flat** (``flat=True``)
        One row per ``(image, caption)`` pair; the caption column is a plain
        string.  Consecutive ``captions_per_image`` rows are assumed to share
        the same image (the ordering produced by img2dataset / WebDataset).
        Compatible with ``clip-benchmark/wds_flickr30k`` and similar repos.

    In both modes captions are stored in image-grouped order so that the
    captions for image *i* occupy positions
    ``[i * captions_per_image, (i + 1) * captions_per_image)``, as required
    by :func:`~mole_jepa.evaluation.retrieval.retrieval_metrics`.

    Use :meth:`image_loader` and :meth:`caption_loader` to obtain DataLoaders
    for the two encoder passes during evaluation.

    Args:
        hf_dataset: Iterable of dataset rows (e.g. a HuggingFace ``Dataset``).
        config: Data configuration supplying processor and tokenizer settings.
        captions_per_image: Number of captions per image.  In grouped mode,
            only the first ``captions_per_image`` captions from each row are
            used.  In flat mode, exactly ``captions_per_image`` consecutive
            rows are consumed per image.
        caption_field: Name of the caption column (default: ``"captions"``).
        image_field: Name of the image column (default: ``"image"``).
            Use ``"jpg"`` for clip-benchmark WebDataset-derived repos.
        flat: If ``True``, treat each row as one ``(image, caption)`` pair
            and group ``captions_per_image`` consecutive rows per image.
            Defaults to ``False``.
    """

    def __init__(
        self,
        hf_dataset: collections.abc.Iterable[dict[str, Any]],
        config: config_module.DataConfig,
        captions_per_image: int = _CAPTIONS_PER_IMAGE,
        caption_field: str = "captions",
        image_field: str = "image",
        flat: bool = False,
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

        if flat:
            # Each row is one (image, caption) string pair.  Consume
            # captions_per_image consecutive rows and emit one image tensor.
            buf: list[dict[str, Any]] = []
            for example in hf_dataset:
                buf.append(example)
                if len(buf) == captions_per_image:
                    pixel_values.append(image_transform(buf[0][image_field]))
                    for row in buf:
                        ids, mask = tokenize(str(row[caption_field]))
                        input_ids.append(ids)
                        attention_masks.append(mask)
                    buf = []
            # Partial trailing group — include it with however many captions
            # are available (retrieval_metrics tolerates uneven tails).
            if buf:
                pixel_values.append(image_transform(buf[0][image_field]))
                for row in buf:
                    ids, mask = tokenize(str(row[caption_field]))
                    input_ids.append(ids)
                    attention_masks.append(mask)
                # Pad caption count so shapes stay consistent.
                while len(input_ids) % captions_per_image != 0:
                    ids, mask = tokenize("")
                    input_ids.append(ids)
                    attention_masks.append(mask)
        else:
            for example in hf_dataset:
                pixel_values.append(image_transform(example[image_field]))
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
