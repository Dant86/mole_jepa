"""Image and text preprocessing utilities."""

import collections.abc

import PIL.Image
import torch
import torchvision.transforms as T
import transformers
from torchvision.transforms.functional import (
    InterpolationMode,  # type: ignore[attr-defined]
)

from mole_jepa import config as config_module


def build_image_transform(
    model_name: str,
    train: bool = True,
    crop_scale: tuple[float, float] = (0.4, 1.0),
) -> collections.abc.Callable[[PIL.Image.Image], torch.Tensor]:
    """Build an image transform using torchvision, parameterised by the model.

    Reads resize/crop/normalisation settings from the HuggingFace
    ``AutoImageProcessor`` config once, then builds a pure-torchvision
    ``Compose`` pipeline.  This is substantially faster than calling the
    HuggingFace processor on every sample because torchvision runs its
    transforms in compiled C++ rather than Python/numpy.

    **Training vs. validation transforms**

    When ``train=True`` (default), a :class:`~torchvision.transforms.RandomResizedCrop`
    and :class:`~torchvision.transforms.RandomHorizontalFlip` are applied so
    that each epoch sees a different spatial view of every image.  This view
    diversity is essential for vision-language pretraining — without it the
    model effectively sees each image only once regardless of the number of
    epochs.

    ``crop_scale`` controls the trade-off between diversity and caption
    fidelity.  Very small scales (e.g. ``(0.08, 1.0)``) are standard for
    pure SSL but can crop away the captioned subject in image-text training.
    ``(0.4, 1.0)`` is a reasonable default: enough spatial variation to
    encourage invariance without frequently excluding the described content.

    When ``train=False``, a deterministic ``Resize → CenterCrop`` is used so
    validation metrics are reproducible.

    Args:
        model_name: HuggingFace model identifier for the image processor.
        train: If ``True``, apply random crop + flip augmentation.
        crop_scale: ``(min_scale, max_scale)`` for
            :class:`~torchvision.transforms.RandomResizedCrop`.  Only used
            when ``train=True``.

    Returns:
        A callable that maps a PIL image (RGB) to a ``(C, H, W)`` tensor.
    """
    processor = transformers.AutoImageProcessor.from_pretrained(model_name)

    # Resolve resize target — some configs use "shortest_edge", others "height"/"width".
    size_cfg = processor.size
    if "shortest_edge" in size_cfg:
        resize_to: int | tuple[int, int] = int(size_cfg["shortest_edge"])
    else:
        resize_to = (int(size_cfg["height"]), int(size_cfg["width"]))

    # Resolve crop size.
    crop_cfg = getattr(processor, "crop_size", None)
    if crop_cfg is not None:
        crop_to: tuple[int, int] = (int(crop_cfg["height"]), int(crop_cfg["width"]))
    elif isinstance(resize_to, tuple):
        crop_to = resize_to
    else:
        crop_to = (resize_to, resize_to)

    mean: list[float] = list(processor.image_mean)
    std: list[float] = list(processor.image_std)

    if train:
        spatial: list[object] = [
            T.RandomResizedCrop(
                crop_to,
                scale=crop_scale,
                ratio=(0.75, 1.3333),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
        ]
    else:
        spatial = [
            T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(crop_to),
        ]

    return T.Compose(
        [
            *spatial,
            T.ToTensor(),  # PIL [0,255] HWC → float [0,1] CHW
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_tokenizer(
    model_name: str,
    max_length: int = 64,
) -> collections.abc.Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    """Build a tokenizer using a HuggingFace AutoTokenizer.

    The returned callable accepts a string and returns a pair of
    ``(input_ids, attention_mask)`` tensors, each of shape ``(max_length,)``.

    Args:
        model_name: HuggingFace model identifier for the tokenizer.
        max_length: Maximum token sequence length; longer text is truncated.

    Returns:
        A callable mapping a string to ``(input_ids, attention_mask)``.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def tokenize(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    return tokenize


def preprocess(
    image: PIL.Image.Image,
    caption: str,
    config: config_module.DataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess a single image-caption pair into model-ready tensors.

    Convenience wrapper around :func:`build_image_transform` and
    :func:`build_tokenizer`. Intended for interactive use in notebooks where
    you want to run the model on one or a few samples without constructing a
    full :class:`~mole_jepa.data.CC3MDataset`.

    Each call rebuilds the processor and tokenizer from HuggingFace, so cache
    the result or use a :class:`~mole_jepa.data.CC3MDataset` for bulk work.

    Args:
        image: A PIL image (any mode; converted to RGB internally).
        caption: Caption string to tokenize.
        config: Data config supplying processor/tokenizer names and
            ``max_seq_length``.

    Returns:
        ``(pixel_values, input_ids, attention_mask)`` — each a 1-D or 3-D
        tensor with a leading batch dimension of 1, ready to pass directly
        to :class:`~mole_jepa.models.MoLeJEPA`.
    """
    image_transform = build_image_transform(config.image_processor_model_name)
    tokenize = build_tokenizer(config.tokenizer_model_name, config.max_seq_length)

    pixel_values = image_transform(image.convert("RGB")).unsqueeze(0)
    input_ids, attention_mask = tokenize(caption)
    return pixel_values, input_ids.unsqueeze(0), attention_mask.unsqueeze(0)
