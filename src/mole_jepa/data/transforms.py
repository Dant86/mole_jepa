"""Image and text preprocessing utilities."""

import collections.abc

import PIL.Image
import torch
import transformers


def build_image_transform(
    model_name: str,
) -> collections.abc.Callable[[PIL.Image.Image], torch.Tensor]:
    """Build an image transform using a HuggingFace image processor.

    The returned callable accepts a PIL image and returns a ``(C, H, W)``
    float tensor normalised according to the processor's configuration.

    Args:
        model_name: HuggingFace model identifier for the image processor.

    Returns:
        A callable that maps a PIL image to a ``(C, H, W)`` tensor.
    """
    processor = transformers.AutoImageProcessor.from_pretrained(model_name)

    def transform(image: PIL.Image.Image) -> torch.Tensor:
        result: torch.Tensor = processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return result

    return transform


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
