"""CC3M streaming dataset."""

import collections.abc
import io
import urllib.error
import urllib.request

import datasets as hf_datasets
import torch
import torch.utils.data
from PIL import Image

from mole_jepa import config as config_module
from mole_jepa.data import transforms

_FETCH_TIMEOUT_S = 10


def _fetch_image(url: str) -> Image.Image:
    """Download an image from *url* and return it as an RGB PIL image.

    Args:
        url: HTTP/HTTPS URL pointing to an image.

    Returns:
        A PIL ``Image`` in RGB mode.

    Raises:
        urllib.error.URLError: If the request fails (e.g. dead link, timeout).
        OSError: If the response body cannot be decoded as an image.
    """
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


class CC3MDataset(torch.utils.data.IterableDataset):  # type: ignore[type-arg]
    """Streaming CC3M dataset for training.

    Wraps a HuggingFace ``IterableDataset``. Each example must expose the
    fields named by :attr:`~mole_jepa.config.DataConfig.image_field` and
    :attr:`~mole_jepa.config.DataConfig.caption_field`.

    The image field may contain either a PIL image *or* a URL string. When it
    is a URL the image is fetched over HTTP; failed fetches (dead links,
    timeouts, corrupt payloads) are printed and skipped — a substantial
    fraction of CC3M URLs are no longer reachable.

    For multi-worker DataLoaders, pass :func:`worker_init_fn` to shard the
    stream evenly across workers and avoid duplicate samples:

    .. code-block:: python

        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=CC3MDataset.worker_init_fn,
        )

    Args:
        hf_dataset: A HuggingFace streaming dataset.
        config: Data configuration supplying processor and tokenizer settings.
    """

    def __init__(
        self,
        hf_dataset: hf_datasets.IterableDataset,
        config: config_module.DataConfig,
    ) -> None:
        super().__init__()
        self._dataset = hf_dataset
        self._config = config

    def __iter__(
        self,
    ) -> collections.abc.Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over ``(pixel_values, input_ids, attention_mask)`` triples.

        Transforms are built here rather than in ``__init__`` so that the
        dataset remains picklable for multi-worker DataLoaders. Each worker
        process constructs its own transforms after the dataset is forked.

        When the image field contains a URL string it is fetched over HTTP.
        Examples whose image cannot be retrieved or decoded are skipped with a
        printed warning.

        Yields:
            A tuple of ``(pixel_values, input_ids, attention_mask)`` tensors
            for each valid example in the stream.
        """
        image_transform = transforms.build_image_transform(
            self._config.image_processor_model_name
        )
        tokenize = transforms.build_tokenizer(
            self._config.tokenizer_model_name, self._config.max_seq_length
        )
        image_field = self._config.image_field
        caption_field = self._config.caption_field
        for example in self._dataset:
            raw = example[image_field]
            if isinstance(raw, str):
                try:
                    image = _fetch_image(raw)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"[cc3m] skipping {raw!r}: {exc}")
                    continue
            else:
                image = raw.convert("RGB")
            pixel_values = image_transform(image)
            input_ids, attention_mask = tokenize(example[caption_field])
            yield pixel_values, input_ids, attention_mask

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:  # noqa: ARG004
        """Shard the stream evenly across DataLoader workers.

        Reads worker metadata from
        :func:`torch.utils.data.get_worker_info` and calls
        :meth:`datasets.IterableDataset.shard` so each worker processes a
        disjoint subset of the stream.

        Args:
            worker_id: Unused directly; worker identity is read from the
                PyTorch worker-info context.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        dataset: CC3MDataset = worker_info.dataset  # type: ignore[assignment]
        n_shards = dataset._dataset.n_shards
        num_shards = min(worker_info.num_workers, n_shards)
        if worker_info.id >= num_shards:
            # More workers than shards; the datasets library stops this worker
            # from yielding data, so nothing more is needed here.
            return
        dataset._dataset = dataset._dataset.shard(
            num_shards=num_shards,
            index=worker_info.id,
        )
