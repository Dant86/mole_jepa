"""CC3M streaming dataset."""

import collections.abc

import datasets as hf_datasets
import torch
import torch.utils.data

from mole_jepa import config as config_module
from mole_jepa.data import transforms


class CC3MDataset(torch.utils.data.IterableDataset):  # type: ignore[type-arg]
    """Streaming CC3M dataset for training.

    Wraps a HuggingFace ``IterableDataset``. Each example must provide an
    ``"image"`` field (a PIL image) and a ``"caption"`` field (a string).
    Malformed examples are silently skipped with a debug log.

    For multi-worker DataLoaders, pass :func:`worker_init_fn` to shard the
    stream evenly across workers and avoid duplicate samples:

    .. code-block:: python

        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=CC3MDataset.worker_init_fn,
        )

    Args:
        hf_dataset: A HuggingFace streaming dataset with ``"image"`` and
            ``"caption"`` fields.
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
        for example in self._dataset:
            pixel_values = image_transform(example["image"])
            input_ids, attention_mask = tokenize(example["caption"])
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
