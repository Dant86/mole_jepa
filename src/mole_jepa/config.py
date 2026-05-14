"""Plain-data configuration for MoLeJEPA models and data loading."""

import dataclasses
import hashlib
from typing import Literal


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a MoLeJEPA model and its loss function.

    All encoder and predictor components share ``embed_dim``, enforcing a
    consistent embedding space. Loss type is selected via ``contrastive``;
    the SIGReg and InfoNCE fields are only active for their respective modes.

    Attributes:
        embed_dim: Shared output dimension for both encoders and the predictor.
        image_encoder_model_name: HuggingFace identifier for the ViT.
        text_encoder_model_name: HuggingFace identifier for the language model.
        predictor_hidden_dim: Hidden width of the MLP predictor.
        predictor_n_layers: Total number of linear layers in the predictor.
        contrastive: If ``True``, use :class:`~mole_jepa.losses.InfoNCELoss`;
            otherwise use :class:`~mole_jepa.losses.JEPALoss` with SIGReg.
        sigreg_dist: Target distribution for the Epps-Pulley statistic.
        sigreg_n_directions: Number of random projection directions in SIGReg.
        sigreg_demean: If ``True``, subtract the batch mean before applying
            SIGReg, removing the zero-mean constraint so each modality can
            maintain its own offset in the embedding space.
        jepa_lam: Regularization weight λ for JEPALoss. The MSE weight is
            ``1 - jepa_lam``. Default 0.05 per the LeJEPA paper.
        jepa_regularize_z_i: If ``True`` (default), apply SIGReg to the image
            embeddings ``z_v``. Set to ``False`` when the image encoder is
            frozen — MSE already prevents projection collapse, so regularizing
            the image side creates a competing objective with no benefit.
        jepa_regularize_z_t: If ``True`` (default), apply SIGReg to the text
            embeddings ``z_t``. If ``False``, the text encoder acts as an
            unregularized target.
        info_nce_temperature: Softmax temperature for InfoNCELoss.
        freeze_image_encoder: If ``True`` (default), freeze the ViT backbone
            weights so only the projection head remains trainable. The
            projection head is always trained regardless of this flag.
            Motivated by VL-JEPA, which treats the vision encoder as a fixed
            semantic feature extractor.
    """

    embed_dim: int = 256
    image_encoder_model_name: str = "google/vit-base-patch16-224"
    text_encoder_model_name: str = "bert-base-uncased"
    predictor_hidden_dim: int = 512
    predictor_n_layers: int = 2
    contrastive: bool = False
    sigreg_dist: Literal["gaussian", "laplace"] = "gaussian"
    sigreg_n_directions: int = 128
    sigreg_demean: bool = False
    jepa_lam: float = 0.05
    jepa_regularize_z_i: bool = True
    jepa_regularize_z_t: bool = True
    info_nce_temperature: float = 0.07
    freeze_image_encoder: bool = True

    def serialize(self) -> str:
        """Serialize this config to a stable SHA-256 hex string.

        The hash is derived from a deterministic string representation of all
        fields. Useful for naming checkpoint directories.

        Returns:
            A 64-character hex string uniquely identifying this config.
        """
        payload = "".join(
            f"{k}={v}" for k, v in sorted(dataclasses.asdict(self).items())
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    The processor and tokenizer model names should match those in
    :class:`ModelConfig` to ensure consistent preprocessing.

    Attributes:
        image_processor_model_name: HuggingFace identifier for the image
            processor. Should match ``ModelConfig.image_encoder_model_name``.
        tokenizer_model_name: HuggingFace identifier for the tokenizer.
            Should match ``ModelConfig.text_encoder_model_name``.
        max_seq_length: Maximum token sequence length; longer captions are
            truncated.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        image_field: Key used to access the image in each dataset example.
            Defaults to ``"jpg"`` to match ``pixparse/cc3m-wds``.
        caption_field: Key used to access the caption string in each dataset
            example. Defaults to ``"txt"`` to match ``pixparse/cc3m-wds``.
    """

    image_processor_model_name: str = "google/vit-base-patch16-224"
    tokenizer_model_name: str = "bert-base-uncased"
    max_seq_length: int = 64
    batch_size: int = 256
    num_workers: int = 4
    image_field: str = "jpg"
    caption_field: str = "txt"
