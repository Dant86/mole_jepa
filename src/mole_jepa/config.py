"""Model configuration and factory for MoLeJEPA."""

import dataclasses
import hashlib
from typing import Literal

from torch import nn

from mole_jepa import losses, models, regularizers, test_statistics


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
        jepa_lam: Reconstruction weight λ for JEPALoss. The regularization
            weight is ``1 - jepa_lam``. Default 0.05 per the LeJEPA paper.
        info_nce_temperature: Softmax temperature for InfoNCELoss.
    """

    embed_dim: int = 256
    image_encoder_model_name: str = "google/vit-base-patch16-224"
    text_encoder_model_name: str = "bert-base-uncased"
    predictor_hidden_dim: int = 512
    predictor_n_layers: int = 2
    contrastive: bool = False
    sigreg_dist: Literal["gaussian", "laplace"] = "gaussian"
    sigreg_n_directions: int = 128
    jepa_lam: float = 0.05
    info_nce_temperature: float = 0.07

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


def build(config: ModelConfig) -> tuple[models.MoLeJEPA, nn.Module]:
    """Instantiate a :class:`~mole_jepa.models.MoLeJEPA` and its loss module.

    Args:
        config: A :class:`ModelConfig` describing the desired architecture.

    Returns:
        A ``(model, loss)`` tuple ready for training.
    """
    model = models.MoLeJEPA(
        image_encoder=models.ImageEncoder(
            model_name=config.image_encoder_model_name,
            embed_dim=config.embed_dim,
        ),
        text_encoder=models.TextEncoder(
            model_name=config.text_encoder_model_name,
            embed_dim=config.embed_dim,
        ),
        predictor=models.Predictor(
            embed_dim=config.embed_dim,
            hidden_dim=config.predictor_hidden_dim,
            n_layers=config.predictor_n_layers,
        ),
    )

    loss: nn.Module
    if config.contrastive:
        loss = losses.InfoNCELoss(temperature=config.info_nce_temperature)
    else:
        loss = losses.JEPALoss(
            regularizer=regularizers.SIGReg(
                test_statistic=test_statistics.epps_pulley(config.sigreg_dist),
                n_directions=config.sigreg_n_directions,
            ),
            lam=config.jepa_lam,
        )

    return model, loss
