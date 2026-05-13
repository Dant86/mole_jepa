"""Factory for building a MoLeJEPA model and its loss from a ModelConfig."""

from torch import nn

from mole_jepa import config as config_module
from mole_jepa import losses, models, regularizers, test_statistics


def build(config: config_module.ModelConfig) -> tuple[models.MoLeJEPA, nn.Module]:
    """Instantiate a :class:`~mole_jepa.models.MoLeJEPA` and its loss module.

    Args:
        config: A :class:`~mole_jepa.config.ModelConfig` describing the
            desired architecture.

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

    if config.freeze_image_encoder:
        # Freeze the ViT backbone; the projection head remains trainable so
        # the model can still learn to map frozen features into the shared
        # embedding space.
        model.image_encoder.vit.requires_grad_(False)

    loss: nn.Module
    if config.contrastive:
        loss = losses.InfoNCELoss(temperature=config.info_nce_temperature)
    else:
        loss = losses.JEPALoss(
            regularizer=regularizers.SIGReg(
                test_statistic=test_statistics.epps_pulley(config.sigreg_dist),
                n_directions=config.sigreg_n_directions,
                demean=config.sigreg_demean,
            ),
            lam=config.jepa_lam,
            regularize_z_t=config.jepa_regularize_z_t,
        )

    return model, loss
