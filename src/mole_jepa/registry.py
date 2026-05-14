"""Named model configurations for MoLeJEPA experiments.

Add a new entry to :data:`CONFIGS` to register a configuration. The key
becomes the value passed to ``--config`` at training time and can also be
used to look up a config in a notebook::

    from mole_jepa import registry, model_io

    cfg = registry.CONFIGS["vit_base_bert_contrastive"]
    model = model_io.load_model(cfg, checkpoint_dir)

Naming convention: ``{image_encoder}_{text_encoder}_{loss_type}``.
"""

from mole_jepa import config as config_module

CONFIGS: dict[str, config_module.ModelConfig] = {
    # ── ViT-base + BERT-base, unfrozen image encoder ──────────────────────────
    "vit_base_bert_contrastive": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=True,
        freeze_image_encoder=False,
    ),
    "vit_base_bert_jepa": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=False,
        freeze_image_encoder=False,
    ),
    "vit_base_bert_jepa_diff_mean": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=False,
        sigreg_demean=True,
        freeze_image_encoder=False,
    ),
    # ── ViT-tiny + BERT-tiny, frozen image encoder ───────────────────────────
    "vit_tiny_bert_tiny_contrastive_frozen": config_module.ModelConfig(
        embed_dim=128,
        image_encoder_model_name="WinKawaks/vit-tiny-patch16-224",
        text_encoder_model_name="google/bert_uncased_L-2_H-128_A-2",
        predictor_hidden_dim=256,
        predictor_n_layers=2,
        contrastive=True,
        freeze_image_encoder=True,
    ),
    "vit_tiny_bert_tiny_jepa_frozen": config_module.ModelConfig(
        embed_dim=128,
        image_encoder_model_name="WinKawaks/vit-tiny-patch16-224",
        text_encoder_model_name="google/bert_uncased_L-2_H-128_A-2",
        predictor_hidden_dim=256,
        predictor_n_layers=2,
        contrastive=False,
        jepa_regularize_z_i=False,
        freeze_image_encoder=True,
    ),
    "vit_tiny_bert_tiny_jepa_diff_mean_frozen": config_module.ModelConfig(
        embed_dim=128,
        image_encoder_model_name="WinKawaks/vit-tiny-patch16-224",
        text_encoder_model_name="google/bert_uncased_L-2_H-128_A-2",
        predictor_hidden_dim=256,
        predictor_n_layers=2,
        contrastive=False,
        sigreg_demean=True,
        jepa_regularize_z_i=False,
        freeze_image_encoder=True,
    ),
    # ── ViT-base + BERT-base, frozen image encoder ────────────────────────────
    "vit_base_bert_contrastive_frozen": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=True,
        freeze_image_encoder=True,
    ),
    "vit_base_bert_jepa_frozen": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=False,
        jepa_regularize_z_i=False,
        freeze_image_encoder=True,
    ),
    "vit_base_bert_jepa_diff_mean_frozen": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=False,
        sigreg_demean=True,
        jepa_regularize_z_i=False,
        freeze_image_encoder=True,
    ),
}
