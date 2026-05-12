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
    # ── ViT-base + BERT-base ──────────────────────────────────────────────────
    "vit_base_bert_contrastive": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=True,
    ),
    "vit_base_bert_jepa": config_module.ModelConfig(
        embed_dim=256,
        image_encoder_model_name="google/vit-base-patch16-224",
        text_encoder_model_name="bert-base-uncased",
        predictor_hidden_dim=512,
        predictor_n_layers=2,
        contrastive=False,
    ),
}
