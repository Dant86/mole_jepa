"""Unit tests for ModelConfig and DataConfig."""

from mole_jepa import config as config_module


class TestSerialize:
    def test_deterministic(self) -> None:
        cfg = config_module.ModelConfig()
        assert cfg.serialize() == cfg.serialize()

    def test_differs_for_different_configs(self) -> None:
        assert config_module.ModelConfig(contrastive=False).serialize() != (
            config_module.ModelConfig(contrastive=True).serialize()
        )

    def test_returns_64_char_hex(self) -> None:
        result = config_module.ModelConfig().serialize()
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
