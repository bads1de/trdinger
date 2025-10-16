import pytest

from backend.app.services.auto_strategy.config.ga_config import GAConfig


class TestGAConfig:
    """GAConfigのテスト"""

    def test_default_regime_adaptation_enabled(self):
        """regime_adaptation_enabledのデフォルト値がFalseであることを確認"""
        config = GAConfig()
        assert config.regime_adaptation_enabled == False

    def test_regime_adaptation_enabled_set_true(self):
        """regime_adaptation_enabledをTrueに設定できることを確認"""
        config = GAConfig(regime_adaptation_enabled=True)
        assert config.regime_adaptation_enabled == True

    def test_serialize_deserialize(self):
        """シリアライズとデシリアライズが正しく動作することを確認"""
        original = GAConfig(regime_adaptation_enabled=True)
        data = original.model_dump()
        assert data["regime_adaptation_enabled"] == True

        restored = GAConfig.model_validate(data)
        assert restored.regime_adaptation_enabled == True
