"""
auto_strategy/configパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__）とエクスポート定義を確認します。
"""

import pytest

import app.services.auto_strategy.config as config_package


class TestAutoStrategyConfigInitExports:
    """auto_strategy/config/__init__.pyのエクスポートテスト"""

    def test_base_config_exported(self):
        """BaseConfigがエクスポートされている"""
        assert hasattr(config_package, "BaseConfig")

    def test_auto_strategy_config_exported(self):
        """AutoStrategyConfigがエクスポートされている"""
        assert hasattr(config_package, "AutoStrategyConfig")

    def test_mutation_config_exported(self):
        """MutationConfigがエクスポートされている"""
        assert hasattr(config_package, "MutationConfig")

    def test_evaluation_config_exported(self):
        """EvaluationConfigがエクスポートされている"""
        assert hasattr(config_package, "EvaluationConfig")

    def test_hybrid_config_exported(self):
        """HybridConfigがエクスポートされている"""
        assert hasattr(config_package, "HybridConfig")

    def test_tuning_config_exported(self):
        """TuningConfigがエクスポートされている"""
        assert hasattr(config_package, "TuningConfig")

    def test_two_stage_selection_config_exported(self):
        """TwoStageSelectionConfigがエクスポートされている"""
        assert hasattr(config_package, "TwoStageSelectionConfig")

    def test_robustness_config_exported(self):
        """RobustnessConfigがエクスポートされている"""
        assert hasattr(config_package, "RobustnessConfig")

    def test_early_termination_settings_exported(self):
        """EarlyTerminationSettingsがエクスポートされている"""
        assert hasattr(config_package, "EarlyTerminationSettings")

    def test_ga_config_lazy_load(self):
        """GAConfigが遅延ロードされる"""
        from app.services.auto_strategy.config.ga import GAConfig

        config = getattr(config_package, "GAConfig")

        assert config is GAConfig

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = config_package.NonExistentAttribute

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "BaseConfig",
            "AutoStrategyConfig",
            "EarlyTerminationSettings",
            "MutationConfig",
            "EvaluationConfig",
            "HybridConfig",
            "TuningConfig",
            "TwoStageSelectionConfig",
            "RobustnessConfig",
        ]

        for item in expected_items:
            assert item in config_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(config_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert config_package.__doc__ is not None
        assert len(config_package.__doc__) > 0
