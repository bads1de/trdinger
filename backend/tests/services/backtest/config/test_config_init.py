"""
backtest/configパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.backtest.config as config_package


class TestBacktestConfigInitExports:
    """backtest/config/__init__.pyのエクスポートテスト"""

    def test_backtest_config_exported(self):
        """BacktestConfigがエクスポートされている"""
        assert hasattr(config_package, "BacktestConfig")

    def test_backtest_run_config_exported(self):
        """BacktestRunConfigがエクスポートされている"""
        assert hasattr(config_package, "BacktestRunConfig")

    def test_backtest_run_config_validation_error_exported(self):
        """BacktestRunConfigValidationErrorがエクスポートされている"""
        assert hasattr(config_package, "BacktestRunConfigValidationError")

    def test_strategy_config_exported(self):
        """StrategyConfigがエクスポートされている"""
        assert hasattr(config_package, "StrategyConfig")

    def test_generated_ga_parameters_exported(self):
        """GeneratedGAParametersがエクスポートされている"""
        assert hasattr(config_package, "GeneratedGAParameters")

    def test_supported_strategies_exported(self):
        """SUPPORTED_STRATEGIESがエクスポートされている"""
        assert hasattr(config_package, "SUPPORTED_STRATEGIES")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "BacktestConfig",
            "BacktestRunConfig",
            "BacktestRunConfigValidationError",
            "GeneratedGAParameters",
            "StrategyConfig",
            "SUPPORTED_STRATEGIES",
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
