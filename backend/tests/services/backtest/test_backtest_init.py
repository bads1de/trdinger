"""
backtestパッケージの__init__.pyのテスト

モジュールエイリアスとエクスポート定義を確認します。
"""

import pytest

import app.services.backtest as backtest_package


class TestBacktestInitExports:
    """backtest/__init__.pyのエクスポートテスト"""

    def test_backtest_config_exported(self):
        """BacktestConfigがエクスポートされている"""
        assert hasattr(backtest_package, "BacktestConfig")

    def test_backtest_run_config_exported(self):
        """BacktestRunConfigがエクスポートされている"""
        assert hasattr(backtest_package, "BacktestRunConfig")

    def test_backtest_run_config_validation_error_exported(self):
        """BacktestRunConfigValidationErrorがエクスポートされている"""
        assert hasattr(backtest_package, "BacktestRunConfigValidationError")

    def test_strategy_config_exported(self):
        """StrategyConfigがエクスポートされている"""
        assert hasattr(backtest_package, "StrategyConfig")

    def test_supported_strategies_exported(self):
        """SUPPORTED_STRATEGIESがエクスポートされている"""
        assert hasattr(backtest_package, "SUPPORTED_STRATEGIES")

    def test_backtest_data_service_exported(self):
        """BacktestDataServiceがエクスポートされている"""
        assert hasattr(backtest_package, "BacktestDataService")

    def test_backtest_service_exported(self):
        """BacktestServiceがエクスポートされている"""
        assert hasattr(backtest_package, "BacktestService")

    def test_module_aliases_exist(self):
        """モジュールエイリアスが存在する"""
        assert hasattr(backtest_package, "backtest_data_service")
        assert hasattr(backtest_package, "backtest_service")
        assert hasattr(backtest_package, "backtest_executor")
        assert hasattr(backtest_package, "backtest_orchestrator")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "BacktestConfig",
            "BacktestRunConfig",
            "BacktestRunConfigValidationError",
            "StrategyConfig",
            "SUPPORTED_STRATEGIES",
            "BacktestDataService",
            "BacktestService",
        ]

        for item in expected_items:
            assert item in backtest_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(backtest_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert backtest_package.__doc__ is not None
        assert len(backtest_package.__doc__) > 0
