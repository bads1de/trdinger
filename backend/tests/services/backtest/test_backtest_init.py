"""
backtestパッケージの__init__.pyのテスト

遅延ロード機能（__getattr__, __dir__）とエクスポート定義を確認します。
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

    def test_backtest_data_service_lazy_load(self):
        """BacktestDataServiceが遅延ロードされる"""
        from app.services.backtest.services.backtest_data_service import (
            BacktestDataService,
        )

        assert backtest_package.BacktestDataService is BacktestDataService

    def test_backtest_service_lazy_load(self):
        """BacktestServiceが遅延ロードされる"""
        from app.services.backtest.services.backtest_service import BacktestService

        assert backtest_package.BacktestService is BacktestService

    def test_lowercase_aliases_removed(self):
        """小文字のモジュールエイリアスは削除された"""
        assert not hasattr(backtest_package, "backtest_data_service")
        assert not hasattr(backtest_package, "backtest_service")
        assert not hasattr(backtest_package, "backtest_executor")
        assert not hasattr(backtest_package, "backtest_orchestrator")

    def test_getattr_raises_for_non_existent(self):
        """存在しない属性でAttributeErrorが発生する"""
        with pytest.raises(AttributeError, match="module.*has no attribute"):
            _ = backtest_package.NonExistentAttribute

    def test_dir_includes_exports(self):
        """__dir__にエクスポートが含まれる"""
        dir_result = dir(backtest_package)

        assert "BacktestDataService" in dir_result
        assert "BacktestService" in dir_result

    def test_dir_returns_list(self):
        """__dir__がリストを返す"""
        dir_result = dir(backtest_package)

        assert isinstance(dir_result, list)
        assert len(dir_result) > 0

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
