"""
リポジトリパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import database.repositories as repos_package


class TestRepositoriesInitExports:
    """repositories/__init__.pyのエクスポートテスト"""

    def test_base_repository_exported(self):
        """BaseRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "BaseRepository")

    def test_ohlcv_repository_exported(self):
        """OHLCVRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "OHLCVRepository")

    def test_funding_rate_repository_exported(self):
        """FundingRateRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "FundingRateRepository")

    def test_open_interest_repository_exported(self):
        """OpenInterestRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "OpenInterestRepository")

    def test_backtest_result_repository_exported(self):
        """BacktestResultRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "BacktestResultRepository")

    def test_ga_experiment_repository_exported(self):
        """GAExperimentRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "GAExperimentRepository")

    def test_generated_strategy_repository_exported(self):
        """GeneratedStrategyRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "GeneratedStrategyRepository")

    def test_long_short_ratio_repository_exported(self):
        """LongShortRatioRepositoryがエクスポートされている"""
        assert hasattr(repos_package, "LongShortRatioRepository")

    def test_all_contains_expected_repositories(self):
        """__all__に期待されるリポジトリが含まれる"""
        expected_repos = [
            "BaseRepository",
            "OHLCVRepository",
            "FundingRateRepository",
            "OpenInterestRepository",
            "BacktestResultRepository",
            "GAExperimentRepository",
            "GeneratedStrategyRepository",
            "LongShortRatioRepository",
        ]

        for repo in expected_repos:
            assert repo in repos_package.__all__, f"{repo} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(repos_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert repos_package.__doc__ is not None
        assert len(repos_package.__doc__) > 0
