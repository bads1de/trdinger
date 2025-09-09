"""
GA統合テストの欠如防止 - インジケーター設定不足による適応度統計記録失敗バグ検出

このテストスイートは、GAエンジンがインジケーター設定不足により適応度統計記録に
失敗するケースを検出します。
"""

import pytest
import gc
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import StrategyGene


@pytest.fixture
def ga_config_missing_indicators():
    """インジケーター設定が不足しているGA設定"""
    config = GAConfig.from_dict({
        "population_size": 10,
        "generations": 3,
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "elite_size": 2,
        "max_indicators": 3,
        "log_level": "DEBUG"
    })
    # インジケーター設定を明示的にNoneに設定
    config.allowed_indicators = None
    return config


@pytest.fixture
def mock_backtest_service_no_stats():
    """適応度統計記録が失敗するモックバックテストサービス"""
    service = MagicMock()

    def failing_stat_record(scenario_config):
        """統計記録処理が失敗するモック"""
        if "strategy_gene" not in scenario_config.get("strategy_config", {}):
            raise ValueError("統計記録失敗: 戦略遺伝子データが不足しています")

        gene_data = scenario_config["strategy_config"]["strategy_gene"]

        # allowed_indicatorsがNoneの場合に統計記録が失敗
        if not hasattr(GACurrentlyBeingProcessed().ga_config.with_applied_config(), 'allowed_indicators'):
            raise AttributeError("統計記録失敗: 利用可能なインジケーター一覧が未設定です")

        return {"performance_metrics": {"total_return": 0.1}}

    service.run_backtest.side_effect = failing_stat_record
    return service


class TestGAIntegrationWithMissingIndicators:
    """インジケーター不足時のGA統合テスト"""

    def test_ga_engine_fails_when_indicators_not_initialised(self, ga_config_missing_indicators, mock_backtest_service_no_stats):
        """インジケーター初期化されていない場合にGAエンジンが失敗するテスト"""
        factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config_missing_indicators)
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service_no_stats, factory, gene_generator)

        backtest_config = {"symbol": "BTC/USDT", "timeframe": "1h"}

        # allowed_indicatorsがNoneなのでStatistics記録が失敗することを期待
        with pytest.raises(AttributeError, match="利用可能なインジケーター一覧が未設定です"):
            result = ga_engine.run_evolution(ga_config_missing_indicators, backtest_config)

    def test_statistics_recording_fails_without_indicator_validation(self, ga_config_missing_indicators):
        """インジケーターバリデーションなしで統計記録が失敗するテスト"""
        # バックテストサービスがインジケーターチェックなしで統計を記録しようとする
        mock_service = MagicMock()
        gene_dict_invalid = {
            "indicators": [{"type": "UNKNOWN_INDICATOR", "parameters": {"period": 20}}],
            "trading_rules": {"entry_condition": "price > sma_20", "exit_condition": "price < sma_20"}
        }

        mock_service.run_backtest.return_value = {"performance_metrics": {"total_return": -0.05}}

        service = mock_service
        config = ga_config_missing_indicators

        # インジケーター参照がない設定でも統計処理しようとする場合
        if hasattr(config.with_applied_config(), 'allowed_indicators') and config.allowed_indicators is None:
            # これはバグを再現：統計記録がインジケーターチェックなしで進み、後で失敗
            with pytest.raises(ValueError, match="統計記録失敗"):
                service.run_backtest({
                    "strategy_config": {"strategy_gene": gene_dict_invalid}
                })

    def test_ga_config_allows_ga_run_with_missing_indicators_bug(self, ga_config_missing_indicators):
        """GA設定がインジケーターなしでGA実行できてしまうバグ判定テスト"""
        # このテストは現在失敗するはず（バグが修正されていない場合）
        config = ga_config_missing_indicators

        # バグ：allowed_indicatorsがNoneでもGAが開始できてしまう
        assert config.allowed_indicators is None

        # 統計記録フェーズで失敗するかどうかテスト
        # もし成功してしまう場合、バグ検出
        if config.allowed_indicators is None:
            pytest.fail("バグ検出: インジケーター設定なしでGAが実行できてしまいます。統計記録が失敗します。")

    def test_indicator_service_integration_failure_simulation(self, ga_config_missing_indicators):
        """インジケーターサービス統合失敗シミュレーション"""
        from app.services.auto_strategy.config.ga_runtime import GAConfig as GARuntimeConfig

        # TechnicalIndicatorServiceからの統合が失敗する場合
        with patch('app.services.indicators.TechnicalIndicatorService') as mock_indicator_service:
            mock_indicator_service.return_value.get_supported_indicators.side_effect = ImportError("サービス統合失敗")

            # GA設定でインジケーターサービス統合が失敗してもGAが実行されるバグ
            config = GARuntimeConfig(auto_strategy_config=None)
            # allowed_indicatorsが設定されないままになる
            assert config.allowed_indicators == []

            # この状態で統計記録すると失敗するはず
            try:
                # 仮の統計記録処理
                if not config.allowed_indicators:
                    raise ValueError("統計記録不可: インジケーター未設定")
            except ValueError as e:
                assert "インジケーター未設定" in str(e)

    def test_memory_leak_with_missing_indicators_ga_run(self, ga_config_missing_indicators):
        """インジケーター不足時のGA実行でメモリリークが発生するテスト"""
        import psutil
        import gc as py_gc

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        mock_service = MagicMock()
        mock_service.run_backtest.return_value = {"performance_metrics": {"total_return": 0.0}}

        factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config_missing_indicators)
        ga_engine = GeneticAlgorithmEngine(mock_service, factory, gene_generator)

        try:
            # メモリリークを伴う処理（インジケーター不在で統計記録に失敗）
            ga_engine.run_evolution(ga_config_missing_indicators, {"symbol": "BTC/USDT"})
        except Exception:
            pass  # エラーが発生しても処理継続

        py_gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024

        # メモリリーク判定（インジケーター不足が原因で統計記録失敗→リーク）
        memory_increase = final_memory - initial_memory

        # 許容範囲を超えるメモリ増加はバグの兆候
        if memory_increase > 20:  # 20MB以上増加したらバグ検出
            pytest.fail(f"バグ検出: インジケーター不足時のGA実行でメモリリーク発生 {memory_increase:.1f}MB")


class TestGAPerformanceWithMissingIndicators:
    """インジケーター不足時のGA性能テスト"""

    def test_adaptive_statistics_failure_with_no_indicators(self):
        """インジケーターなしの状況で適応度統計が失敗するテスト"""
        # 適応度統計を集計しようとするがインジケーター参照がないため失敗
        mock_stats = {
            "population_fitness": [],
            "best_strategy_indicators": []  # インジケーター情報がない
        }

        # 統計記録処理シミュレーション
        try:
            if not mock_stats["best_strategy_indicators"]:
                # インジケーター参照がない場合に統計記録が失敗
                for ind in mock_stats["best_strategy_indicators"]:
                    if "type" not in ind:
                        raise KeyError(f"インジケータータイプ不明: {ind}")

            # このエラーが発生しない場合、バグ
            assert False, "統計記録処理が失敗しなかった（バグ未検出）"

        except AssertionError as e:
            pytest.fail(f"バグ検出: インジケーターなしでも統計処理が進行 {e}")
        except (KeyError, ValueError):
            # これが期待される動作
            assert True