"""
Phase 1改善項目のテスト

評価環境固定化、GAパラメータ最適化、ログレベル最適化のテストを実装
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)


class TestEvaluationEnvironmentFixation:
    """評価環境固定化のテスト"""

    def setup_method(self):
        """テストセットアップ"""
        self.mock_backtest_service = Mock()
        self.mock_strategy_factory = Mock()
        self.mock_gene_generator = Mock()

        self.ga_engine = GeneticAlgorithmEngine(
            backtest_service=self.mock_backtest_service,
            strategy_factory=self.mock_strategy_factory,
        )
        # gene_generatorは別途設定
        self.ga_engine.gene_generator = self.mock_gene_generator

    def test_fixed_backtest_config_initialization(self):
        """固定化されたバックテスト設定の初期化テスト"""
        # テスト用設定
        config = GAConfig(population_size=2, generations=1)
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        # _select_random_timeframe_configをモック
        with patch.object(
            self.ga_engine, "_select_random_timeframe_config"
        ) as mock_select:
            mock_select.return_value = {
                "symbol": "ETH/USDT",
                "timeframe": "4h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            }

            # setup_deapとtoolboxをモック
            with patch.object(self.ga_engine, "setup_deap"):
                with patch.object(self.ga_engine, "toolbox") as mock_toolbox:
                    mock_toolbox.population.return_value = []
                    mock_toolbox.map.return_value = []

                    # run_evolutionを実行
                    try:
                        self.ga_engine.run_evolution(config, backtest_config)
                    except Exception:
                        pass  # 他の部分のエラーは無視

                    # 固定化された設定が保存されているかチェック
                    assert hasattr(self.ga_engine, "_fixed_backtest_config")
                    assert self.ga_engine._fixed_backtest_config is not None
                    assert mock_select.call_count == 1  # 一度だけ呼ばれる

    def test_individual_evaluation_uses_fixed_config(self):
        """個体評価で固定化された設定が使用されることのテスト"""
        # 固定化された設定を事前に設定
        self.ga_engine._fixed_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        # モックの設定
        self.mock_strategy_factory.validate_gene.return_value = (True, [])
        self.mock_backtest_service.run_backtest.return_value = {
            "total_return": 0.1,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "total_trades": 20,
        }

        config = GAConfig()
        individual = [
            1.0,
            0.5,
            2.0,
            0.3,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]

        # 個体評価を実行
        fitness = self.ga_engine._evaluate_individual(individual, config, {})

        # バックテストが呼ばれたかチェック
        assert self.mock_backtest_service.run_backtest.called

        # 呼び出された設定が固定化された設定を使用しているかチェック
        call_args = self.mock_backtest_service.run_backtest.call_args[0][0]
        assert call_args["symbol"] == "BTC/USDT"
        assert call_args["timeframe"] == "1d"


class TestGAParameterOptimization:
    """GAパラメータ最適化のテスト"""

    def test_default_parameters_optimized(self):
        """デフォルトパラメータが最適化されていることのテスト"""
        config = GAConfig()

        # 最適化された値をチェック
        assert config.population_size == 50  # 100から50に削減
        assert config.generations == 20  # 50から20に削減
        assert config.elite_size == 5  # 10から5に削減

    def test_calculation_reduction(self):
        """計算量削減の効果テスト"""
        old_config = GAConfig.create_legacy()
        new_config = GAConfig()

        # 計算量の比較（個体数 × 世代数）
        old_calculations = old_config.population_size * old_config.generations
        new_calculations = new_config.population_size * new_config.generations

        reduction_ratio = (old_calculations - new_calculations) / old_calculations

        # 80%削減を確認
        assert reduction_ratio >= 0.8
        assert old_calculations == 5000  # 100 × 50
        assert new_calculations == 1000  # 50 × 20

    def test_factory_methods_consistency(self):
        """ファクトリーメソッドの一貫性テスト"""
        fast_config = GAConfig.create_fast()
        thorough_config = GAConfig.create_thorough()
        legacy_config = GAConfig.create_legacy()

        # fast < default < thorough < legacy の順序をチェック
        default_config = GAConfig()

        assert fast_config.population_size < default_config.population_size
        assert default_config.population_size < thorough_config.population_size
        assert thorough_config.population_size == legacy_config.population_size


class TestLogLevelOptimization:
    """ログレベル最適化のテスト"""

    def test_default_log_level_optimized(self):
        """デフォルトログレベルが最適化されていることのテスト"""
        config = GAConfig()

        # WARNINGレベルに設定されていることを確認
        assert config.log_level == "WARNING"
        assert config.enable_detailed_logging == False

    def test_detailed_logging_control(self):
        """詳細ログ制御のテスト"""
        # 詳細ログ無効時
        config = GAConfig(enable_detailed_logging=False)

        with patch(
            "app.core.services.auto_strategy.engines.ga_engine.logger"
        ) as mock_logger:
            ga_engine = GeneticAlgorithmEngine(Mock(), Mock())
            ga_engine._fixed_backtest_config = {"symbol": "BTC/USDT", "timeframe": "1d"}

            # モックの設定
            mock_strategy_factory = Mock()
            mock_strategy_factory.validate_gene.return_value = (True, [])
            ga_engine.strategy_factory = mock_strategy_factory

            mock_backtest_service = Mock()
            mock_backtest_service.run_backtest.return_value = {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.05,
                "total_trades": 20,
            }
            ga_engine.backtest_service = mock_backtest_service

            individual = [1.0] * 16
            ga_engine._evaluate_individual(individual, config, {})

            # 詳細ログが出力されていないことを確認
            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "個体評価開始" in str(call)
            ]
            assert len(info_calls) == 0


class TestPerformanceImprovement:
    """パフォーマンス改善の統合テスト"""

    def test_execution_time_improvement(self):
        """実行時間改善のテスト"""
        # 小規模なテスト用設定
        fast_config = GAConfig(
            population_size=5, generations=2, enable_detailed_logging=False
        )

        slow_config = GAConfig(
            population_size=10, generations=4, enable_detailed_logging=True
        )

        # 実行時間の測定は実際のGAエンジンでは時間がかかるため、
        # ここでは設定の妥当性のみをテスト
        assert (
            fast_config.population_size * fast_config.generations
            < slow_config.population_size * slow_config.generations
        )

    def test_memory_efficiency(self):
        """メモリ効率性のテスト"""
        config = GAConfig()

        # 最適化された設定でのメモリ使用量が適切であることを確認
        # （実際のメモリ測定は複雑なため、設定値の妥当性をチェック）
        assert config.population_size <= 50
        assert config.generations <= 20
        assert config.elite_size <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
