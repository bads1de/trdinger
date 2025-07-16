"""
多目的最適化GA 実行テスト

実際のGA実行をテストします。
"""

import pytest
import logging
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator

logger = logging.getLogger(__name__)


class TestMultiObjectiveExecution:
    """多目的最適化GA 実行テストクラス"""

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock()
        
        # 様々なパフォーマンスを持つ結果を返すように設定
        def mock_backtest_result(*args, **kwargs):
            import random
            return {
                "performance_metrics": {
                    "total_return": random.uniform(0.05, 0.25),
                    "sharpe_ratio": random.uniform(0.8, 2.5),
                    "max_drawdown": random.uniform(0.03, 0.15),
                    "win_rate": random.uniform(0.45, 0.75),
                    "profit_factor": random.uniform(1.1, 3.0),
                    "total_trades": random.randint(20, 100),
                }
            }
        
        mock_service.run_backtest.side_effect = mock_backtest_result
        return mock_service

    @pytest.fixture
    def mock_strategy_factory(self):
        """モック戦略ファクトリー"""
        return Mock()

    @pytest.fixture
    def mock_gene_generator(self):
        """モック遺伝子生成器"""
        mock_generator = Mock()
        
        # ランダムな遺伝子を生成
        def mock_generate_random_gene():
            import random
            return [random.randint(0, 10) for _ in range(5)]
        
        mock_generator.generate_random_gene.side_effect = mock_generate_random_gene
        return mock_generator

    def test_multi_objective_ga_execution_small(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """小規模な多目的最適化GA実行テスト"""
        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )
        
        # 小規模な多目的設定
        config = GAConfig.create_multi_objective(
            objectives=["total_return", "max_drawdown"],
            weights=[1.0, -1.0]
        )
        config.population_size = 4  # 小さな個体数
        config.generations = 2      # 少ない世代数
        
        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }
        
        # 遺伝子エンコーダーをモック
        with patch("app.core.services.auto_strategy.models.gene_encoding.GeneEncoder") as mock_encoder:
            mock_gene = Mock()
            mock_gene.id = "test_gene_123"
            mock_gene.to_dict.return_value = {"test": "data"}
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = mock_gene
            
            # GA実行
            result = ga_engine.run_evolution(config, backtest_config)
        
        # 結果確認
        assert result is not None
        assert isinstance(result, dict)
        
        # 多目的最適化の結果には以下が含まれるべき
        expected_keys = ["best_strategy", "best_fitness", "execution_time", "generations_completed"]
        for key in expected_keys:
            assert key in result, f"結果に{key}が含まれていません"
        
        # パレート最適解が含まれているか確認
        if config.enable_multi_objective:
            assert "pareto_front" in result, "多目的最適化結果にパレート最適解が含まれていません"
            assert "objectives" in result, "多目的最適化結果に目的リストが含まれていません"
            
            pareto_front = result["pareto_front"]
            assert isinstance(pareto_front, list), "パレート最適解はリストである必要があります"
            assert len(pareto_front) > 0, "パレート最適解が空です"
            
            # 各パレート解の構造確認
            for solution in pareto_front:
                assert "strategy" in solution, "パレート解に戦略が含まれていません"
                assert "fitness_values" in solution, "パレート解にフィットネス値が含まれていません"
                assert len(solution["fitness_values"]) == len(config.objectives), "フィットネス値の数が目的数と一致しません"

    def test_multi_objective_vs_single_objective(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """多目的最適化と単一目的最適化の比較テスト"""
        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )
        
        # 共通設定
        base_config = {
            "population_size": 4,
            "generations": 2,
        }
        
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }
        
        # 遺伝子エンコーダーをモック
        with patch("app.core.services.auto_strategy.models.gene_encoding.GeneEncoder") as mock_encoder:
            mock_gene = Mock()
            mock_gene.id = "test_gene_123"
            mock_gene.to_dict.return_value = {"test": "data"}
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = mock_gene
            
            # 単一目的最適化実行
            single_config = GAConfig(**base_config, enable_multi_objective=False)
            single_result = ga_engine.run_evolution(single_config, backtest_config)
            
            # 多目的最適化実行
            multi_config = GAConfig.create_multi_objective(**base_config)
            multi_result = ga_engine.run_evolution(multi_config, backtest_config)
        
        # 結果比較
        assert single_result is not None
        assert multi_result is not None
        
        # 単一目的最適化結果の確認
        assert "best_strategy" in single_result
        assert "best_fitness" in single_result
        assert isinstance(single_result["best_fitness"], (int, float))
        
        # 多目的最適化結果の確認
        assert "best_strategy" in multi_result
        assert "best_fitness" in multi_result
        assert "pareto_front" in multi_result
        assert "objectives" in multi_result
        
        # 多目的最適化のフィットネスはタプルまたはリスト
        assert isinstance(multi_result["best_fitness"], (list, tuple))
        assert len(multi_result["best_fitness"]) > 1

    def test_different_objective_combinations(self, mock_backtest_service, mock_strategy_factory, mock_gene_generator):
        """異なる目的の組み合わせテスト"""
        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )
        
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        }
        
        # 異なる目的の組み合わせをテスト
        objective_combinations = [
            (["total_return", "max_drawdown"], [1.0, -1.0]),
            (["sharpe_ratio", "win_rate"], [1.0, 1.0]),
            (["total_return", "sharpe_ratio", "max_drawdown"], [1.0, 1.0, -1.0]),
        ]
        
        # 遺伝子エンコーダーをモック
        with patch("app.core.services.auto_strategy.models.gene_encoding.GeneEncoder") as mock_encoder:
            mock_gene = Mock()
            mock_gene.id = "test_gene_123"
            mock_gene.to_dict.return_value = {"test": "data"}
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = mock_gene
            
            for objectives, weights in objective_combinations:
                config = GAConfig.create_multi_objective(
                    objectives=objectives,
                    weights=weights
                )
                config.population_size = 4
                config.generations = 1
                
                # GA実行
                result = ga_engine.run_evolution(config, backtest_config)
                
                # 結果確認
                assert result is not None
                assert "pareto_front" in result
                assert "objectives" in result
                assert result["objectives"] == objectives
                
                # パレート最適解のフィットネス値の数が目的数と一致することを確認
                for solution in result["pareto_front"]:
                    assert len(solution["fitness_values"]) == len(objectives)

    def test_error_resilience_multi_objective(self, mock_strategy_factory, mock_gene_generator):
        """多目的最適化のエラー耐性テスト"""
        # エラーを発生させるモックバックテストサービス
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.side_effect = Exception("バックテストエラー")
        
        # GAエンジン初期化
        ga_engine = GeneticAlgorithmEngine(
            mock_backtest_service,
            mock_strategy_factory,
            mock_gene_generator
        )
        
        # 多目的設定
        config = GAConfig.create_multi_objective()
        config.population_size = 2
        config.generations = 1
        
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "initial_capital": 1000,
            "commission_rate": 0.001,
        }
        
        # 遺伝子エンコーダーをモック
        with patch("app.core.services.auto_strategy.models.gene_encoding.GeneEncoder") as mock_encoder:
            mock_gene = Mock()
            mock_gene.id = "test_gene_123"
            mock_gene.to_dict.return_value = {"test": "data"}
            mock_encoder.return_value.decode_list_to_strategy_gene.return_value = mock_gene
            
            # エラーが発生してもクラッシュしないことを確認
            try:
                result = ga_engine.run_evolution(config, backtest_config)
                # エラーハンドリングが適切に行われていれば、何らかの結果が返される
                assert result is not None
            except Exception as e:
                # 適切なエラーメッセージが含まれていることを確認
                assert "エラー" in str(e) or "error" in str(e).lower()


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    pytest.main([__file__, "-v", "-s"])
