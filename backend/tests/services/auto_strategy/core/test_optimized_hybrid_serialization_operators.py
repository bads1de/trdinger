"""
最新の最適化（ハイブリッドGA+ML、シリアライゼーション、交叉・突然変異演算子）のテスト
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import pytest
from unittest.mock import MagicMock, Mock

from app.services.auto_strategy.config.ga import GAConfig


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def ga_config():
    """GA設定のフィクスチャ"""
    config = GAConfig()
    config.fitness_weights = {
        "total_return": 0.3,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.2,
        "win_rate": 0.1,
        "balance_score": 0.1,
    }
    config.max_indicators = 5
    config.min_indicators = 1
    config.max_conditions = 3
    config.min_conditions = 1
    return config


# =============================================================================
# OptimizedHybridIndividualEvaluator のテスト
# =============================================================================

class TestOptimizedHybridIndividualEvaluator:
    """最適化されたハイブリッドGA個体評価器のテスト"""

    def test_initialization(self):
        """初期化テスト"""
        from app.services.auto_strategy.core.hybrid.optimized_hybrid_individual_evaluator import (
            OptimizedHybridIndividualEvaluator,
        )

        mock_backtest_service = MagicMock()
        evaluator = OptimizedHybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            cache_size=100,
        )

        assert evaluator is not None
        assert evaluator._cache_size == 100

    def test_cache_statistics(self):
        """キャッシュ統計テスト"""
        from app.services.auto_strategy.core.hybrid.optimized_hybrid_individual_evaluator import (
            OptimizedHybridIndividualEvaluator,
        )

        mock_backtest_service = MagicMock()
        evaluator = OptimizedHybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            cache_size=100,
        )

        stats = evaluator.get_cache_statistics()

        assert "prediction_cache_size" in stats
        assert "feature_cache_size" in stats
        assert "cache_limit" in stats

    def test_clear_caches(self):
        """キャッシュクリアテスト"""
        from app.services.auto_strategy.core.hybrid.optimized_hybrid_individual_evaluator import (
            OptimizedHybridIndividualEvaluator,
        )

        mock_backtest_service = MagicMock()
        evaluator = OptimizedHybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            cache_size=100,
        )

        # キャッシュにデータを追加
        evaluator._prediction_cache["test"] = MagicMock()
        evaluator._feature_cache["test"] = MagicMock()

        # キャッシュクリア
        evaluator.clear_caches()

        assert len(evaluator._prediction_cache) == 0
        assert len(evaluator._feature_cache) == 0


# =============================================================================
# OptimizedDictConverter のテスト
# =============================================================================

class TestOptimizedDictConverter:
    """最適化されたシリアライゼーションのテスト"""

    def test_initialization(self):
        """初期化テスト"""
        from app.services.auto_strategy.serializers.optimized_serialization import (
            OptimizedDictConverter,
        )

        converter = OptimizedDictConverter(cache_size=100)

        assert converter is not None
        assert converter._cache_size == 100

    def test_cache_statistics(self):
        """キャッシュ統計テスト"""
        from app.services.auto_strategy.serializers.optimized_serialization import (
            OptimizedDictConverter,
        )

        converter = OptimizedDictConverter(cache_size=100)

        stats = converter.get_cache_statistics()

        assert "serialize_cache_size" in stats
        assert "deserialize_cache_size" in stats
        assert "cache_limit" in stats

    def test_clear_caches(self):
        """キャッシュクリアテスト"""
        from app.services.auto_strategy.serializers.optimized_serialization import (
            OptimizedDictConverter,
        )

        converter = OptimizedDictConverter(cache_size=100)

        # キャッシュにデータを追加
        converter._serialize_cache["test"] = {"type": "SMA"}
        converter._deserialize_cache["test"] = MagicMock()

        # キャッシュクリア
        converter.clear_caches()

        assert len(converter._serialize_cache) == 0
        assert len(converter._deserialize_cache) == 0


# =============================================================================
# 最適化された交叉・突然変異演算子のテスト
# =============================================================================

class TestOptimizedStrategyOperators:
    """最適化された交叉・突然変異演算子のテスト"""

    def test_mutate_strategy_gene_batch(self, ga_config):
        """突然変異バッチ処理テスト"""
        from app.services.auto_strategy.genes.optimized_strategy_operators import (
            mutate_strategy_gene_batch,
        )
        from app.services.auto_strategy.genes import StrategyGene, IndicatorGene, Condition

        # テスト用遺伝子を生成
        individuals = []
        for i in range(5):
            gene = StrategyGene(
                id=f"gene_{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                long_entry_conditions=[
                    Condition(left_operand="sma_20", operator=">", right_operand="close")
                ],
                short_entry_conditions=[],
            )
            individuals.append(gene)

        # 突然変異を実行
        results = mutate_strategy_gene_batch(individuals, ga_config, mutation_rate=0.1)

        assert len(results) == len(individuals)
        for result in results:
            assert result is not None

    def test_crossover_strategy_genes_batch(self, ga_config):
        """交叉バッチ処理テスト"""
        from app.services.auto_strategy.genes.optimized_strategy_operators import (
            crossover_strategy_genes_batch,
        )
        from app.services.auto_strategy.genes import StrategyGene, IndicatorGene, Condition

        # テスト用遺伝子を生成
        individuals = []
        for i in range(4):
            gene = StrategyGene(
                id=f"gene_{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                long_entry_conditions=[
                    Condition(left_operand="sma_20", operator=">", right_operand="close")
                ],
                short_entry_conditions=[],
            )
            individuals.append(gene)

        # 交叉を実行
        results = crossover_strategy_genes_batch(individuals, ga_config, crossover_rate=0.8)

        assert len(results) == len(individuals) // 2
        for child1, child2 in results:
            assert child1 is not None
            assert child2 is not None


# =============================================================================
# 統合テスト
# =============================================================================

class TestIntegration:
    """最新の最適化の統合テスト"""

    def test_all_optimizations_work_together(self, ga_config):
        """全ての最適化が連携して動作するかテスト"""
        from app.services.auto_strategy.core.hybrid.optimized_hybrid_individual_evaluator import (
            OptimizedHybridIndividualEvaluator,
        )
        from app.services.auto_strategy.serializers.optimized_serialization import (
            OptimizedDictConverter,
        )
        from app.services.auto_strategy.genes.optimized_strategy_operators import (
            mutate_strategy_gene_batch,
        )
        from app.services.auto_strategy.genes import StrategyGene, IndicatorGene, Condition

        # ハイブリッド評価器
        mock_backtest_service = MagicMock()
        hybrid_evaluator = OptimizedHybridIndividualEvaluator(
            backtest_service=mock_backtest_service,
            cache_size=100,
        )

        # シリアライザー
        serializer = OptimizedDictConverter(cache_size=100)

        # テスト用遺伝子を生成
        individuals = []
        for i in range(3):
            gene = StrategyGene(
                id=f"gene_{i}",
                indicators=[IndicatorGene(type="SMA", parameters={"period": 20})],
                long_entry_conditions=[
                    Condition(left_operand="sma_20", operator=">", right_operand="close")
                ],
                short_entry_conditions=[],
            )
            individuals.append(gene)

        # 突然変異を実行
        mutated = mutate_strategy_gene_batch(individuals, ga_config, mutation_rate=0.1)

        # シリアライズを実行
        for gene in mutated:
            serialized = serializer.strategy_gene_to_dict(gene)
            assert serialized is not None

        # キャッシュ統計を確認
        hybrid_stats = hybrid_evaluator.get_cache_statistics()
        serializer_stats = serializer.get_cache_statistics()

        assert hybrid_stats["cache_limit"] == 100
        assert serializer_stats["cache_limit"] == 100
