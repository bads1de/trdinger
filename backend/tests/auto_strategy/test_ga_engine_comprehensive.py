"""
GAエンジン機能の包括的テスト
"""

import pytest
import asyncio
import time
import json
import random
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import sys
import os

# パスを追加
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    IndicatorGene,
    Condition,
    encode_gene_to_list,
    decode_list_to_gene,
)
from app.core.services.auto_strategy.models.ga_config import GAConfig, GAProgress
from app.core.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.core.services.auto_strategy.engines.ga_engine import GeneticAlgorithmEngine


class TestGAEngineComprehensive:
    """GAエンジンの包括的テスト"""

    def test_deap_integration(self):
        """DEAP統合テスト"""
        print("\n=== DEAP統合テスト ===")

        # モックのBacktestServiceを作成
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "total_trades": 25,
            }
        }

        factory = StrategyFactory()
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service, factory)

        # 小規模GA設定
        config = GAConfig(
            population_size=10,
            generations=3,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elite_size=2,
        )

        try:
            # DEAP環境のセットアップ
            ga_engine.setup_deap(config)
            print("✅ DEAP環境セットアップ成功")

            # ツールボックスの確認
            assert ga_engine.toolbox is not None, "ツールボックスが初期化されていません"
            assert hasattr(
                ga_engine.toolbox, "individual"
            ), "個体生成関数が登録されていません"
            assert hasattr(
                ga_engine.toolbox, "population"
            ), "個体群生成関数が登録されていません"
            assert hasattr(
                ga_engine.toolbox, "evaluate"
            ), "評価関数が登録されていません"

            print("✅ DEAP ツールボックス検証成功")

            # 個体生成テスト
            individual = ga_engine.toolbox.individual()
            assert len(individual) == 16, f"個体長が不正: {len(individual)}"
            print(f"✅ 個体生成成功: 長さ{len(individual)}")

            # 個体群生成テスト
            population = ga_engine.toolbox.population(n=5)
            assert len(population) == 5, f"個体群サイズが不正: {len(population)}"
            print(f"✅ 個体群生成成功: {len(population)}個体")

        except Exception as e:
            print(f"❌ DEAP統合テストエラー: {e}")
            raise

    def test_fitness_calculation(self):
        """フィットネス計算テスト"""
        print("\n=== フィットネス計算テスト ===")

        # 様々なパフォーマンス結果でテスト
        test_results = [
            # 良好な結果
            {
                "performance_metrics": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.05,
                    "win_rate": 0.7,
                    "total_trades": 30,
                }
            },
            # 平均的な結果
            {
                "performance_metrics": {
                    "total_return": 0.10,
                    "sharpe_ratio": 0.8,
                    "max_drawdown": 0.15,
                    "win_rate": 0.55,
                    "total_trades": 20,
                }
            },
            # 悪い結果
            {
                "performance_metrics": {
                    "total_return": -0.05,
                    "sharpe_ratio": 0.2,
                    "max_drawdown": 0.35,
                    "win_rate": 0.4,
                    "total_trades": 5,
                }
            },
        ]

        mock_backtest_service = Mock()
        factory = StrategyFactory()
        ga_engine = GeneticAlgorithmEngine(mock_backtest_service, factory)

        config = GAConfig()

        fitness_scores = []
        for i, result in enumerate(test_results):
            fitness = ga_engine._calculate_fitness(result, config)
            fitness_scores.append(fitness)
            print(f"✅ 結果{i+1}: フィットネス = {fitness:.4f}")

        # フィットネススコアの順序確認
        assert fitness_scores[0] > fitness_scores[1], "良好な結果のフィットネスが低い"
        assert fitness_scores[1] > fitness_scores[2], "平均的な結果のフィットネスが低い"

        print("✅ フィットネス計算順序確認成功")


if __name__ == "__main__":
    print("GAエンジン包括的テスト実行")
    test_suite = TestGAEngineComprehensive()
    test_suite.test_deap_integration()
    test_suite.test_fitness_calculation()
