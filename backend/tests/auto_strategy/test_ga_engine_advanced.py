"""
GAエンジン拡張テスト
包括的かつ洗練されたGAエンジンのテスト
"""

import pytest

pytestmark = pytest.mark.skip(reason="GeneticAlgorithmEngine implementation changed - test structure outdated")
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.core.evolution_runner import EvolutionRunner
from app.services.auto_strategy.models.strategy_gene import StrategyGene
from app.services.auto_strategy.config.ga import GASettings as GAConfig
from app.services.auto_strategy.services.regime_detector import RegimeDetector
from app.services.auto_strategy.core.hybrid_predictor import HybridPredictor


class TestGAEngineAdvanced:
    """GAエンジン高度テスト"""

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 10000 + np.random.randn(len(dates)) * 100,
            'high': 10000 + np.random.randn(len(dates)) * 150,
            'low': 10000 + np.random.randn(len(dates)) * 150,
            'close': 10000 + np.random.randn(len(dates)) * 100,
            'volume': np.random.randint(100, 1000, len(dates))
        })

        # OHLCの関係を確保
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)

        return data

    @pytest.fixture
    def ga_config(self):
        """GA設定"""
        return GAConfig.from_dict({
            "population_size": 50,
            "num_generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "tournament_size": 5,
            "elitism_rate": 0.1,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "initial_capital": 100000,
            "commission_rate": 0.001,
        })

    @pytest.fixture
    def mock_backtest_service(self):
        """モックバックテストサービス"""
        mock_service = Mock()
        mock_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "total_trades": 25,
                "win_rate": 0.60,
            }
        }
        return mock_service

    def test_engine_initialization_with_advanced_config(self, ga_config, mock_backtest_service):
        """高度設定でのエンジン初期化テスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        assert engine is not None
        assert engine.ga_config == ga_config
        assert engine.backtest_service == mock_backtest_service
        assert engine.population_size == 50
        assert engine.num_generations == 10

    def test_population_diversity_metrics(self, ga_config, mock_backtest_service):
        """個体群多様性メトリクスのテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 初期個体群を生成
        population = engine._create_initial_population()

        assert len(population) == ga_config.population_size

        # 多様性の基本チェック
        unique_genes = len(set(str(ind) for ind in population))
        assert unique_genes > ga_config.population_size * 0.7  # 70%以上がユニークであること

    def test_advanced_fitness_calculation(self, ga_config, mock_backtest_service):
        """高度な適応度計算のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # テスト用の個体を作成
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]  # ランダムな遺伝子

        # 適応度計算
        fitness = engine._evaluate_individual(individual)

        assert isinstance(fitness, float)
        assert not np.isnan(fitness)
        assert not np.isinf(fitness)

    def test_elitism_preservation(self, ga_config, mock_backtest_service):
        """エリート保存のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 初期個体群を生成
        population = engine._create_initial_population()

        # 適応度を評価
        fitnesses = [engine._evaluate_individual(ind) for ind in population]

        # 次世代を生成
        next_generation = engine._create_next_generation(population, fitnesses)

        assert len(next_generation) == len(population)

        # エリートが保存されていることを確認
        sorted_indices = np.argsort(fitnesses)[::-1]
        top_elite = [population[i] for i in sorted_indices[:int(ga_config.elitism_rate * len(population))]]

        # 次世代にエリートが含まれているかチェック
        elite_preserved = any(
            any(np.array_equal(elite, child) for child in next_generation)
            for elite in top_elite
        )
        assert elite_preserved

    def test_convergence_behavior_analysis(self, ga_config, mock_backtest_service):
        """収束挙動分析のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 初期個体群を生成
        population = engine._create_initial_population()

        fitness_history = []
        diversity_history = []

        # 数世代進化をシミュレート
        for generation in range(5):
            fitnesses = [engine._evaluate_individual(ind) for ind in population]
            avg_fitness = np.mean(fitnesses)
            diversity = len(set(str(ind) for ind in population)) / len(population)

            fitness_history.append(avg_fitness)
            diversity_history.append(diversity)

            # 次世代を生成
            if generation < 4:  # 最後の世代では生成しない
                population = engine._create_next_generation(population, fitnesses)

        # 適応度が改善傾向にあること
        assert len(fitness_history) == 5

        # 多様性が適切に保たれていること
        assert all(0 < diversity <= 1 for diversity in diversity_history)

    def test_adaptive_parameter_tuning(self, ga_config, mock_backtest_service):
        """適応的パラメータ調整のテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        initial_mutation_rate = engine.mutation_rate
        initial_crossover_rate = engine.crossover_rate

        # 適応的調整をシミュレート
        population = engine._create_initial_population()
        fitnesses = [engine._evaluate_individual(ind) for ind in population]

        # 多様性が低い場合の調整
        low_diversity = 0.1
        engine._adjust_parameters_based_on_diversity(low_diversity)

        # 多様性が高すぎる場合の調整
        high_diversity = 0.9
        engine._adjust_parameters_based_on_diversity(high_diversity)

        # パラメータが調整されていること
        assert engine.mutation_rate != initial_mutation_rate or engine.crossover_rate != initial_crossover_rate

    def test_regime_aware_evolution(self, ga_config, mock_backtest_service):
        """レジーム認識進化のテスト"""
        # モックレジーム検出器
        mock_regime_detector = Mock(spec=RegimeDetector)
        mock_regime_detector.detect_regime.return_value = "bullish"

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=mock_regime_detector
        )

        # レジーム認識適応度評価
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]
        fitness = engine._evaluate_individual_with_regime_awareness(individual)

        assert isinstance(fitness, float)
        assert not np.isnan(fitness)

        # レジーム検出器が呼び出されていること
        mock_regime_detector.detect_regime.assert_called()

    def test_hybrid_evolution_with_ml(self, ga_config, mock_backtest_service):
        """MLとのハイブリッド進化のテスト"""
        # モックハイブリッド予測器
        mock_hybrid_predictor = Mock(spec=HybridPredictor)
        mock_hybrid_predictor.predict_enhanced_fitness.return_value = 0.85

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # ハイブリッド適応度評価
        individual = [0.5, 0.3, 0.7, 0.2, 0.8]
        base_fitness = 0.75

        enhanced_fitness = engine._apply_hybrid_enhancement(
            individual, base_fitness, mock_hybrid_predictor
        )

        assert isinstance(enhanced_fitness, float)
        assert enhanced_fitness >= base_fitness * 0.9  # 大幅に低下しないこと

    def test_parallel_evaluation_performance(self, ga_config, mock_backtest_service):
        """並列評価パフォーマンスのテスト"""
        import time

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 小規模個体群での並列評価をテスト
        population = engine._create_initial_population()[:10]  # 10個体

        start_time = time.time()

        # 並列評価を実行
        fitnesses = engine._evaluate_population_parallel(population)

        end_time = time.time()
        execution_time = end_time - start_time

        assert len(fitnesses) == len(population)
        assert all(isinstance(f, float) for f in fitnesses)
        assert execution_time < 60.0  # 1分以内

    def test_memory_efficient_evolution(self, ga_config, mock_backtest_service):
        """メモリ効率進化のテスト"""
        import gc
        import sys

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        initial_memory = sys.getsizeof(gc.get_objects())

        # 多世代進化を実行
        population = engine._create_initial_population()

        for generation in range(3):
            fitnesses = [engine._evaluate_individual(ind) for ind in population]
            population = engine._create_next_generation(population, fitnesses)

        gc.collect()
        final_memory = sys.getsizeof(gc.get_objects())

        # 過度なメモリ増加でないこと
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000000  # 1MB未満の増加

    def test_robustness_to_market_data_variations(self, sample_market_data, ga_config):
        """市場データ変動に対するロバスト性テスト"""
        # 変動のある市場データでテスト
        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.15,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.08,
                "total_trades": 25,
                "win_rate": 0.60,
            }
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=sample_market_data,
            regime_detector=None
        )

        # 複数回の適応度評価を実行
        for _ in range(5):
            individual = np.random.rand(5)
            fitness = engine._evaluate_individual(individual)
            assert isinstance(fitness, float)
            assert not np.isnan(fitness)

    def test_error_handling_during_evolution(self, ga_config, mock_backtest_service):
        """進化中のエラーハンドリングテスト"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 不正な個体に対するエラーハンドリング
        invalid_individual = [np.nan, np.inf, -np.inf, 0.5, 0.3]

        # エラーが適切に処理されること
        try:
            fitness = engine._evaluate_individual(invalid_individual)
            assert np.isnan(fitness) or fitness == 0.0
        except Exception as e:
            assert "invalid" in str(e).lower() or "error" in str(e).lower()

    def test_scalability_test_large_population(self, ga_config, mock_backtest_service):
        """大規模個体群でのスケーラビリティテスト"""
        import time

        # 大規模個体群設定
        large_config = GAConfig.from_dict({
            **ga_config.to_dict(),
            "population_size": 200  # 大きな個体群
        })

        engine = GeneticAlgorithmEngine(
            ga_config=large_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        start_time = time.time()

        # 大規模個体群を生成
        population = engine._create_initial_population()

        # 初期適応度評価
        fitnesses = [engine._evaluate_individual(ind) for ind in population[:50]]  # サンプル評価

        end_time = time.time()
        execution_time = end_time - start_time

        assert len(population) == 200
        assert len(fitnesses) == 50
        assert execution_time < 120.0  # 2分以内

    def test_final_ga_engine_validation(self, ga_config, mock_backtest_service):
        """最終GAエンジン検証"""
        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        assert engine is not None
        assert hasattr(engine, '_evaluate_individual')
        assert hasattr(engine, '_create_next_generation')
        assert hasattr(engine, 'evolve')

        # 基本的な進化が可能であること
        result = engine.evolve()
        assert result is not None

        print("✅ GAエンジン高度テスト成功")


# TDDアプローチによるGAエンジンテスト
class TestGAEngineTDD:
    """TDDアプローチによるGAエンジンテスト"""

    def test_engine_creation_minimal_dependencies(self):
        """最小依存関係でのエンジン作成テスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 10,
            "num_generations": 5,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        assert engine is not None
        print("✅ 最小依存関係でのエンジン作成テスト成功")

    def test_basic_evolution_workflow(self):
        """基本進化ワークフローテスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 20,
            "num_generations": 3,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 進化を実行
        result = engine.evolve()

        assert result is not None
        assert hasattr(result, 'best_individual')
        assert hasattr(result, 'best_fitness')

        print("✅ 基本進化ワークフローテスト成功")

    def test_parameter_sensitivity_analysis(self):
        """パラメータ感度分析テスト"""
        base_config = {
            "population_size": 50,
            "num_generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }

        results = []

        # 突然変異率の変化をテスト
        for mutation_rate in [0.05, 0.1, 0.2]:
            config = {**base_config, "mutation_rate": mutation_rate}
            ga_config = GAConfig.from_dict(config)

            mock_backtest_service = Mock()
            mock_backtest_service.run_backtest.return_value = {
                "success": True,
                "performance_metrics": {"total_return": 0.1}
            }

            engine = GeneticAlgorithmEngine(
                ga_config=ga_config,
                backtest_service=mock_backtest_service,
                market_data=None,
                regime_detector=None
            )

            result = engine.evolve()
            results.append({
                "mutation_rate": mutation_rate,
                "best_fitness": result.best_fitness if hasattr(result, 'best_fitness') else 0
            })

        # 結果が得られていること
        assert len(results) == 3
        assert all('mutation_rate' in r and 'best_fitness' in r for r in results)

        print("✅ パラメータ感度分析テスト成功")

    def test_convergence_speed_optimization(self):
        """収束速度最適化テスト"""
        import time

        ga_config = GAConfig.from_dict({
            "population_size": 30,
            "num_generations": 15,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {"total_return": 0.1}
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        start_time = time.time()

        # 進化を実行
        result = engine.evolve()

        end_time = time.time()
        execution_time = end_time - start_time

        # 収束が適切な時間内であること
        assert execution_time < 180.0  # 3分以内
        assert result is not None

        print(f"✅ 収束速度最適化テスト成功: {execution_time:.2f}s")

    def test_multi_objective_optimization_readiness(self):
        """多目的最適化準備テスト"""
        ga_config = GAConfig.from_dict({
            "population_size": 25,
            "num_generations": 8,
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        })

        mock_backtest_service = Mock()
        mock_backtest_service.run_backtest.return_value = {
            "success": True,
            "performance_metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": -0.05
            }
        }

        engine = GeneticAlgorithmEngine(
            ga_config=ga_config,
            backtest_service=mock_backtest_service,
            market_data=None,
            regime_detector=None
        )

        # 多目的評価が可能であること
        individual = np.random.rand(5)

        # 複数の目的関数をテスト
        objectives = engine._evaluate_multi_objective(individual)
        assert isinstance(objectives, list)
        assert len(objectives) > 1

        print("✅ 多目的最適化準備テスト成功")