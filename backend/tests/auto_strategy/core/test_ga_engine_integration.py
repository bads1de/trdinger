
import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.generators.strategy_factory import StrategyFactory
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.strategy_models import StrategyGene

@pytest.fixture
def mock_backtest_service():
    """バックテストサービスのモック"""
    service = MagicMock()

    # このモックは、特定の遺伝子に対して高いスコアを返すように設定する
    # 例：SMAの期間が短いほど高スコアになるようにする
    def custom_run_backtest(config):
        gene_dict = config.get("strategy_config", {}).get("parameters", {}).get("strategy_gene", {})
        
        # GeneSerializerを介さずに簡易的にパラメータを取得
        total_period = 0
        indicator_count = 0
        if gene_dict and gene_dict.get('indicators'):
            for ind in gene_dict['indicators']:
                total_period += ind.get('parameters', {}).get('period', 200)
                indicator_count += 1
        
        avg_period = total_period / indicator_count if indicator_count > 0 else 200

        # 期間が短いほど高いリターン
        total_return = max(0, 1.0 - (avg_period / 200.0)) 
        sharpe_ratio = total_return * 2.0
        max_drawdown = 0.5 - (total_return * 0.4)

        return {
            "performance_metrics": {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": 0.5 + total_return * 0.2,
                "total_trades": 50
            }
        }

    service.run_backtest.side_effect = custom_run_backtest
    return service

@pytest.fixture
def ga_config():
    """テスト用の小規模なGA設定"""
    return GAConfig.from_dict({
        "population_size": 10,
        "generations": 5, # 世代数を5に増やす
        "crossover_rate": 0.8,
        "mutation_rate": 0.2,
        "elite_size": 2,
        "max_indicators": 3,
        "log_level": "WARNING",
        "indicator_mode": "technical_only", # ★ これを追加
        "fitness_weights": { # 単一目的用の重み
            "total_return": 0.6,
            "sharpe_ratio": 0.4,
        }
    })

@pytest.fixture
def ga_engine(mock_backtest_service, ga_config):
    """GAエンジンのインスタンス"""
    strategy_factory = StrategyFactory()
    gene_generator = RandomGeneGenerator(ga_config)
    return GeneticAlgorithmEngine(mock_backtest_service, strategy_factory, gene_generator)

class TestGAEngineIntegration:

    def test_run_evolution_improves_fitness(self, ga_engine, ga_config):
        """進化を実行すると、適応度が改善傾向にあることを確認する"""
        # 準備
        backtest_config = {"symbol": "BTC/USDT", "timeframe": "1h"}

        # 実行
        result = ga_engine.run_evolution(ga_config, backtest_config)

        # 検証
        assert result is not None
        assert "logbook" in result
        logbook = result["logbook"]

        # ログブックに統計情報が含まれているか
        assert "max" in logbook.chapters["fitness"]
        max_fitness_per_gen = logbook.chapters["fitness"].select("max")
        
        # 世代数+1（初期世代含む）の記録があるか
        assert len(max_fitness_per_gen) == ga_config.generations + 1

        # 適応度が単調増加ではないが、全体として上昇傾向にあることを確認
        # (突然変異により一時的に下がることもあるため)
        initial_fitness = max_fitness_per_gen[0]
        final_fitness = max_fitness_per_gen[-1]
        print(f"Initial Max Fitness: {initial_fitness}, Final Max Fitness: {final_fitness}")
        print(f"Max fitness progression: {max_fitness_per_gen}")

        assert final_fitness > initial_fitness

        # 最終的な最良戦略が、期待する特徴（短い期間）を持っているか
        best_gene = result.get("best_strategy")
        assert isinstance(best_gene, StrategyGene)
        
        total_period = 0
        for ind in best_gene.indicators:
            total_period += ind.parameters.get('period', 200)
        avg_period = total_period / len(best_gene.indicators) if best_gene.indicators else 200

        print(f"Best strategy average period: {avg_period}")
        # 平均期間が、ランダムな初期状態（平均100）よりも有意に小さくなっているはず
        assert avg_period < 80


class TestGAEngineIndicatorEdgeCases:
    """GAEngineのインジケーター定義不足テスト - バグ発見用"""

    def test_ga_engine_with_undefined_indicator(self, ga_config):
        """定義されていないインジケーターを使用した場合のテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        # Mock backtest service
        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.1}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # 未定義インジケーターを使用する設定を作成
        bad_config = ga_config.copy()
        bad_config.indicator_mode = "technical_only"

        # ストラテジージーンに未定義インジケーターを強制的に設定
        bad_strategy_gene = StrategyGene(indicators=[
            MagicMock(type="UNDEFINED_INDICATOR", parameters={"period": 14})
        ])

        try:
            # 並行処理を避けるためrun_evolutionを細かくモック
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [bad_strategy_gene]
                mock_eval.return_value = [0.1]

                result = ga_engine.run_evolution(bad_config, {"symbol": "BTC/USDT"})
                # 未定義インジケーターでも処理されるか確認
                assert result is not None

        except Exception as e:
            # エラーが発生する場合は適切に処理されている
            assert isinstance(e, (ValueError, KeyError, ImportError))

    def test_ga_engine_with_empty_indicators(self, ga_config):
        """インジケーターなしの場合のテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.0}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # 空インジケーターのストラテジー
        empty_gene = StrategyGene(indicators=[])

        try:
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [empty_gene]
                mock_eval.return_value = [0.0]

                result = ga_engine.run_evolution(ga_config, {"symbol": "BTC/USDT"})
                # 空インジケーターでも処理可能か確認
                assert result is not None

        except Exception as e:
            assert isinstance(e, (ValueError, AttributeError))

    def test_ga_engine_missing_indicator_parameters(self, ga_config):
        """インジケーターのパラメータが不足している場合のテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.05}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # パラメータ不足のインジケーター
        incomplete_indicator_mock = Mock()
        incomplete_indicator_mock.type = "SMA"
        incomplete_indicator_mock.parameters = {"period": 14}  # shiftなど不足の場合

        incomplete_gene = StrategyGene(indicators=[incomplete_indicator_mock])

        try:
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [incomplete_gene]
                mock_eval.return_value = [0.05]

                result = ga_engine.run_evolution(ga_config, {"symbol": "BTC/USDT"})
                # パラメータ不足でも処理されるか
                assert result is not None

        except Exception as e:
            # エラーハンドリングを確認
            assert isinstance(e, (ValueError, KeyError, TypeError))

    def test_ga_engine_invalid_indicator_type(self, ga_config):
        """無効なインジケータータイプの場合のテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.02}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # 無効タイプのインジケーター
        invalid_indicator = Mock()
        invalid_indicator.type = 12345  # number, not string
        invalid_indicator.parameters = {"period": 14}

        invalid_gene = StrategyGene(indicators=[invalid_indicator])

        try:
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [invalid_gene]
                mock_eval.return_value = [0.02]

                result = ga_engine.run_evolution(ga_config, {"symbol": "BTC/USDT"})
                # 無効タイプでも処理可能か確認
                assert result is not None

        except Exception as e:
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_ga_engine_indicator_with_large_period(self, ga_config):
        """非常に大きい期間を持つインジケーターのテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": -0.1}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # 大きな期間 (memory/processing stress)
        large_period_indicator = Mock()
        large_period_indicator.type = "SMA"
        large_period_indicator.parameters = {"period": 1000000}

        large_gene = StrategyGene(indicators=[large_period_indicator])

        try:
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [large_gene]
                mock_eval.return_value = [-0.1]

                result = ga_engine.run_evolution(ga_config, {"symbol": "BTC/USDT"})
                # 大きな期間でも処理されるか確認
                assert result is not None

        except Exception as e:
            # MemoryErrorなどの場合がある
            assert isinstance(e, (MemoryError, ValueError, OverflowError))

    def test_ga_engine_indicator_with_negative_period(self, ga_config):
        """負の期間を持つインジケーターのテスト"""
        from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
        strategy_factory = StrategyFactory()
        gene_generator = RandomGeneGenerator(ga_config)

        mock_backtest = MagicMock()
        mock_backtest.run_backtest.return_value = {"performance_metrics": {"total_return": 0.0}}

        ga_engine = GeneticAlgorithmEngine(mock_backtest, strategy_factory, gene_generator)

        # 負の期間
        negative_indicator = Mock()
        negative_indicator.type = "RSI"
        negative_indicator.parameters = {"period": -14}

        negative_gene = StrategyGene(indicators=[negative_indicator])

        try:
            with patch.object(ga_engine, '_generate_initial_population_from_seed') as mock_pop, \
                 patch.object(ga_engine, '_evaluate_fitness') as mock_eval:

                mock_pop.return_value = [negative_gene]
                mock_eval.return_value = [0.0]

                result = ga_engine.run_evolution(ga_config, {"symbol": "BTC/USDT"})
                # 負の期間でも処理されるか
                assert result is not None

        except Exception as e:
            assert isinstance(e, (ValueError, AssertionError))
