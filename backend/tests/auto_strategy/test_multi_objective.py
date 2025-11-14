"""
多目的最適化のテストモジュール

EvolutionRunnerの多目的最適化機能をテストする。
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from app.services.auto_strategy.config.ga_runtime import GAConfig
from app.services.auto_strategy.core.evolution_runner import EvolutionRunner


class TestEvolutionRunnerMultiObjective:
    """EvolutionRunnerの多目的最適化テスト"""

    @pytest.fixture
    def mock_toolbox(self):
        """Mock DEAPツールボックス"""
        from types import SimpleNamespace

        toolbox = Mock()

        # カスタムFitnessクラスを定義（delattrが動作するように）
        class MockFitness:
            def __init__(self, valid=True, values=()):
                self.valid = valid
                self.values = values

            def __delattr__(self, name):
                if name == "values":
                    self.valid = False
                    self.values = ()
                else:
                    super().__delattr__(name)

        # cloneは個体のコピーを返す - 属性を適切に設定
        def clone_func(ind):
            cloned = SimpleNamespace()
            # fitnessオブジェクトを作成
            try:
                fitness = MockFitness(
                    valid=getattr(ind.fitness, "valid", True),
                    values=getattr(ind.fitness, "values", ()),
                )
            except AttributeError:
                fitness = MockFitness(valid=True, values=())

            cloned.fitness = fitness
            return cloned

        toolbox.clone = Mock(side_effect=clone_func)

        # mapは関数をリストの各要素に適用
        def mock_map(func, items):
            return [func(item) for item in items]

        toolbox.map = mock_map
        toolbox.evaluate = Mock(return_value=(1.5, 0.8, 0.2))  # 多目的フィットネス

        # mateとmutateは個体のタプルを返す（元の個体を変更するのでそのまま返す）
        def mock_mate(ind1, ind2):
            # 交叉は2つの個体を返す
            return (ind1, ind2)

        def mock_mutate(ind):
            # 突然変異は1つの個体のタプルを返す
            return (ind,)

        toolbox.mate = Mock(side_effect=mock_mate)
        toolbox.mutate = Mock(side_effect=mock_mutate)
        toolbox.select = Mock(return_value=[])
        return toolbox

    @pytest.fixture
    def mock_stats(self):
        """Mock統計オブジェクト"""
        stats = Mock()
        stats.compile = Mock(return_value={"avg": 1.0, "std": 0.5})
        return stats

    @pytest.fixture
    def mock_config(self):
        """Mock GA設定（多目的最適化用）"""
        config = Mock()
        config.generations = 2
        config.crossover_rate = 0.8
        config.mutation_rate = 0.2
        config.enable_fitness_sharing = False
        config.dynamic_objective_reweighting = False
        config.objectives = ["sharpe_ratio", "total_return", "max_drawdown"]
        return config

    @pytest.fixture
    def mock_population(self):
        """Mock個体群"""
        # Mock個体を作成
        population = []
        for i in range(4):
            individual = Mock()
            individual.fitness = Mock()
            individual.fitness.valid = True
            individual.fitness.values = (1.0 + i * 0.1, 0.5 + i * 0.05, 0.1 + i * 0.02)
            population.append(individual)
        return population

    def test_run_multi_objective_evolution_basic(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """多目的最適化の基本実行テスト"""
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        # モックの設定 - return_value not side_effect, and return the actual list
        def mock_select(pop, k):
            return pop[:k]

        mock_toolbox.select = mock_select

        with patch(
            "app.services.auto_strategy.core.evolution_runner.tools.selNSGA2",
            side_effect=lambda pop, k: pop[:k],  # NSGA2選択のモック
        ) as mock_sel_nsga2:
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
            ) as mock_pareto_front:
                with patch(
                    "app.services.auto_strategy.core.evolution_runner.tools.Logbook"
                ) as mock_logbook_class:
                    mock_logbook = Mock()
                    mock_logbook.record = Mock()
                    mock_logbook_class.return_value = mock_logbook

                    result_pop, logbook = runner.run_multi_objective_evolution(
                        mock_population, mock_config
                    )

        # ParetoFrontが作成されたことを確認
        mock_pareto_front.assert_called_once()

        # 結果が返されることを確認
        assert isinstance(result_pop, list)
        assert logbook is not None

    def test_run_multi_objective_evolution_with_fitness_sharing(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """適応度共有ありの多目的最適化テスト"""
        mock_config.enable_fitness_sharing = True
        mock_fitness_sharing = Mock()
        mock_fitness_sharing.apply_fitness_sharing.return_value = mock_population

        runner = EvolutionRunner(
            mock_toolbox, mock_stats, fitness_sharing=mock_fitness_sharing
        )

        def mock_select(pop, k):
            return pop[:k]

        mock_toolbox.select = mock_select

        with patch(
            "app.services.auto_strategy.core.evolution_runner.tools.selNSGA2",
            side_effect=lambda pop, k: pop[:k],
        ):
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
            ):
                with patch(
                    "app.services.auto_strategy.core.evolution_runner.tools.Logbook"
                ):
                    runner.run_multi_objective_evolution(mock_population, mock_config)

        # 適応度共有が適用されたことを確認
        assert mock_fitness_sharing.apply_fitness_sharing.called

    def test_run_multi_objective_evolution_with_halloffame(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """Hall of Fameありの多目的最適化テスト"""
        mock_halloffame = Mock()
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        mock_toolbox.select.side_effect = [mock_population, mock_population]

        with patch("app.services.auto_strategy.core.evolution_runner.tools.selNSGA2"):
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
            ):
                runner.run_multi_objective_evolution(
                    mock_population, mock_config, halloffame=mock_halloffame
                )

        # Hall of Fameが更新されたことを確認
        mock_halloffame.update.assert_called()

    def test_evaluate_population_multi_objective(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """個体群評価テスト（多目的）"""
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        result = runner._evaluate_population(mock_population)

        # 各個体のfitness.valuesが設定されたことを確認
        for individual in result:
            assert individual.fitness.values == (1.5, 0.8, 0.2)

    def test_update_dynamic_objective_scalars_disabled(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """動的目標スケーリング無効テスト"""
        mock_config.dynamic_objective_reweighting = False
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner._update_dynamic_objective_scalars(mock_population, mock_config)

        assert mock_config.objective_dynamic_scalars == {}

    def test_update_dynamic_objective_scalars_enabled(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """動的目標スケーリング有効テスト"""
        mock_config.dynamic_objective_reweighting = True
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner._update_dynamic_objective_scalars(mock_population, mock_config)

        # スケーリングファクタが設定されたことを確認
        assert isinstance(mock_config.objective_dynamic_scalars, dict)

    def test_update_dynamic_objective_scalars_empty_population(
        self, mock_toolbox, mock_stats, mock_config
    ):
        """空個体群の動的目標スケーリングテスト"""
        mock_config.dynamic_objective_reweighting = True
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        runner._update_dynamic_objective_scalars([], mock_config)

        assert mock_config.objective_dynamic_scalars == {}

    def test_run_multi_objective_evolution_logging(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """多目的最適化のログ出力テスト"""
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        mock_toolbox.select.side_effect = [mock_population, mock_population]

        with patch(
            "app.services.auto_strategy.core.evolution_runner.logger"
        ) as mock_logger:
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.selNSGA2"
            ):
                with patch(
                    "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
                ):
                    runner.run_multi_objective_evolution(mock_population, mock_config)

        # ログが正しく出力されたことを確認
        mock_logger.info.assert_any_call("多目的最適化アルゴリズム（NSGA-II）を開始")
        mock_logger.info.assert_any_call("多目的最適化アルゴリズム（NSGA-II）完了")

    def test_run_multi_objective_evolution_select_function_restoration(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """選択関数復元テスト"""
        original_select = Mock()
        mock_toolbox.select = original_select
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        mock_toolbox.select.side_effect = [mock_population, mock_population]

        with patch(
            "app.services.auto_strategy.core.evolution_runner.tools.selNSGA2"
        ) as mock_sel_nsga2:
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
            ):
                runner.run_multi_objective_evolution(mock_population, mock_config)

        # 選択関数が元に戻されたことを確認
        assert mock_toolbox.select == original_select

    def test_run_multi_objective_evolution_generations_zero(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """世代数0の多目的最適化テスト"""
        mock_config.generations = 0
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        mock_toolbox.select.side_effect = [mock_population]

        with patch("app.services.auto_strategy.core.evolution_runner.tools.selNSGA2"):
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
            ):
                result_pop, logbook = runner.run_multi_objective_evolution(
                    mock_population, mock_config
                )

        # 初期評価のみが行われることを確認
        assert len(mock_toolbox.evaluate.call_args_list) == len(mock_population)

    def test_multi_objective_with_crossover_and_mutation(
        self, mock_toolbox, mock_stats, mock_config, mock_population
    ):
        """交叉と突然変異を含む多目的最適化テスト"""
        runner = EvolutionRunner(mock_toolbox, mock_stats)

        # 交叉と突然変異が発生するように設定
        def mock_select(pop, k):
            return pop[:k]

        mock_toolbox.select = mock_select

        with patch(
            "app.services.auto_strategy.core.evolution_runner.random.random",
            side_effect=[0.1] * 20,  # 十分な数の値を提供（交叉+突然変異のチェック用）
        ):  # 確率以下
            with patch(
                "app.services.auto_strategy.core.evolution_runner.tools.selNSGA2",
                side_effect=lambda pop, k: pop[:k],
            ):
                with patch(
                    "app.services.auto_strategy.core.evolution_runner.tools.ParetoFront"
                ):
                    with patch(
                        "app.services.auto_strategy.core.evolution_runner.tools.Logbook"
                    ):
                        runner.run_multi_objective_evolution(
                            mock_population, mock_config
                        )

        # 交叉と突然変異が呼ばれたことを確認
        assert mock_toolbox.mate.called
        assert mock_toolbox.mutate.called
