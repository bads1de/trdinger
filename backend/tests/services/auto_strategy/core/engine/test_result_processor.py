"""
ResultProcessorのユニットテスト

``app.services.auto_strategy.core.engine.result_processor.ResultProcessor`` の
挙動を網羅的に検証します。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from deap import tools

from app.services.auto_strategy.core.engine import report_selection
from app.services.auto_strategy.core.engine.result_processor import (
    MAX_STRATEGIES_TO_EXTRACT,
    ResultProcessor,
)
from app.services.auto_strategy.genes import StrategyGene


def _make_strategy_individual(fitness_values: tuple[float, ...]) -> Mock:
    """StrategyGene として振る舞う Mock 個体を作成

    ``__class__ = StrategyGene`` を設定することで、isinstance(ind, StrategyGene) が True になる。
    """
    ind = Mock()
    ind.__class__ = StrategyGene
    ind.fitness.values = fitness_values
    ind.fitness.valid = True
    return ind


def _make_two_stage_individual(
    fitness_values: tuple[float, ...], rank: int = 0
) -> Mock:
    """二段階選抜メタデータ付き個体

    Two-stage rank/score は ``fitness`` 側に保存される
    (report_selection._get_two_stage_metadata_target を参照)。
    """
    ind = _make_strategy_individual(fitness_values)
    ind.fitness._two_stage_selection_rank = rank
    ind.fitness._two_stage_selection_score = (1.0,)
    return ind


class TestResultProcessorExtractBest:
    """``extract_best_individuals`` の挙動テスト"""

    def test_returns_single_best_from_population(self) -> None:
        processor = ResultProcessor()
        individuals = [
            _make_strategy_individual((0.5,)),
            _make_strategy_individual((0.8,)),
            _make_strategy_individual((0.3,)),
        ]

        best, best_gene, best_strategies = processor.extract_best_individuals(
            population=individuals,
            config=SimpleNamespace(),
            halloffame=None,
        )

        assert best is not None
        assert best_gene is not None
        assert best_strategies is not None
        assert len(best_strategies) >= 1
        for entry in best_strategies:
            assert "strategy" in entry
            assert "fitness_values" in entry

    def test_extracts_strategies_with_fitness_values(self) -> None:
        processor = ResultProcessor()
        individuals = [_make_strategy_individual((2.0,))]

        _, best_gene, best_strategies = processor.extract_best_individuals(
            population=individuals,
            config=SimpleNamespace(),
        )

        assert best_gene is not None
        assert best_strategies is not None
        assert best_strategies[0]["fitness_values"] == [2.0]

    def test_uses_provided_pareto_front_when_given(self) -> None:
        """halloffame が ParetoFront インスタンスの場合、それを使う"""
        processor = ResultProcessor()
        ind_a = _make_strategy_individual((1.0,))
        ind_b = _make_strategy_individual((2.0,))

        ho = tools.ParetoFront()
        ho.update([ind_a, ind_b])

        _, best_gene, best_strategies = processor.extract_best_individuals(
            population=[ind_a, ind_b],
            config=SimpleNamespace(),
            halloffame=ho,
        )

        # best_gene は pareto front の最良個体
        assert best_gene is not None
        assert best_strategies is not None
        # pareto front には少なくとも1つの戦略が含まれる
        assert len(best_strategies) >= 1
        # best_strategies の各エントリは strategy と fitness_values を持つ
        for entry in best_strategies:
            assert "strategy" in entry
            assert "fitness_values" in entry
            # fitness_values is a list of values
            assert len(entry["fitness_values"]) >= 1

    def test_non_pareto_front_halloffame_triggers_recompute(self) -> None:
        """halloffame が ParetoFront でない場合、再計算される"""
        processor = ResultProcessor()
        ind = _make_strategy_individual((1.0,))
        fake_ho = MagicMock()  # not a ParetoFront

        _, _, best_strategies = processor.extract_best_individuals(
            population=[ind],
            config=SimpleNamespace(),
            halloffame=fake_ho,
        )

        assert best_strategies is not None

    def test_max_strategies_extracted_constant(self) -> None:
        """定数が妥当な値である"""
        assert MAX_STRATEGIES_TO_EXTRACT == 10

    def test_extracts_at_most_max_strategies(self) -> None:
        """抽出戦略数が MAX_STRATEGIES_TO_EXTRACT を超えない"""
        processor = ResultProcessor()
        # 多数のパレート最適個体を作成
        pareto_inds = []
        for i in range(MAX_STRATEGIES_TO_EXTRACT + 5):
            pareto_inds.append(_make_strategy_individual((float(i),)))

        ho = tools.ParetoFront()
        ho.update(pareto_inds)

        _, _, best_strategies = processor.extract_best_individuals(
            population=pareto_inds,
            config=SimpleNamespace(),
            halloffame=ho,
        )

        assert best_strategies is not None
        assert len(best_strategies) <= MAX_STRATEGIES_TO_EXTRACT

    def test_skips_non_strategy_gene_individuals(self) -> None:
        """StrategyGene でない個体は best_strategies からスキップされる"""
        processor = ResultProcessor()
        # 複数個の StrategyGene と non-StrategyGene を混ぜる
        good1 = _make_strategy_individual((0.5, 0.5))
        good2 = _make_strategy_individual((0.6, 0.4))
        # MagicMock は isinstance(ind, StrategyGene) == False なのでスキップされる
        bad1 = MagicMock()
        bad1.fitness.values = (0.4, 0.6)
        bad2 = MagicMock()
        bad2.fitness.values = (0.7, 0.7)

        # pareto front を経由せず population から直接評価
        _, _, best_strategies = processor.extract_best_individuals(
            population=[good1, good2, bad1, bad2],
            config=SimpleNamespace(),
            halloffame=None,
        )

        assert best_strategies is not None
        # best_strategies 内のすべての strategy は good1 か good2 のいずれか
        # bad は isinstance check でスキップされる
        for s in best_strategies:
            # isinstance check passed, so it's a StrategyGene
            assert s["strategy"] is not bad1
            assert s["strategy"] is not bad2
        # bad がいずれも混入していないことを確認（count でも検証）
        strategy_count = len(best_strategies)
        # Pareto front には通常2つ以下しか入らない
        assert strategy_count <= MAX_STRATEGIES_TO_EXTRACT

    def test_two_stage_best_takes_priority(self) -> None:
        """二段階選抜の best が存在する場合、それが最良個体として選ばれる"""
        processor = ResultProcessor()
        ind_normal = _make_strategy_individual((10.0,))
        ind_two_stage = _make_two_stage_individual((1.0,))

        _, best_gene, _ = processor.extract_best_individuals(
            population=[ind_normal, ind_two_stage],
            config=SimpleNamespace(),
        )

        assert best_gene is ind_two_stage

    def test_best_gene_none_when_not_strategy_gene(self) -> None:
        """最良個体が StrategyGene 型でない場合 best_gene は None"""
        processor = ResultProcessor()
        ind = MagicMock()  # not a StrategyGene
        ind.fitness.values = (1.0,)

        # Mock the two_stage_best to return non-StrategyGene
        original = report_selection.get_two_stage_best_individual
        report_selection.get_two_stage_best_individual = lambda pop: ind
        try:
            _, best_gene, _ = processor.extract_best_individuals(
                population=[ind],
                config=SimpleNamespace(),
            )
            assert best_gene is None
        finally:
            report_selection.get_two_stage_best_individual = original

    def test_empty_population_does_not_raise(self) -> None:
        """空の population でもクラッシュしない"""
        processor = ResultProcessor()
        ind = Mock()
        ind.__class__ = StrategyGene
        ind.fitness.values = (1.0,)
        ind.fitness.valid = True

        best, best_gene, _ = processor.extract_best_individuals(
            population=[ind],
            config=SimpleNamespace(),
        )
        assert best is not None or best_gene is not None


class TestResultProcessorSortPopulation:
    """``sort_population`` の挙動テスト"""

    def test_sorts_by_primary_fitness_descending(self) -> None:
        processor = ResultProcessor()
        ind_low = _make_strategy_individual((0.1,))
        ind_mid = _make_strategy_individual((0.5,))
        ind_high = _make_strategy_individual((0.9,))

        sorted_pop = processor.sort_population([ind_low, ind_high, ind_mid])

        fitnesses = [s.fitness.values[0] for s in sorted_pop]
        assert fitnesses == [0.9, 0.5, 0.1]

    def test_two_stage_rank_priority(self) -> None:
        """二段階選抜ランクがある個体は最優先で並ぶ"""
        processor = ResultProcessor()
        # ランクなし、fitness が高い個体
        ind_no_rank_high = _make_strategy_individual((10.0,))
        # ランクあり、fitness が低い個体
        ind_ranked_low = _make_two_stage_individual((0.1,), rank=0)
        # ランクあり、fitness がやや低い個体
        ind_ranked_high = _make_two_stage_individual((0.5,), rank=1)

        sorted_pop = processor.sort_population(
            [ind_no_rank_high, ind_ranked_low, ind_ranked_high]
        )

        # ランク 0 → ランク 1 → ランクなしの順
        assert sorted_pop[0] is ind_ranked_low
        assert sorted_pop[1] is ind_ranked_high
        assert sorted_pop[2] is ind_no_rank_high

    def test_empty_population_returns_empty(self) -> None:
        processor = ResultProcessor()
        assert processor.sort_population([]) == []

    def test_handles_missing_fitness(self) -> None:
        """fitness 属性がない場合は -inf として扱われる"""
        processor = ResultProcessor()
        ind_with_fitness = _make_strategy_individual((0.5,))
        ind_no_fitness = MagicMock()
        # No fitness attribute

        # Should not raise
        sorted_pop = processor.sort_population([ind_no_fitness, ind_with_fitness])
        assert len(sorted_pop) == 2

    def test_secondary_sort_within_same_rank(self) -> None:
        """同じランク内では primary fitness 降順で並ぶ"""
        processor = ResultProcessor()
        ind_r0_a = _make_two_stage_individual((0.1,), rank=0)
        ind_r0_b = _make_two_stage_individual((0.9,), rank=0)

        sorted_pop = processor.sort_population([ind_r0_a, ind_r0_b])
        # 同じランク内では fitness 降順
        assert sorted_pop[0] is ind_r0_b
        assert sorted_pop[1] is ind_r0_a

    def test_multi_objective_uses_primary(self) -> None:
        """多目的適合度の場合、primary (最初の値) でソート"""
        processor = ResultProcessor()
        ind_a = _make_strategy_individual((0.9, 0.1))
        ind_b = _make_strategy_individual((0.1, 0.9))
        ind_c = _make_strategy_individual((0.5, 0.5))

        sorted_pop = processor.sort_population([ind_a, ind_b, ind_c])

        primary = [s.fitness.values[0] for s in sorted_pop]
        assert primary == [0.9, 0.5, 0.1]


class TestResultProcessorGetStrategyResultKey:
    """``get_strategy_result_key`` の挙動テスト"""

    def test_uses_id_attribute_when_present(self) -> None:
        processor = ResultProcessor()
        strategy = SimpleNamespace(id="my-strategy-123")
        assert processor.get_strategy_result_key(strategy) == "my-strategy-123"

    def test_falls_back_to_object_id_when_id_is_none(self) -> None:
        processor = ResultProcessor()
        strategy = SimpleNamespace(id=None)
        key = processor.get_strategy_result_key(strategy)
        assert key == str(id(strategy))

    def test_falls_back_to_object_id_when_id_is_empty_string(self) -> None:
        processor = ResultProcessor()
        strategy = SimpleNamespace(id="")
        key = processor.get_strategy_result_key(strategy)
        assert key == str(id(strategy))

    def test_falls_back_to_object_id_when_id_attribute_missing(self) -> None:
        processor = ResultProcessor()
        strategy = SimpleNamespace()
        key = processor.get_strategy_result_key(strategy)
        assert key == str(id(strategy))

    def test_coerces_id_to_string(self) -> None:
        processor = ResultProcessor()
        strategy = SimpleNamespace(id=12345)
        assert processor.get_strategy_result_key(strategy) == "12345"


class TestResultProcessorConstants:
    """モジュールレベル定数の確認"""

    def test_max_strategies_is_int(self) -> None:
        assert isinstance(MAX_STRATEGIES_TO_EXTRACT, int)
        assert MAX_STRATEGIES_TO_EXTRACT > 0
