# ruff: noqa: E402
"""
制限トーナメント置換 (RTR) のテスト
"""

import os
import random
import sys
from types import SimpleNamespace

import pytest

# プロジェクトルートをパスに追加
# tests/services/auto_strategy/core/engine/ から backend/ まで5階層上
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
sys.path.insert(0, backend_dir)

from deap import base, creator  # noqa: E402

from app.services.auto_strategy.config.ga_config import (  # noqa: E402
    GAConfig,
    SurvivalSelectionConfig,
)
from app.services.auto_strategy.core.engine.evolution_runner import (  # noqa: E402
    EvolutionRunner,
)
from app.services.auto_strategy.core.engine.restricted_tournament import (  # noqa: E402
    indicator_set_elements,
    jaccard_distance,
    restricted_tournament_replace,
)

creator.create("FitnessRTRTest", base.Fitness, weights=(1.0,))


def make_gene(indicator_specs, fitness_value=None):
    """指標構成と fitness を持つ最小の戦略個体を作る。"""
    gene = SimpleNamespace(
        indicators=[
            SimpleNamespace(
                type=type_, timeframe=timeframe, enabled=True, parameters={}
            )
            for type_, timeframe in indicator_specs
        ],
        long_entry_conditions=[],
        short_entry_conditions=[],
        long_exit_conditions=[],
        short_exit_conditions=[],
        stateful_conditions=[],
        tool_genes=[],
        risk_management={},
        tpsl_gene=None,
        long_tpsl_gene=None,
        short_tpsl_gene=None,
        position_sizing_gene=None,
        entry_gene=None,
        long_entry_gene=None,
        short_entry_gene=None,
        exit_gene=None,
    )
    if fitness_value is not None:
        gene.fitness = creator.FitnessRTRTest()
        gene.fitness.values = (fitness_value,)
    return gene


class TestIndicatorSetElements:
    def test_extracts_enabled_indicator_type_and_timeframe(self) -> None:
        gene = make_gene([("SMA", "1d"), ("RSI", None)])
        assert indicator_set_elements(gene) == frozenset({("SMA", "1d"), ("RSI", "")})

    def test_disabled_indicators_are_excluded(self) -> None:
        gene = make_gene([("SMA", None)])
        gene.indicators[0].enabled = False
        assert indicator_set_elements(gene) == frozenset()

    def test_missing_indicators_attribute_yields_empty(self) -> None:
        assert indicator_set_elements(SimpleNamespace()) == frozenset()


class TestJaccardDistance:
    def test_identical_sets_have_zero_distance(self) -> None:
        a = frozenset({("SMA", ""), ("RSI", "")})
        assert jaccard_distance(a, a) == 0.0

    def test_disjoint_sets_have_unit_distance(self) -> None:
        a = frozenset({("SMA", "")})
        b = frozenset({("RSI", "")})
        assert jaccard_distance(a, b) == 1.0

    def test_both_empty_sets_have_zero_distance(self) -> None:
        assert jaccard_distance(frozenset(), frozenset()) == 0.0

    def test_empty_versus_non_empty_is_unit_distance(self) -> None:
        assert jaccard_distance(frozenset(), frozenset({("SMA", "")})) == 1.0

    def test_partial_overlap_distance(self) -> None:
        base_set = frozenset({(f"IND{i}", "") for i in range(5)})
        extended = base_set | {("NEW", "")}
        # 5/6 共有 → 距離 1 - 5/6 = 1/6
        assert jaccard_distance(base_set, extended) == pytest.approx(1.0 / 6.0)


class TestRestrictedTournamentReplace:
    def setup_method(self) -> None:
        self.rng = random.Random(42)
        self.parent_abc = make_gene([("A", ""), ("B", ""), ("C", "")], 1.0)
        self.parent_xyz = make_gene([("X", ""), ("Y", ""), ("Z", "")], 1.0)
        self.parents = [self.parent_abc, self.parent_xyz]

    def test_dominating_child_replaces_closest_structural_neighbor(self) -> None:
        child = make_gene([("A", ""), ("B", ""), ("D", "")], 2.0)
        survivors = restricted_tournament_replace(
            self.parents, [child], crowding_factor=10, rng=self.rng
        )
        assert child in survivors
        assert self.parent_abc not in survivors
        assert self.parent_xyz in survivors

    def test_dominated_child_is_rejected(self) -> None:
        child = make_gene([("A", ""), ("B", ""), ("D", "")], 0.5)
        survivors = restricted_tournament_replace(
            self.parents, [child], crowding_factor=10, rng=self.rng
        )
        assert survivors == self.parents

    def test_structural_clone_is_rejected_when_nondominated(self) -> None:
        clone = make_gene([("A", ""), ("B", ""), ("C", "")], 1.0)
        survivors = restricted_tournament_replace(
            self.parents, [clone], crowding_factor=10, rng=self.rng
        )
        assert survivors == self.parents

    def test_nondominated_structurally_distinct_child_is_appended(self) -> None:
        child = make_gene([("Q", ""), ("R", "")], 1.0)
        survivors = restricted_tournament_replace(
            self.parents, [child], crowding_factor=10, rng=self.rng
        )
        assert len(survivors) == 3
        assert child in survivors
        assert self.parent_abc in survivors and self.parent_xyz in survivors

    def test_invalid_fitness_child_is_skipped(self) -> None:
        child = make_gene([("A", ""), ("B", ""), ("D", "")])
        survivors = restricted_tournament_replace(
            self.parents, [child], crowding_factor=10, rng=self.rng
        )
        assert survivors == self.parents

    def test_empty_parents_accept_all_children(self) -> None:
        child = make_gene([("A", "")], 1.0)
        survivors = restricted_tournament_replace(
            [], [child], crowding_factor=4, rng=self.rng
        )
        assert survivors == [child]

    def test_competition_targets_nearest_niche_not_weakest_parent(self) -> None:
        # P1: 構造 {A} 最強 / P2: 構造 {A,B} 最弱 / P3: 構造 {Z}
        p1 = make_gene([("A", "")], 5.0)
        p2 = make_gene([("A", ""), ("B", "")], 1.0)
        p3 = make_gene([("Z", "")], 1.0)
        child = make_gene([("A", ""), ("B", "")], 2.0)
        survivors = restricted_tournament_replace(
            [p1, p2, p3], [child], crowding_factor=3, rng=self.rng
        )
        # 最寄りは P2（距離 1/3）で、child は P2 を支配する → P2 のみ置換
        assert p1 in survivors and p3 in survivors
        assert p2 not in survivors and child in survivors

    def test_crowding_factor_is_clamped_to_one(self) -> None:
        child = make_gene([("A", ""), ("B", ""), ("D", "")], 2.0)
        survivors = restricted_tournament_replace(
            self.parents, [child], crowding_factor=0, rng=self.rng
        )
        assert child in survivors

    def test_does_not_mutate_parent_list(self) -> None:
        child = make_gene([("A", ""), ("B", ""), ("D", "")], 2.0)
        original = list(self.parents)
        restricted_tournament_replace(
            self.parents, [child], crowding_factor=10, rng=self.rng
        )
        assert self.parents == original


class TestEvolutionRunnerRestrictedTournament:
    def _make_runner(self):
        runner = EvolutionRunner(toolbox=SimpleNamespace(), stats=SimpleNamespace())
        runner._two_stage_selection = SimpleNamespace(
            apply_two_stage_selection=lambda pool, size, cfg: list(pool[:size])
        )
        return runner

    def _rtr_config(self, enabled: bool) -> SurvivalSelectionConfig:
        return SurvivalSelectionConfig(
            enable_restricted_tournament=enabled,
            restricted_tournament_crowding_factor=10,
        )

    def test_rtr_prunes_pool_before_two_stage_selection(self) -> None:
        runner = self._make_runner()
        parents = [make_gene([("A", "")], 1.0), make_gene([("X", "")], 1.0)]
        clones = [
            make_gene([("A", "")], 1.0),
            make_gene([("X", "")], 1.0),
        ]
        config = SimpleNamespace(
            fitness_sharing=SimpleNamespace(enable_fitness_sharing=False),
            survival_selection_config=self._rtr_config(True),
        )
        selected = runner._apply_shared_selection(
            parents + clones, 2, config, offspring=clones, parents=parents
        )
        # クローンは棄却され、プールは親2体に絞られる
        assert sorted(str(id(ind)) for ind in selected) == sorted(
            str(id(ind)) for ind in parents
        )

    def test_rtr_disabled_passes_full_pool(self) -> None:
        runner = self._make_runner()
        parents = [make_gene([("A", "")], 1.0)]
        clones = [make_gene([("A", "")], 1.0)]
        config = SimpleNamespace(
            fitness_sharing=SimpleNamespace(enable_fitness_sharing=False),
            survival_selection_config=self._rtr_config(False),
        )
        pool = parents + clones
        selected = runner._apply_shared_selection(
            pool, 1, config, offspring=clones, parents=parents
        )
        assert len(selected) == 1

    def test_non_survival_config_object_disables_rtr(self) -> None:
        runner = self._make_runner()
        parents = [make_gene([("A", "")], 1.0)]
        clones = [make_gene([("A", "")], 1.0)]
        config = SimpleNamespace(
            fitness_sharing=SimpleNamespace(enable_fitness_sharing=False),
            survival_selection_config=SimpleNamespace(
                enable_restricted_tournament=True
            ),
        )
        pool = parents + clones
        selected = runner._apply_shared_selection(
            pool, 2, config, offspring=clones, parents=parents
        )
        # SurvivalSelectionConfig 型でなければ RTR は適用しない
        assert len(selected) == 2


class TestSurvivalSelectionConfigWiring:
    def test_ga_config_accepts_nested_dict(self) -> None:
        config = GAConfig(
            survival_selection_config={
                "enable_restricted_tournament": True,
                "restricted_tournament_crowding_factor": 5,
            }
        )
        assert isinstance(config.survival_selection_config, SurvivalSelectionConfig)
        assert config.survival_selection_config.enable_restricted_tournament is True
        assert (
            config.survival_selection_config.restricted_tournament_crowding_factor == 5
        )

    def test_default_is_disabled(self) -> None:
        config = GAConfig()
        assert config.survival_selection_config.enable_restricted_tournament is False
        assert (
            config.survival_selection_config.restricted_tournament_crowding_factor == 8
        )

    def test_roundtrip_through_from_dict(self) -> None:
        config = GAConfig(
            survival_selection_config={"enable_restricted_tournament": True}
        )
        restored = GAConfig.from_dict(config.to_dict())
        assert isinstance(restored.survival_selection_config, SurvivalSelectionConfig)
        assert restored.survival_selection_config.enable_restricted_tournament is True
