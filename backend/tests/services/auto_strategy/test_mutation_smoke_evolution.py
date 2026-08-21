# ruff: noqa: E402
"""
突然変異変更後のGAスモークテスト

実際の進化ループ（DEAPSetup + EvolutionRunner）を小規模設定で回し、
変異後個体がバリデーションを通ること・fitness が機能することを確認する。
バックテストは行わず、遺伝子構造から決定的な合成 fitness を返す。
"""

import random
import zlib

import pytest
from deap import tools

from app.services.auto_strategy.config.ga_config import GAConfig
from app.services.auto_strategy.core.engine.deap_setup import DEAPSetup
from app.services.auto_strategy.core.engine.evolution_runner import EvolutionRunner
from app.services.auto_strategy.core.engine.ga_utils import create_deap_mutate_wrapper
from app.services.auto_strategy.genes import StrategyGene


def _synthetic_fitness(
    gene: StrategyGene, config: GAConfig | None = None
) -> tuple[float]:
    """遺伝子構造から決定的な合成 fitness を計算する。

    実バックテストの代わりに、指標数・条件数・stateful 条件数・
    サブ遺伝子の有無を組み合わせたスコアを返す。
    """
    score = 0.0
    score += min(len(gene.indicators), 5) * 0.1
    score += (
        min(len(gene.long_entry_conditions) + len(gene.short_entry_conditions), 6)
        * 0.05
    )
    score += min(len(gene.stateful_conditions), 3) * 0.2
    if gene.tpsl_gene is not None:
        score += 0.3
    if gene.position_sizing_gene is not None:
        score += 0.2
    if gene.entry_gene is not None:
        score += 0.1
    if gene.exit_gene is not None:
        score += 0.1
    # 決定的ハッシュで個体ごとの微細な差を付ける（同一構造の個体が並ぶのを防ぐ）。
    # 組み込み hash() は PYTHONHASHSEED によりプロセス毎に変わるため zlib.crc32 を使う
    seed = zlib.crc32(str(sorted(str(i.type) for i in gene.indicators)).encode("utf-8"))
    score += (seed % 100) / 1000.0
    return (score,)


@pytest.fixture
def smoke_config() -> GAConfig:
    config = GAConfig()
    config.population_size = 12
    config.generations = 4
    config.mutation_rate = 0.5  # 変異を強めて新規変異パスを通す
    config.crossover_rate = 0.8
    config.elite_size = 2
    config.max_indicators = 4
    config.min_indicators = 1
    config.enable_multi_timeframe = True
    config.use_seed_strategies = False
    config.fitness_sharing.enable_fitness_sharing = False
    config.survival_selection_config.enable_restricted_tournament = False
    return config


class TestMutationSmokeEvolution:
    """変異変更後の進化ループスモークテスト"""

    def test_evolution_completes_and_individuals_valid(self, smoke_config):
        """小規模進化が完了し、全個体が評価・バリデーションを通ることを確認"""
        random.seed(123)
        deap_setup = DEAPSetup()

        def create_individual() -> StrategyGene:
            from app.services.auto_strategy.generators.random_gene_generator import (
                RandomGeneGenerator,
            )

            generator = RandomGeneGenerator(smoke_config)
            gene = generator.generate_random_gene()
            return deap_setup.Individual(**_gene_params(gene))

        def _gene_params(gene: StrategyGene) -> dict:
            from app.services.auto_strategy.genes.genetic_utils import GeneticUtils

            return GeneticUtils.extract_gene_params(gene)

        deap_setup.setup_deap(
            smoke_config,
            create_individual,
            _synthetic_fitness,  # type: ignore[arg-type]
            _crossover_adapter,
        )

        toolbox = deap_setup.get_toolbox()
        assert toolbox is not None

        individual_class = deap_setup.get_individual_class()
        assert individual_class is not None

        mutate_wrapper = create_deap_mutate_wrapper(
            individual_class, None, smoke_config
        )
        toolbox.register("mutate", mutate_wrapper)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda v: sum(x[0] for x in v) / len(v))
        stats.register("max", lambda v: max(x[0] for x in v))

        population = toolbox.population(n=smoke_config.population_size)

        runner = EvolutionRunner(toolbox, stats, None, population, None, None)
        result_population, logbook = runner.run_evolution(
            population, smoke_config, tools.ParetoFront()
        )

        # 進化が完了している
        assert len(result_population) == smoke_config.population_size
        records = list(logbook)
        assert len(records) == smoke_config.generations

        # 収束挙動: 最終世代の avg/max が初期世代以上であること
        first_avg = records[0]["avg"]
        last_avg = records[-1]["avg"]
        assert last_avg >= first_avg, (
            f"Fitness should not degrade: first_avg={first_avg}, last_avg={last_avg}"
        )
        assert records[-1]["max"] >= records[0]["max"]

        # 全個体が評価済み
        for ind in result_population:
            assert ind.fitness.valid
            values = ind.fitness.values
            assert len(values) == 1
            assert all(isinstance(v, float) for v in values)

        # 全個体がバリデーションを通る（参照切れ条件・不正パラメータがない）
        failures: list[tuple[str, list[str]]] = []
        for ind in result_population:
            is_valid, errors = ind.validate()
            if not is_valid:
                failures.append((ind.id, errors))
        assert not failures, f"Invalid individuals after evolution: {failures}"

        # 条件参照の整合性（dangling reference がない）
        for ind in result_population:
            valid_names = ind.valid_operand_names()
            for cond in (
                ind.long_entry_conditions
                + ind.short_entry_conditions
                + ind.long_exit_conditions
                + ind.short_exit_conditions
            ):
                _assert_condition_operands_resolvable(cond, valid_names)

        deap_setup.cleanup()

    def test_high_mutation_rate_preserves_structure_limits(self, smoke_config):
        """高変異率でも max_indicators / max_conditions 制限が守られることを確認"""
        random.seed(456)
        smoke_config.mutation_rate = 1.0

        from app.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )
        from app.services.auto_strategy.genes.operators.mutation import (
            mutate_strategy_gene,
        )

        generator = RandomGeneGenerator(smoke_config)
        base_gene = generator.generate_random_gene()

        for i in range(30):
            random.seed(i)
            mutated = mutate_strategy_gene(base_gene, smoke_config, mutation_rate=1.0)
            assert len(mutated.indicators) <= smoke_config.max_indicators
            assert len(mutated.indicators) >= smoke_config.min_indicators
            assert len(mutated.long_entry_conditions) <= smoke_config.max_conditions
            assert len(mutated.short_entry_conditions) <= smoke_config.max_conditions
            assert len(mutated.stateful_conditions) <= 3
            # weekend_filter は常に存在
            tool_names = [t.tool_name for t in mutated.tool_genes]
            assert "weekend_filter" in tool_names
            # id は新規採番される
            assert mutated.id != base_gene.id
            # metadata に変異マーカー
            assert mutated.metadata.get("mutated") is True


def _assert_condition_operands_resolvable(cond, valid_names: set[str]) -> None:
    """条件のオペランドが解決可能か再帰的に検証する。"""
    from app.services.auto_strategy.genes.conditions import Condition, ConditionGroup

    if isinstance(cond, ConditionGroup):
        for sub in cond.conditions:
            _assert_condition_operands_resolvable(sub, valid_names)
        return
    if isinstance(cond, Condition):
        for operand in (cond.left_operand, cond.right_operand):
            if not isinstance(operand, str):
                continue
            try:
                float(operand)
                continue
            except ValueError:
                pass
            lowered = operand.lower()
            if lowered in {"close", "open", "high", "low", "volume"}:
                continue
            base = operand
            if base.rsplit("_", 1)[-1].isdigit() and "_" in base:
                base = base.rsplit("_", 1)[0]
            assert operand in valid_names or base in valid_names, (
                f"Dangling operand reference: {operand}"
            )


def _crossover_adapter(parent1, parent2, config=None):
    """DEAP mate 登録用の交叉アダプタ（親と同じクラスで交叉する）。"""
    return type(parent1).crossover(parent1, parent2, config)
