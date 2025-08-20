import types
import pytest

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from tests.common.test_stubs import DummyBacktestService




@pytest.fixture
def engine_fast():
    backtest = DummyBacktestService()
    strategy_factory = StrategyClassFactory()
    config = GAConfig.create_fast()
    gene_gen = RandomGeneGenerator(config)
    engine = GeneticAlgorithmEngine(backtest, strategy_factory, gene_gen)
    return engine


def test_nsga2_uses_eaMuPlusLambda(monkeypatch, engine_fast):
    """_run_nsga2_evolution が deap.algorithms.eaMuPlusLambda を用いることを確認する。
    Context7のDEAPドキュメント（algorithms.eaMuPlusLambda）に基づく。
    """
    called = {"count": 0}

    # IndividualEvaluator を軽量化
    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (
            (1.0,) if not config.enable_multi_objective else (1.0,)
        ),
        raising=True,
    )

    # deap.algorithms.eaMuPlusLambda の呼び出しをフック
    import deap.algorithms as algorithms

    def fake_eaMuPlusLambda(
        pop,
        toolbox,
        mu,
        lambda_,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=False,
    ):
        _ = (
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            stats,
            halloffame,
            verbose,
        )  # 未使用パラメータ
        called["count"] += 1

        # そのまま population, 空logbook 風のオブジェクトを返す
        class Log:
            def __init__(self):
                self.select("gen")

            def select(self, *args, **kwargs):
                return []

        return pop, Log()

    monkeypatch.setattr(algorithms, "eaMuPlusLambda", fake_eaMuPlusLambda, raising=True)

    cfg = GAConfig.create_multi_objective(["total_return"], [1.0])
    # 短時間で回るように
    cfg.population_size = 6
    cfg.generations = 1
    cfg.enable_fitness_sharing = False

    result = engine_fast.run_evolution(cfg, backtest_config={})

    assert called["count"] >= 1, "eaMuPlusLambda が呼ばれていません"
    assert "population" in result and "logbook" in result


def test_selection_wrapper_applies_fitness_sharing(monkeypatch, engine_fast):
    """fitness_sharing 有効時に、選択前に共有が適用されることを確認。
    selection ラッパーが適用されたかを副作用で検証。
    """
    # 軽量評価
    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (1.0,),
        raising=True,
    )

    applied = {"count": 0}

    def fake_apply(pop):
        applied["count"] += 1
        return pop

    # FitnessSharing インスタンスは setup_deap 後に作られるため、run 中に差し替えやすいよう hook を定義
    original_setup_deap = engine_fast.setup_deap

    def wrapped_setup(cfg):
        original_setup_deap(cfg)
        # fitness_sharing をダミーに差し替え
        if engine_fast.fitness_sharing:
            engine_fast.fitness_sharing.apply_fitness_sharing = fake_apply  # type: ignore

    monkeypatch.setattr(engine_fast, "setup_deap", wrapped_setup, raising=True)

    cfg = GAConfig.create_multi_objective(["total_return"], [1.0])
    cfg.population_size = 6
    cfg.generations = 1
    cfg.enable_fitness_sharing = True

    _ = engine_fast.run_evolution(cfg, backtest_config={})

    # 初期化時と世代内で最低1回は呼ばれているはず
    assert applied["count"] >= 1, "フィットネス共有が適用されていません"
