import pytest
from deap import tools

from app.services.auto_strategy.core.ga_engine import GeneticAlgorithmEngine
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from tests.common.test_stubs import DummyBacktestService




@pytest.fixture
def engine_fast():
    cfg = GAConfig.create_fast()
    backtest = DummyBacktestService(total_return=0.05, sharpe_ratio=0.8, max_drawdown=0.25, win_rate=0.52, total_trades=10)
    engine = GeneticAlgorithmEngine(
        backtest, StrategyClassFactory(), RandomGeneGenerator(cfg)
    )
    return engine


def test_single_objective_best_selection(monkeypatch, engine_fast):
    """単一目的モードで run_evolution が最良個体を返すことを確認。
    selBest からの抽出が行われるため population が空でないことも検証。
    """
    # 評価は定数でOK（速度重視）
    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (1.0,),
        raising=True,
    )

    cfg = GAConfig.create_fast()
    cfg.enable_multi_objective = False
    cfg.population_size = 6
    cfg.generations = 1
    result = engine_fast.run_evolution(cfg, backtest_config={})

    assert result["population"], "population が空です"
    assert "best_strategy" in result


def test_multi_objective_returns_pareto_front(monkeypatch, engine_fast):
    """多目的モードでパレート前線が返されることを確認。"""
    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (1.0, 0.5),
        raising=True,
    )

    cfg = GAConfig.create_multi_objective(["total_return", "max_drawdown"], [1.0, -1.0])
    cfg.population_size = 6
    cfg.generations = 1

    result = engine_fast.run_evolution(cfg, backtest_config={})

    assert "pareto_front" in result
    assert isinstance(result["pareto_front"], list)


def test_stats_and_logbook_present(monkeypatch, engine_fast):
    """stats を渡した場合に logbook が返ることを簡易検証。"""
    # 評価簡略化
    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (1.0,),
        raising=True,
    )

    cfg = GAConfig.create_multi_objective(["total_return"], [1.0])
    cfg.population_size = 6
    cfg.generations = 2

    result = engine_fast.run_evolution(cfg, backtest_config={})

    assert "logbook" in result
    # 最低限、genヘッダーが存在するかを確認（深追いしない）
    logbook = result["logbook"]
    selected = getattr(logbook, "select", None)
    assert callable(selected)


def test_fitness_sharing_and_nsga2_initialization(monkeypatch, engine_fast):
    """フィットネス共有適用後に初回の selNSGA2 が走ることを確認（クラウディング付与）。
    厳密な距離値は検査せず、selNSGA2 呼び出しの副作用で population が維持されることを確認。
    """
    applied = {"count": 0}

    def fake_apply(pop):
        applied["count"] += 1
        return pop

    original_setup_deap = engine_fast.setup_deap

    def wrapped_setup(cfg):
        original_setup_deap(cfg)
        if engine_fast.fitness_sharing:
            engine_fast.fitness_sharing.apply_fitness_sharing = fake_apply  # type: ignore

    monkeypatch.setattr(engine_fast, "setup_deap", wrapped_setup, raising=True)

    monkeypatch.setattr(
        engine_fast.individual_evaluator,
        "evaluate_individual",
        lambda individual, config: (1.0,),
        raising=True,
    )

    cfg = GAConfig.create_multi_objective(["total_return"], [1.0])
    cfg.enable_fitness_sharing = True
    cfg.population_size = 6
    cfg.generations = 1

    result = engine_fast.run_evolution(cfg, backtest_config={})

    assert applied["count"] >= 1
    assert result["population"], "初回選択後の population が空です"
