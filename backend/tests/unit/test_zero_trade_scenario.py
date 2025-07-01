import pytest
from unittest.mock import MagicMock, patch
from app.core.services.auto_strategy.services.auto_strategy_service import (
    AutoStrategyService,
)
from app.core.services.auto_strategy.models.strategy_gene import (
    StrategyGene,
    Condition,
    IndicatorGene,
)


# テスト用の設定
@pytest.fixture
def auto_strategy_service():
    """AutoStrategyServiceのモックインスタンスを生成"""
    with (
        patch(
            "app.core.services.auto_strategy.services.auto_strategy_service.SessionLocal"
        ),
        patch(
            "app.core.services.auto_strategy.services.auto_strategy_service.BacktestService"
        ) as MockBacktestService,
        patch(
            "app.core.services.auto_strategy.services.auto_strategy_service.BacktestDataService"
        ),
    ):

        # バックテストサービスのモック設定
        mock_backtest_service = MockBacktestService.return_value
        # run_backtestが返す値を設定
        mock_backtest_service.run_backtest.return_value = {
            "performance_metrics": {
                "total_return": 99.43,
                "sharpe_ratio": 0.96,
                "max_drawdown": -29.48,
                "win_rate": 0.0,
                "total_trades": 0,
            },
            "trade_history": [],
            "equity_curve": [],
        }

        service = AutoStrategyService()
        service.backtest_service = mock_backtest_service
        return service


@pytest.fixture
def zero_trade_gene():
    """取引を発生させない単純なStrategyGeneを生成"""
    return StrategyGene(
        id="zero_trade_test",
        indicators=[IndicatorGene(type="SMA", parameters={"length": 14}, enabled=True)],
        entry_conditions=[
            Condition(
                left_operand={"type": "indicator", "value": "SMA_14"},
                operator="GREATER_THAN",
                right_operand={
                    "type": "literal",
                    "value": 999999,
                },  # 常にFalseになる条件
            )
        ],
        exit_conditions=[
            Condition(
                left_operand={"type": "indicator", "value": "SMA_14"},
                operator="LESS_THAN",
                right_operand={"type": "literal", "value": 0},  # 常にFalseになる条件
            )
        ],
    )


def test_zero_trade_scenario_handling(auto_strategy_service, zero_trade_gene):
    """
    取引数が0の場合に、FitnessCalculatorがパフォーマンス指標を正しく0に設定するかをテスト
    """
    # GAConfigのモック
    mock_ga_config = MagicMock()
    mock_ga_config.fitness_constraints = {"min_trades": 1}

    # バックテスト結果（取引数0）
    backtest_result_zero_trades = {
        "performance_metrics": {
            "total_return": 99.43,
            "sharpe_ratio": 0.96,
            "max_drawdown": -29.48,
            "win_rate": 0.0,
            "total_trades": 0,
        },
        "data": MagicMock(),
    }

    # FitnessCalculatorのインスタンスを取得
    fitness_calculator = (
        auto_strategy_service.ga_engine.fitness_calculator
        if auto_strategy_service.ga_engine
        else None
    )
    if not fitness_calculator:
        # もしga_engineがNoneなら、テスト用に直接インスタンス化
        from app.core.services.auto_strategy.engines.fitness_calculator import (
            FitnessCalculator,
        )

        fitness_calculator = FitnessCalculator(
            auto_strategy_service.backtest_service,
            auto_strategy_service.strategy_factory,
        )

    # calculate_fitnessを呼び出し
    fitness = fitness_calculator.calculate_fitness(
        backtest_result=backtest_result_zero_trades,
        config=mock_ga_config,
        strategy_gene=zero_trade_gene,
    )

    # 検証：フィットネスが0.0であること
    # 取引数が0の場合、制約違反で0.0が返されるはず
    assert (
        fitness == 0.0
    ), f"取引数0の場合、フィットネスは0.0になるべきですが、{fitness}でした。"
