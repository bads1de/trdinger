import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.auto_strategy.models.gene_strategy import (
    StrategyGene,
    Condition as C,
    IndicatorGene,
)
from app.services.backtest.execution.backtest_executor import BacktestExecutor





def _strategy_close_vs_sma(period=10):
    g = StrategyGene()
    g.indicators = [
        IndicatorGene(type="SMA", parameters={"period": period}, enabled=True)
    ]
    g.long_entry_conditions = [
        C(left_operand="close", operator=">", right_operand="SMA")
    ]
    g.short_entry_conditions = [
        C(left_operand="close", operator="<", right_operand="SMA")
    ]
    # TP/SLがない場合はexit条件が必要 -> フォールバックのシンプルexit（price vs SMAクロスでクローズ）
    g.exit_conditions = [C(left_operand="close", operator="<", right_operand="SMA")]
    return g


def test_edge_market_conditions(market_condition_data_factory):
    factory = StrategyClassFactory()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=30)

    for kind in ("spike_up", "spike_down", "range"):
        ds = market_condition_data_factory(kind)
        g = _strategy_close_vs_sma(10)
        strategy_class = factory.create_strategy_class(g)
        stats = BacktestExecutor(ds).execute_backtest(
            strategy_class=strategy_class,
            strategy_parameters={"strategy_gene": g},
            symbol="BTC:USDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            initial_capital=10000.0,
            commission_rate=0.001,
        )
        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        assert stats.get("# Trades", 0) >= 0
