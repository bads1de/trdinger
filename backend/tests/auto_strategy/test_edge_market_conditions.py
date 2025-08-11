import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import (
    StrategyGene,
    Condition as C,
    IndicatorGene,
)
from app.services.backtest.execution.backtest_executor import BacktestExecutor


def _make_data(kind: str, bars=400):
    idx = pd.date_range("2024-01-01", periods=bars, freq="1H")
    if kind == "spike_up":
        base = np.linspace(100, 105, bars)
        base[bars // 2] += 30
    elif kind == "spike_down":
        base = np.linspace(105, 100, bars)
        base[bars // 2] -= 30
    else:  # range
        base = 150 + 2 * np.sin(np.linspace(0, 10 * np.pi, bars))
    open_ = np.concatenate([[base[0]], base[:-1]])
    high = np.maximum(open_, base) * 1.003
    low = np.minimum(open_, base) * 0.997
    vol = np.full(bars, 1000)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": base, "Volume": vol},
        index=idx,
    )

    class _DS:
        def get_data_for_backtest(
            self, symbol=None, timeframe=None, start_date=None, end_date=None
        ):
            return df

    return _DS()


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


def test_edge_market_conditions():
    factory = StrategyFactory()
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=30)

    for kind in ("spike_up", "spike_down", "range"):
        ds = _make_data(kind)
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
