import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add backend to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from tests.common.test_stubs import _SyntheticDataService


def _make_hourly_data(start: datetime, bars: int) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=bars, freq="1h")
    t = np.linspace(0, 20 * np.pi, bars)
    trend = np.linspace(100, 180, bars)
    wavy = 8 * np.sin(t) + 4 * np.sin(3.1 * t)
    noise = np.random.normal(0, 0.8, size=bars)
    close = trend + wavy + noise
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.random.normal(0.0008, 0.0005, size=bars))
    low = np.minimum(open_, close) * (1 - np.random.normal(0.0008, 0.0005, size=bars))
    vol = np.full(bars, 2000.0)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_aggressive_profile_trade_count_ge_100():
    data_service = _SyntheticDataService()
    executor = BacktestExecutor(data_service)
    factory = StrategyClassFactory()

    ga_cfg = GAConfig(
        indicator_mode="technical_only",
        max_indicators=5,
        min_indicators=3,
        max_conditions=6,
        min_conditions=2,
    )

    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1) - timedelta(hours=1)

    max_trades = 0
    # 試行回数を増やして少なくとも1つは3桁を狙う
    for _ in range(25):
        gene = RandomGeneGenerator(
            config=ga_cfg,
            enable_smart_generation=True,
            smart_context={"timeframe": "1h", "threshold_profile": "aggressive"},
        ).generate_random_gene()
        strategy_class = factory.create_strategy_class(gene)
        stats = executor.execute_backtest(
            strategy_class=strategy_class,
            strategy_parameters={"strategy_gene": gene},
            symbol="BTC:USDT",
            timeframe="1h",
            start_date=start,
            end_date=end,
            initial_capital=10000.0,
            commission_rate=0.001,
        )
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        trades = int(stats_dict.get("# Trades", 0))
        max_trades = max(max_trades, trades)
        if max_trades >= 100:
            break

    assert max_trades >= 100, f"最大トレード数が不足: {max_trades} (<100)"

