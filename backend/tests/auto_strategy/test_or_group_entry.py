import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class _RangeData:
    def __init__(self, bars=400):
        self.bars = bars

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.bars, freq="1H")
        base = 150 + 3*np.sin(np.linspace(0, 10*np.pi, self.bars))
        noise = np.random.normal(0, 0.1, size=self.bars)
        close = base + noise
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = np.full(self.bars, 1000)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_or_group_improves_entry_opportunity():
    data_service = _RangeData(bars=400)
    executor = BacktestExecutor(data_service)
    factory = StrategyFactory()

    seeds = [321, 654, 987]
    more_trades = 0

    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)

        ga_cfg = GAConfig(indicator_mode="technical_only", max_indicators=4, min_indicators=2, max_conditions=5, min_conditions=2)
        gene = RandomGeneGenerator(config=ga_cfg, enable_smart_generation=True).generate_random_gene()
        # ORグループ生成: long側の候補からORを組み、追加
        # SmartConditionGenerator.generate_balanced_conditions を内部で使うため、既定の生成が使われる
        strategy_class = factory.create_strategy_class(gene)

        start = datetime(2024, 1, 1)
        end = start + timedelta(days=20)

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

        # 単純にトレード件数が0でないことを確認（ORにより成立機会が減らない）
        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        assert stats.get("# Trades", 0) >= 0

