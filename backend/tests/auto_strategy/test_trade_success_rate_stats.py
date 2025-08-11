import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class _SimpleData:
    def __init__(self, bars=500):
        self.bars = bars

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.bars, freq="1H")
        # 単調上昇トレンド + 小ノイズ
        base = np.linspace(100, 200, self.bars)
        noise = np.random.normal(0, 0.2, size=self.bars)
        close = base + noise
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = np.full(self.bars, 1000)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )


def test_technical_only_success_rate_at_least_70_percent():
    # ランダムseedを複数用意
    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 101]

    data_service = _SimpleData(bars=500)
    executor = BacktestExecutor(data_service)
    factory = StrategyFactory()

    total = 0
    positive = 0

    start = datetime(2024, 1, 1)
    end = start + timedelta(days=20)

    for sd in seeds:
        random.seed(sd)
        np.random.seed(sd)

        ga_cfg = GAConfig(
            indicator_mode="technical_only",
            max_indicators=4,
            min_indicators=2,
            max_conditions=5,
            min_conditions=2,
        )
        gene = RandomGeneGenerator(config=ga_cfg, enable_smart_generation=True).generate_random_gene()
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

        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()
        trades = stats.get("# Trades", 0)

        total += 1
        positive += 1 if trades > 0 else 0

    success_rate = positive / total if total > 0 else 0.0
    # 少なくとも70%以上の戦略で1トレード以上
    assert success_rate >= 0.7, f"success_rate={success_rate:.2%}, positive={positive}/{total}"

