import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.indicators.config import indicator_registry


class _WavyData:
    def __init__(self, bars=600):
        self.bars = bars

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.bars, freq="1H")
        t = np.linspace(0, 6 * np.pi, self.bars)
        base = 150 + 10 * np.sin(t) + 5 * np.sin(3 * t)
        noise = np.random.normal(0, 0.2, size=self.bars)
        close = base + noise
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * 1.002
        low = np.minimum(open_, close) * 0.998
        vol = np.full(self.bars, 1000)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )


def test_diversity_and_success_rate_across_timeframes():
    timeframes = ["15m", "30m", "1h", "4h", "1d"]
    seeds = [101, 202, 303, 404, 505, 606, 707, 808]

    data_service = _WavyData(bars=600)
    executor = BacktestExecutor(data_service)
    factory = StrategyClassFactory()

    overall_categories = set()
    any_short_trade = False

    for tf in timeframes:
        total = 0
        positive = 0

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
            gene = RandomGeneGenerator(
                config=ga_cfg, enable_smart_generation=True
            ).generate_random_gene()

            # 多様性: カテゴリ収集
            for ind in gene.indicators:
                cfg = indicator_registry.get_indicator_config(ind.type)
                if cfg and getattr(cfg, "category", None):
                    overall_categories.add(cfg.category)

            strategy_class = factory.create_strategy_class(gene)

            start = datetime(2024, 1, 1)
            end = start + timedelta(days=40)
            stats = executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters={"strategy_gene": gene},
                symbol="BTC:USDT",
                timeframe=tf,
                start_date=start,
                end_date=end,
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            # 短期売りの存在を堅牢に検出: _tradesのSize<0 を見る
            if hasattr(stats, "_trades") and getattr(stats, "_trades") is not None:
                trades_df = getattr(stats, "_trades")
                if len(trades_df) > 0 and (trades_df["Size"] < 0).any():
                    any_short_trade = True

            if hasattr(stats, "to_dict"):
                stats = stats.to_dict()
            trades = stats.get("# Trades", 0)
            shorts = stats.get("# Trades Short", None)
            if shorts is not None and shorts > 0:
                any_short_trade = True

            total += 1
            positive += 1 if trades > 0 else 0

        success_rate = positive / total if total > 0 else 0.0
        # 各タイムフレームで50%以上成立（保守的な下限、成立性の回帰監視目的）
        assert (
            success_rate >= 0.5
        ), f"tf={tf} success_rate={success_rate:.2%} ({positive}/{total})"

    # 全体で少なくとも3カテゴリ以上カバー
    assert len(overall_categories) >= 3, f"categories={overall_categories}"
    # ショートが一度は成立
    assert any_short_trade, "No short trades observed in the sample runs"
