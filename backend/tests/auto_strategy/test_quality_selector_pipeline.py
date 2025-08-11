from datetime import datetime, timedelta
import random
import numpy as np

from app.services.auto_strategy.utils.selector import filter_and_rank_strategies
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory


def test_quality_selector_filters_and_ranks():
    rng = random.Random(777)
    np.random.seed(777)

    start = datetime(2024, 1, 1)
    end = start + timedelta(days=15)

    # 合成データサービス（単純レンジ）
    import pandas as pd

    class _DS:
        def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
            idx = pd.date_range(start=start_date, end=end_date, freq="1H")
            base = 120 + np.sin(np.linspace(0, 20, len(idx)))
            open_ = np.concatenate([[base[0]], base[:-1]])
            high = np.maximum(open_, base) * 1.001
            low = np.minimum(open_, base) * 0.999
            vol = np.full(len(idx), 1000)
            return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": base, "Volume": vol}, index=idx)

    executor = BacktestExecutor(_DS())
    factory = StrategyFactory()

    stats_list = []
    for seed in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]:
        random.seed(seed)
        np.random.seed(seed)
        cfg = GAConfig(indicator_mode="technical_only", max_indicators=4, min_indicators=2, max_conditions=5, min_conditions=2)
        gene = RandomGeneGenerator(cfg, enable_smart_generation=True).generate_random_gene()
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
        stats_list.append(stats)

    selected, rejected = filter_and_rank_strategies(stats_list, min_trades=1, require_quality_threshold=True)

    assert isinstance(selected, list) and isinstance(rejected, list)
    # 成立率>=60%ならselectedが存在する可能性が高い（データ生成の性質に依存）
    assert len(selected) + len(rejected) == len(stats_list)
    if selected:
        # スコアが降順になっていること
        from app.services.auto_strategy.utils.metrics import score_strategy_quality
        scores = [score_strategy_quality(s) for s in selected]
        assert scores == sorted(scores, reverse=True)

