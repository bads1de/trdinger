import random
import numpy as np
from datetime import datetime, timedelta

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.auto_strategy.utils.metrics import (
    score_strategy_quality,
    passes_quality_threshold,
)


def test_quality_selection_thresholds_and_scoring():
    rng = random.Random(123)
    np.random.seed(123)

    configs = [
        GAConfig(
            indicator_mode="technical_only",
            max_indicators=4,
            min_indicators=2,
            max_conditions=5,
            min_conditions=2,
        )
        for _ in range(5)
    ]

    start = datetime(2024, 1, 1)
    end = start + timedelta(days=30)

    # 合成データはBacktestExecutor内のフェイクサービスに依存（本テストはスモーク）
    # ここでは品質関数がエラー無く動作し、閾値判定がboolを返すことを確認
    # シンプルな上昇トレンドの合成データサービス
    import pandas as pd

    class _DS:
        def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
            import pandas as _pd
            import numpy as _np

            idx = _pd.date_range(start=start_date, end=end_date, freq="1H")
            base = _np.linspace(100, 120, len(idx))
            open_ = _np.concatenate([[base[0]], base[:-1]])
            high = _np.maximum(open_, base) * 1.001
            low = _np.minimum(open_, base) * 0.999
            vol = _np.full(len(idx), 1000)
            return _pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": base, "Volume": vol},
                index=idx,
            )

    executor = BacktestExecutor(_DS())
    factory = StrategyFactory()

    for cfg in configs:
        gene = RandomGeneGenerator(
            cfg, enable_smart_generation=True
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
        if hasattr(stats, "to_dict"):
            stats = stats.to_dict()

        score = score_strategy_quality(stats)
        assert isinstance(score, float)
        ok = passes_quality_threshold(stats)
        assert isinstance(ok, bool)
