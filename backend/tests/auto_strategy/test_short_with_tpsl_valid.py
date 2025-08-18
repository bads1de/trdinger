from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, Condition as ConditionGene, IndicatorGene
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingGene, PositionSizingMethod
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class _DowntrendData:
    def __init__(self, bars=300):
        self.bars = bars

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.bars, freq="1H")
        base = np.linspace(200, 100, self.bars)  # 緩やかな下落
        noise = np.random.normal(0, 0.1, size=self.bars)
        close = base + noise
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = np.full(self.bars, 1000)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_short_entry_with_tpsl_executes():
    # シンプルにSMA下抜けでショートし、TP/SL付きで約定できること
    gene = StrategyGene()
    gene.indicators = [IndicatorGene(type="SMA", parameters={"period": 10}, enabled=True)]
    gene.short_entry_conditions = [ConditionGene(left_operand="close", operator="<", right_operand="SMA")]
    gene.long_entry_conditions = []

    gene.tpsl_gene = TPSLGene(method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.02, take_profit_pct=0.03)

    # 固定比率で十分な枚数を持つ
    gene.position_sizing_gene = PositionSizingGene(method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.2)

    strategy_class = StrategyClassFactory().create_strategy_class(gene)

    data_service = _DowntrendData(bars=300)
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=20)

    stats = BacktestExecutor(data_service).execute_backtest(
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

    assert stats.get("# Trades", 0) > 0
    # backtestingのStatsはショート件数キーがない場合もあるが、ショート条件のみなのでプラスの期待

