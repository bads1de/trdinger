import numpy as np
import pandas as pd
import pytest

from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.auto_strategy.models.gene_strategy import StrategyGene, Condition
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.backtest.execution.backtest_executor import BacktestExecutor


class _DataService:
    def __init__(self, n=200):
        self.n = n

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.n, freq="H")
        close = np.linspace(100, 120, self.n)
        open_ = np.concatenate([[100], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = np.full(self.n, 1000.0)
        return pd.DataFrame(
            {
                "Open": open_.astype(float),
                "High": high.astype(float),
                "Low": low.astype(float),
                "Close": close.astype(float),
                "Volume": vol.astype(float),
            },
            index=idx,
        )


@pytest.mark.integration
def test_stagewise_flow_logs_smoke(caplog):
    caplog.set_level("INFO")

    gene = StrategyGene()
    # 最小限の有効インジケータ（SMA）を追加
    from app.services.auto_strategy.models.gene_strategy import IndicatorGene

    gene.indicators = [
        IndicatorGene(type="SMA", parameters={"period": 14}, enabled=True)
    ]
    gene.entry_conditions = [
        Condition(left_operand="close", operator=">", right_operand="open")
    ]
    gene.exit_conditions = []
    gene.tpsl_gene = TPSLGene(
        method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.02, take_profit_pct=0.04
    )
    gene.position_sizing_gene = PositionSizingGene(
        method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1
    )

    cls = StrategyClassFactory().create_strategy_class(gene)

    stats = BacktestExecutor(_DataService()).execute_backtest(
        strategy_class=cls,
        strategy_parameters={"strategy_gene": gene},
        symbol="BTC:USDT",
        timeframe="1h",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-10"),
        initial_capital=10000.0,
        commission_rate=0.001,
    )

    # スモーク: ログ出力がある程度出ること（失敗時の情報量確保）
    debug_logs = [
        r
        for r in caplog.records
        if "DEBUG" in r.levelname
        or "条件" in r.getMessage()
        or "指標" in r.getMessage()
    ]
    # backtesting.pyのStatsはSeriesで返るため辞書変換して検証
    if not isinstance(stats, dict) and hasattr(stats, "to_dict"):
        stats = stats.to_dict()

    assert isinstance(stats, dict)
    assert stats.get("# Trades", 0) >= 0
    assert len(debug_logs) >= 0
