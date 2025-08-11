from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.auto_strategy.models.gene_strategy import (
    StrategyGene,
    Condition as ConditionGene,
)
from app.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from database.models import Base
from database.repositories.backtest_result_repository import BacktestResultRepository


class _SimpleData:
    def __init__(self, bars=200):
        self.bars = bars

    def get_data_for_backtest(self, symbol, timeframe, start_date, end_date):
        idx = pd.date_range(start=start_date, periods=self.bars, freq="1H")
        # 単調上昇トレンド
        close = np.linspace(100, 200, self.bars)
        open_ = np.concatenate([[100], close[:-1]])
        high = np.maximum(open_, close) * 1.001
        low = np.minimum(open_, close) * 0.999
        vol = np.full(self.bars, 1000)
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )


def test_minimal_entry_executes_with_tp_sl_and_size():
    # DBはメモリ
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    # 最小戦略: 条件は常にTrueに近い（Close > Open）
    gene = StrategyGene()
    # 最小限の有効インジケータ（SMA）を追加
    from app.services.auto_strategy.models.gene_strategy import IndicatorGene

    gene.indicators = [
        IndicatorGene(type="SMA", parameters={"length": 5}, enabled=True)
    ]
    gene.entry_conditions = [
        ConditionGene(left_operand="close", operator=">", right_operand="open")
    ]
    gene.exit_conditions = []  # TP/SL運用なので空

    # TP/SL設定（固定比率）
    gene.tpsl_gene = TPSLGene(
        method=TPSLMethod.FIXED_PERCENTAGE, stop_loss_pct=0.02, take_profit_pct=0.04
    )

    # ポジションサイジング（固定比率10%）
    ps = PositionSizingGene(method=PositionSizingMethod.FIXED_RATIO, fixed_ratio=0.1)
    gene.position_sizing_gene = ps

    strategy_class = StrategyFactory().create_strategy_class(gene)

    start = datetime(2024, 1, 1)
    end = start + timedelta(days=20)
    data_service = _SimpleData(bars=200)

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

    result = BacktestResultConverter().convert_backtest_results(
        stats=stats,
        strategy_name="minimal",
        symbol="BTC:USDT",
        timeframe="1h",
        initial_capital=10000.0,
        start_date=start,
        end_date=end,
        config_json={},
    )

    # 保存
    db = Session()
    try:
        saved = BacktestResultRepository(db).save_backtest_result(result)
    finally:
        db.close()

    # 1件以上のトレードが成立していること
    assert saved.get("total_trades", 0) > 0
    assert isinstance(saved.get("trade_history", []), list)
