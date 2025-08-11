import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.auto_strategy.factories.strategy_factory import StrategyFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.conversion.backtest_result_converter import BacktestResultConverter
from database.models import Base
from database.repositories.backtest_result_repository import BacktestResultRepository


class _SyntheticBacktestDataService:
    """バックテスト用の合成データを提供する軽量スタブ"""

    def __init__(self, n=300, seed=0):
        self.n = n
        self.seed = seed

    def get_data_for_backtest(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        idx = pd.date_range(start=start_date, end=end_date, periods=self.n)
        base = 50000.0
        # ランダムウォークにボラティリティを加味
        returns = rng.normal(0, 0.01, size=self.n)
        prices = [base]
        for r in returns[1:]:
            prices.append(max(1000.0, prices[-1] * (1 + r)))
        close = np.array(prices)
        high = close * (1 + np.abs(rng.normal(0, 0.004, size=self.n)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, size=self.n)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        vol = rng.integers(100, 10000, size=self.n)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)
        return df


def test_technical_only_backtest_and_persistence_runs_60_times():
    # インメモリSQLiteを用意（本番DBは触らない）
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    converter = BacktestResultConverter()
    strategy_factory = StrategyFactory()

    # 60回実行
    for i, seed in enumerate(range(2000, 2060), start=1):
        random.seed(seed)
        np.random.seed(seed)

        # テクニカルオンリー構成
        ga_cfg = GAConfig(
            indicator_mode="technical_only",
            max_indicators=3,
            min_indicators=2,
            max_conditions=4,
            min_conditions=2,
        )

        # 遺伝子生成
        gene = RandomGeneGenerator(config=ga_cfg, enable_smart_generation=True).generate_random_gene()

        # 戦略クラス生成
        strategy_class = strategy_factory.create_strategy_class(gene)

        # 合成データサービス/実行器
        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(hours=300)
        data_service = _SyntheticBacktestDataService(n=300, seed=seed)
        executor = BacktestExecutor(data_service)

        # バックテスト実行
        stats = executor.execute_backtest(
            strategy_class=strategy_class,
            strategy_parameters={},
            symbol="BTC:USDT",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000.0,
            commission_rate=0.001,
        )

        # 結果変換
        config_json = {"strategy_config": {"strategy_gene": "..."}, "commission_rate": 0.001}
        result = converter.convert_backtest_results(
            stats=stats,
            strategy_name="auto_strategy",
            symbol="BTC:USDT",
            timeframe="1h",
            initial_capital=10000.0,
            start_date=start_date,
            end_date=end_date,
            config_json=config_json,
        )

        # 保存
        db = TestingSessionLocal()
        try:
            saved = BacktestResultRepository(db).save_backtest_result(result)
        finally:
            db.close()

        # 基本検証: 必須キーが存在し、数値/配列がシリアライズされている
        assert "performance_metrics" in saved
        pm = saved["performance_metrics"] or {}
        assert "sharpe_ratio" in pm
        assert "total_trades" in pm
        assert "max_drawdown" in pm
        assert "equity_curve" in saved
        assert "trade_history" in saved
        # 少なくともエクイティカーブは存在（保存可能なリスト）
        assert isinstance(saved["equity_curve"], list)
        assert isinstance(saved["trade_history"], list)

