import os
from datetime import datetime, timedelta, timezone
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from database.repositories.ohlcv_repository import OHLCVRepository


DB_PATH = r"C:\Users\buti3\trading\backend\trdinger.db"


class _DBDataService:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)

    def get_data_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        db = self.Session()
        try:
            repo = OHLCVRepository(db)
            df = repo.get_ohlcv_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            # backtesting互換のカラム名（先頭大文字）に変換済み: get_ohlcv_dataframe returns lower-case; adjust
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            # タイムスタンプをindexに
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        finally:
            db.close()


def _discover_symbols(engine, limit=5):
    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT DISTINCT symbol FROM ohlcv_data LIMIT :lim"), {"lim": limit}
        )
        return [row[0] for row in res.fetchall()]


def _get_latest_timestamp(engine, symbol, timeframe):
    with engine.connect() as conn:
        res = conn.execute(
            text(
                """
            SELECT MAX(timestamp) FROM ohlcv_data WHERE symbol=:sym AND timeframe=:tf
            """
            ),
            {"sym": symbol, "tf": timeframe},
        )
        row = res.fetchone()
        val = row[0]
        # SQLiteは文字列で返る場合がある
        if isinstance(val, str):
            try:
                return pd.to_datetime(val, utc=True).to_pydatetime()
            except Exception:
                return datetime.fromisoformat(val)
        return val


@pytest.mark.slow
def test_db_backtest_one_year_smoke():
    assert os.path.exists(DB_PATH), f"DB not found at {DB_PATH}"

    engine = create_engine(f"sqlite:///{DB_PATH}")
    data_service = _DBDataService(engine)
    executor = BacktestExecutor(data_service)
    strategy_factory = StrategyClassFactory()

    # シンボル探索（優先順位: BTC/USDT -> BTC:USDT -> BTCUSDT -> DB先頭）
    symbols = _discover_symbols(engine, limit=10)
    candidates = [s for s in ["BTC/USDT", "BTC:USDT", "BTCUSDT"] if s in symbols]
    symbol = candidates[0] if candidates else (symbols[0] if symbols else None)
    assert symbol, "No symbols found in DB"

    timeframes = ["15m", "30m", "1h", "4h", "1d"]

    per_tf = 3  # 軽量化のため各tf 3戦略
    zero_trade_count = 0
    total = 0

    for tf in timeframes:
        latest = _get_latest_timestamp(engine, symbol, tf)
        if latest is None:
            # データがない時間軸はスキップ
            continue
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        start = latest - timedelta(days=365)

        # データが少なすぎる場合スキップ（例: 200本未満）
        df = data_service.get_data_for_backtest(symbol, tf, start, latest)
        if df.empty or len(df) < 200:
            continue

        for i in range(per_tf):
            seed = hash((tf, i)) % (2**32)
            np.random.seed(seed)

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
            strategy_class = strategy_factory.create_strategy_class(gene)

            stats = executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters={"strategy_gene": gene},
                symbol=symbol,
                timeframe=tf,
                start_date=start,
                end_date=latest,
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            # backtesting.StatsはSeries想定
            if hasattr(stats, "to_dict"):
                stats = stats.to_dict()

            trades = stats.get("# Trades", 0)
            if trades == 0:
                zero_trade_count += 1
            total += 1

    # 少なくとも1つは1年分のデータでテスト実施されること
    assert total > 0, "No timeframe had sufficient data for 1-year backtest"
    # 備考: 現状はゼロ決済トレードが多く観測される（後述のレポート参照）
