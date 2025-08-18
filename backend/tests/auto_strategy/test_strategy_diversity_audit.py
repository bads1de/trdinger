import random
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.auto_strategy.generators.random_gene_generator import (
    RandomGeneGenerator,
)
from app.services.auto_strategy.models.gene_tpsl import TPSLMethod
from app.services.auto_strategy.models.gene_position_sizing import PositionSizingMethod
from app.services.auto_strategy.models.ga_config import GAConfig
from app.services.backtest.factories.strategy_class_factory import StrategyClassFactory
from app.services.backtest.execution.backtest_executor import BacktestExecutor
from app.services.backtest.conversion.backtest_result_converter import (
    BacktestResultConverter,
)
from database.models import Base
from database.repositories.backtest_result_repository import BacktestResultRepository


class _SyntheticBacktestDataService:
    def __init__(self, n=600, seed=0):
        self.n = n
        self.seed = seed

    def get_data_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        periods = self.n
        idx = pd.date_range(start=start_date, end=end_date, periods=periods)
        base = 30000.0
        returns = rng.normal(0, 0.008, size=periods)
        prices = [base]
        for r in returns[1:]:
            prices.append(max(500.0, prices[-1] * (1 + r)))
        close = np.array(prices)
        high = close * (1 + np.abs(rng.normal(0, 0.004, size=periods)))
        low = close * (1 - np.abs(rng.normal(0, 0.004, size=periods)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        vol = rng.integers(10, 10000, size=periods)
        df = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )
        return df


def test_strategy_diversity_audit():
    # DBはインメモリSQLite
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    converter = BacktestResultConverter()
    strategy_factory = StrategyClassFactory()

    timeframes = ["15m", "30m", "1h", "4h", "1d"]

    # 生成する戦略数（各時間軸で20、合計100程度）
    per_tf = 20

    # 集計用
    zero_trade_count = 0
    total_count = 0
    tpsl_methods_counter = Counter()
    pos_methods_counter = Counter()
    tf_counter = Counter()

    # 各時間軸で戦略を作成
    for tf_ix, tf in enumerate(timeframes):
        for i in range(per_tf):
            seed = tf_ix * 1000 + i
            random.seed(seed)
            np.random.seed(seed)

            # GAConfigは技術/統計/MATH/PRなど多様性を許容（デフォルト）
            ga_cfg = GAConfig(
                indicator_mode="technical_only",  # まずはテクニカル主体
                max_indicators=4,
                min_indicators=2,
                max_conditions=5,
                min_conditions=2,
            )

            gene = RandomGeneGenerator(
                config=ga_cfg, enable_smart_generation=True
            ).generate_random_gene()

            # 時間軸を遺伝子metadataに載せる（分析用）。tpsl/position sizingはランダム生成される
            if not hasattr(gene, "metadata") or gene.metadata is None:
                gene.metadata = {}
            gene.metadata["timeframe"] = tf

            # 戦略クラス生成
            strategy_class = strategy_factory.create_strategy_class(gene)

            # 合成データ（時間軸に応じた期間を調整）
            start_date = datetime(2023, 1, 1)
            # 期間は各tfに合わせてbar本数が十分になるよう適当に調整
            if tf == "15m":
                end_date = start_date + timedelta(days=30)
                n = 24 * 4 * 30
            elif tf == "30m":
                end_date = start_date + timedelta(days=60)
                n = 24 * 2 * 60
            elif tf == "1h":
                end_date = start_date + timedelta(days=120)
                n = 24 * 120
            elif tf == "4h":
                end_date = start_date + timedelta(days=365)
                n = 6 * 365
            else:  # 1d
                end_date = start_date + timedelta(days=5 * 365)
                n = 5 * 365

            data_service = _SyntheticBacktestDataService(n=n, seed=seed)
            executor = BacktestExecutor(data_service)

            # バックテスト実行
            stats = executor.execute_backtest(
                strategy_class=strategy_class,
                strategy_parameters={"strategy_gene": gene},
                symbol="BTC:USDT",
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000.0,
                commission_rate=0.001,
            )

            # 結果変換と保存
            config_json = {
                "strategy_config": {"strategy_gene": "..."},
                "commission_rate": 0.001,
            }
            result = converter.convert_backtest_results(
                stats=stats,
                strategy_name="auto_strategy",
                symbol="BTC:USDT",
                timeframe=tf,
                initial_capital=10000.0,
                start_date=start_date,
                end_date=end_date,
                config_json=config_json,
            )

            db = TestingSessionLocal()
            try:
                saved = BacktestResultRepository(db).save_backtest_result(result)
            finally:
                db.close()

            # ゼロトレード判定
            trades = saved.get("trade_history", [])
            if not trades:
                zero_trade_count += 1
            total_count += 1

            # 遺伝子からメソッド種類をカウント
            tpsl_gene = getattr(gene, "tpsl_gene", None)
            if tpsl_gene and tpsl_gene.enabled:
                tpsl_methods_counter[tpsl_gene.method.value] += 1
            pos_gene = getattr(gene, "position_sizing_gene", None)
            if pos_gene and pos_gene.enabled:
                pos_methods_counter[pos_gene.method.value] += 1

            tf_counter[tf] += 1

    # 結果の簡易アサーション
    assert total_count == per_tf * len(timeframes)

    # 少なくとも各時間軸が生成されている
    for tf in timeframes:
        assert tf_counter[tf] == per_tf

    # TPSLメソッドに多様性がある（少なくとも2種類は出現）
    assert len([m for m, c in tpsl_methods_counter.items() if c > 0]) >= 2

    # ポジションサイジングメソッドに多様性がある（少なくとも2種類は出現）
    assert len([m for m, c in pos_methods_counter.items() if c > 0]) >= 2

    # テストが通れば統計をprint（pytest -s時に見える）。本テストは集計の存在のみ検証
    print(
        {
            "total_strategies": total_count,
            "zero_trade": zero_trade_count,
            "tpsl_methods": dict(tpsl_methods_counter),
            "position_sizing_methods": dict(pos_methods_counter),
            "timeframes": dict(tf_counter),
        }
    )
