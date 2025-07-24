"""
テストデータ生成ユーティリティ

各種テストで使用するデータ生成関数を提供します。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.gene_strategy import StrategyGene
from app.core.services.auto_strategy.models.gene_tpsl import TPSLGene, TPSLMethod
from app.core.services.auto_strategy.models.gene_position_sizing import (
    PositionSizingGene,
    PositionSizingMethod,
)


class TestDataGenerator:
    """テストデータ生成クラス"""

    @staticmethod
    def generate_ohlcv_data(
        length: int = 100,
        start_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0,
    ) -> pd.DataFrame:
        """OHLCV データを生成"""
        np.random.seed(42)  # 再現可能性のため

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=length), periods=length, freq="D"
        )

        prices = []
        current_price = start_price

        for i in range(length):
            # トレンドとランダムウォークを組み合わせ
            change = np.random.normal(trend, volatility)
            current_price *= 1 + change

            # OHLC を生成
            daily_volatility = volatility * 0.5
            high = current_price * (1 + abs(np.random.normal(0, daily_volatility)))
            low = current_price * (1 - abs(np.random.normal(0, daily_volatility)))
            open_price = current_price * (
                1 + np.random.normal(0, daily_volatility * 0.3)
            )

            # 論理的な制約を適用
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)

            volume = np.random.uniform(1000, 10000)

            prices.append(
                {
                    "timestamp": dates[i],
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": current_price,
                    "volume": volume,
                }
            )

        return pd.DataFrame(prices)

    @staticmethod
    def generate_market_scenarios() -> Dict[str, pd.DataFrame]:
        """様々な市場シナリオのデータを生成"""
        scenarios = {}

        # 上昇トレンド
        scenarios["bull_market"] = TestDataGenerator.generate_ohlcv_data(
            length=200, trend=0.001, volatility=0.015
        )

        # 下降トレンド
        scenarios["bear_market"] = TestDataGenerator.generate_ohlcv_data(
            length=200, trend=-0.001, volatility=0.02
        )

        # 横ばい市場
        scenarios["sideways_market"] = TestDataGenerator.generate_ohlcv_data(
            length=200, trend=0.0, volatility=0.01
        )

        # 高ボラティリティ市場
        scenarios["high_volatility"] = TestDataGenerator.generate_ohlcv_data(
            length=200, trend=0.0, volatility=0.05
        )

        # 低ボラティリティ市場
        scenarios["low_volatility"] = TestDataGenerator.generate_ohlcv_data(
            length=200, trend=0.0, volatility=0.005
        )

        return scenarios

    @staticmethod
    def generate_ga_config(
        population_size: int = 10,
        generations: int = 5,
        max_indicators: int = 3,
        min_indicators: int = 1,
    ) -> GAConfig:
        """GAConfig を生成"""
        return GAConfig(
            population_size=population_size,
            generations=generations,
            max_indicators=max_indicators,
            min_indicators=min_indicators,
            max_conditions=3,
            min_conditions=1,
        )

    @staticmethod
    def generate_strategy_gene() -> StrategyGene:
        """StrategyGene を生成"""
        from app.core.services.auto_strategy.generators.random_gene_generator import (
            RandomGeneGenerator,
        )

        config = TestDataGenerator.generate_ga_config()
        generator = RandomGeneGenerator(config)
        return generator.generate_random_gene()

    @staticmethod
    def generate_tpsl_gene(
        method: TPSLMethod = TPSLMethod.FIXED_PERCENTAGE,
    ) -> TPSLGene:
        """TPSLGene を生成"""
        return TPSLGene(
            method=method,
            take_profit_percentage=Decimal("2.0"),
            stop_loss_percentage=Decimal("1.0"),
            trailing_stop_percentage=Decimal("0.5"),
            atr_multiplier_tp=Decimal("2.0"),
            atr_multiplier_sl=Decimal("1.5"),
        )

    @staticmethod
    def generate_position_sizing_gene(
        method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE,
    ) -> PositionSizingGene:
        """PositionSizingGene を生成"""
        return PositionSizingGene(
            method=method,
            fixed_percentage=Decimal("10.0"),
            kelly_multiplier=Decimal("0.25"),
            volatility_target=Decimal("2.0"),
            max_position_size=Decimal("20.0"),
            min_position_size=Decimal("1.0"),
        )

    @staticmethod
    def generate_extreme_market_conditions() -> Dict[str, pd.DataFrame]:
        """極端な市場条件のデータを生成"""
        conditions = {}

        # フラッシュクラッシュ
        flash_crash_data = TestDataGenerator.generate_ohlcv_data(100)
        # 50日目に急落を挿入
        crash_idx = 50
        flash_crash_data.loc[crash_idx, "low"] *= 0.7
        flash_crash_data.loc[crash_idx, "close"] *= 0.8
        conditions["flash_crash"] = flash_crash_data

        # ギャップアップ
        gap_up_data = TestDataGenerator.generate_ohlcv_data(100)
        gap_idx = 50
        gap_multiplier = 1.15
        gap_up_data.loc[gap_idx:, ["open", "high", "low", "close"]] *= gap_multiplier
        conditions["gap_up"] = gap_up_data

        # 極低ボラティリティ
        low_vol_data = TestDataGenerator.generate_ohlcv_data(100, volatility=0.001)
        conditions["extremely_low_volatility"] = low_vol_data

        return conditions


class PerformanceTestHelper:
    """パフォーマンステスト用ヘルパー"""

    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """関数の実行時間を測定"""
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time

    @staticmethod
    def measure_memory_usage(func, *args, **kwargs) -> Tuple[Any, float]:
        """関数のメモリ使用量を測定"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        return result, memory_used


class ValidationHelper:
    """検証用ヘルパー"""

    @staticmethod
    def validate_financial_calculation(
        result: float, expected: float, tolerance: float = 1e-8
    ) -> bool:
        """財務計算の結果を検証"""
        return abs(result - expected) <= tolerance

    @staticmethod
    def validate_dataframe_structure(
        df: pd.DataFrame, required_columns: List[str]
    ) -> bool:
        """DataFrameの構造を検証"""
        return all(col in df.columns for col in required_columns)

    @staticmethod
    def validate_decimal_precision(value: Decimal, expected_places: int) -> bool:
        """Decimal の精度を検証"""
        return len(str(value).split(".")[-1]) <= expected_places
