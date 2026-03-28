"""
ラベル生成キャッシュクラス

Optunaの最適化中に同じパラメータでラベルを再生成するコストを削減します。
ラベル生成パラメータの組み合わせをキーとしてメモ化します。
"""

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from enum import Enum
from ..cross_validation import get_t1_series
from .presets import (
    trend_scanning_preset,
    triple_barrier_method_preset,
)

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """閾値計算方法"""
 
    TRIPLE_BARRIER = "triple_barrier"  # Triple Barrier Method (利確/損切り/時間切れ)
    TREND_SCANNING = "trend_scanning"  # Trend Scanning Method (t値によるトレンド判定)
 
 
class LabelCache:
    """ラベル生成結果をメモ化するキャッシュ

    同じパラメータの組み合わせで複数回ラベル生成が要求された場合、
    キャッシュから結果を返すことで計算時間を大幅に削減します。

    Attributes:
        ohlcv_df: OHLCVデータフレーム（キャッシュ全体で共有）
        cache: ラベル生成結果のキャッシュ辞書
        hit_count: キャッシュヒット回数
        miss_count: キャッシュミス回数
    """

    def __init__(self, ohlcv_df: pd.DataFrame):
        """初期化

        Args:
            ohlcv_df: OHLCVデータフレーム
        """
        self.ohlcv_df = ohlcv_df
        self.cache: Dict[Tuple, pd.Series] = {}
        self.hit_count = 0
        self.miss_count = 0

    def get_labels(
        self,
        horizon_n: int,
        threshold_method: str,
        threshold: float,
        timeframe: str = "1h",
        price_column: str = "close",
        pt_factor: float = 1.0,
        sl_factor: float = 1.0,
        use_atr: bool = False,
        atr_period: int = 14,
        t_events: Optional[pd.DatetimeIndex] = None,
        min_window: int = 5,
        window_step: int = 1,
        **kwargs,
    ) -> pd.Series:
        """キャッシュを使ってラベルを取得"""
        use_cache = t_events is None
        cache_key = (
            horizon_n,
            threshold_method,
            threshold,
            timeframe,
            price_column,
            pt_factor,
            sl_factor,
            use_atr,
            atr_period,
            min_window,
            window_step,
        )

        if use_cache and cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key]

        if use_cache:
            self.miss_count += 1
            logger.info(f"ラベル生成: {threshold_method}")

        try:
            m = ThresholdMethod[threshold_method]
        except KeyError:
            raise ValueError(f"無効な閾値計算方法: {threshold_method}")

        if m == ThresholdMethod.TRIPLE_BARRIER:
            labels = triple_barrier_method_preset(
                self.ohlcv_df,
                timeframe,
                horizon_n,
                pt_factor,
                sl_factor,
                0.0001,
                price_column,
                24,
                use_atr,
                atr_period,
                t_events,
            )
        elif m == ThresholdMethod.TREND_SCANNING:
            labels = trend_scanning_preset(
                self.ohlcv_df,
                timeframe,
                horizon_n,
                threshold,
                min_window,
                window_step,
                price_column,
                t_events,
            )
        else:
            raise NotImplementedError(f"Method {threshold_method} not supported")

        if use_cache:
            self.cache[cache_key] = labels
        return labels

    def get_t1(
        self, indices: pd.DatetimeIndex, horizon_n: int, timeframe: str = "1h"
    ) -> pd.Series:
        """各観測点のラベル終了時刻 (t1) を取得"""
        return get_t1_series(indices, horizon_n, timeframe=timeframe)

    def get_hit_rate(self) -> float:
        """キャッシュヒット率を取得

        Returns:
            float: ヒット率（0-100%）
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100

    def clear(self) -> None:
        """キャッシュをクリア"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("キャッシュをクリアしました")

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得

        Returns:
            Dict: ヒット数、ミス数、キャッシュサイズを含む辞書
        """
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "cache_size": len(self.cache),
            "hit_rate_pct": self.get_hit_rate(),
        }
