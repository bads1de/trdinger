"""
ラベル生成キャッシュクラス

Optunaの最適化中に同じパラメータでラベルを再生成するコストを削減します。
ラベル生成パラメータの組み合わせをキーとしてメモ化します。
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd

from app.services.ml.label_generation.enums import ThresholdMethod
from app.services.ml.label_generation.triple_barrier import TripleBarrier
from app.services.ml.label_generation.trend_scanning import TrendScanning
from app.services.ml.common.volatility_utils import (
    calculate_volatility_atr,
    calculate_volatility_std,
)

logger = logging.getLogger(__name__)


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
        binary_label: bool = False,
        t_events: Optional[pd.DatetimeIndex] = None,
        min_window: int = 5,
        window_step: int = 1,
    ) -> pd.Series:
        """キャッシュを使ってラベルを取得

        Args:
            horizon_n: N本先を見る (Trend Scanningの場合はmax_window)
            threshold_method: 閾値計算方法
            threshold: 閾値 (Trend Scanningの場合はmin_t_value)
            timeframe: 時間足
            price_column: 価格カラム名
            pt_factor: Profit Taking multiplier for Triple Barrier.
            sl_factor: Stop Loss multiplier for Triple Barrier.
            use_atr: Use ATR for volatility in Triple Barrier.
            atr_period: ATR calculation period.
            binary_label: Return binary (0/1) labels.
            t_events: ラベル付け対象のイベント時刻
            min_window: Trend Scanningの最小ウィンドウサイズ
            window_step: Trend Scanningのウィンドウステップ

        Returns:
            pd.Series: ラベル
        """
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
            binary_label,
            min_window,
            window_step,
        )

        if use_cache and cache_key in self.cache:
            self.hit_count += 1
            logger.debug(f"キャッシュヒット: {threshold_method}")
            return self.cache[cache_key]

        if use_cache:
            self.miss_count += 1
            logger.info(f"ラベル生成: {threshold_method}")

        try:
            threshold_method_enum = ThresholdMethod[threshold_method]
        except KeyError:
            valid_methods = [m.name for m in ThresholdMethod]
            raise ValueError(f"無効な閾値計算方法: {threshold_method}")

        if threshold_method_enum == ThresholdMethod.TRIPLE_BARRIER:
            close_prices = self.ohlcv_df[price_column]

            if use_atr:
                volatility = calculate_volatility_atr(
                    high=self.ohlcv_df["high"],
                    low=self.ohlcv_df["low"],
                    close=close_prices,
                    window=atr_period,
                    as_percentage=True,
                )
            else:
                returns = close_prices.pct_change(fill_method=None)
                volatility = calculate_volatility_std(returns=returns, window=24)

            t1_vertical_barrier = pd.Series(
                close_prices.index, index=close_prices.index
            ).shift(-horizon_n)

            if t_events is None:
                events_index = close_prices.index
            else:
                events_index = t_events

            tb = TripleBarrier(pt=pt_factor, sl=sl_factor, min_ret=0.0001)
            events = tb.get_events(
                close=close_prices,
                t_events=events_index,
                pt_sl=[pt_factor, sl_factor],
                target=volatility,
                min_ret=0.0001,
                vertical_barrier_times=t1_vertical_barrier,
            )
            labels_df = tb.get_bins(events, close_prices, binary_label=binary_label)

            if binary_label:
                labels = labels_df["bin"]
            else:
                labels_map = {1.0: "UP", -1.0: "DOWN", 0.0: "RANGE"}
                labels = labels_df["bin"].map(labels_map)

            labels = labels.reindex(close_prices.index)

        elif threshold_method_enum == ThresholdMethod.TREND_SCANNING:
            close_prices = self.ohlcv_df[price_column]

            ts = TrendScanning(
                min_window=min_window,
                max_window=horizon_n,
                step=window_step,
                min_t_value=threshold,
            )

            labels_df = ts.get_labels(close=close_prices, t_events=t_events)

            if binary_label:
                labels = labels_df["bin"].abs()
            else:
                labels_map = {1.0: "UP", -1.0: "DOWN", 0.0: "RANGE"}
                labels = labels_df["bin"].map(labels_map)

            labels = labels.reindex(close_prices.index)

        else:
            raise NotImplementedError(f"Method {threshold_method} not supported")

        if use_cache:
            self.cache[cache_key] = labels

        return labels

    def get_t1(
        self, indices: pd.DatetimeIndex, horizon_n: int, timeframe: str = "1h"
    ) -> pd.Series:
        """
        各観測点のラベル終了時刻 (t1) を取得

        PurgedKFoldで使用するために、各サンプルのラベルがいつ確定するかを返します。
        単純なホライズン加算ですが、Triple Barrier Methodなどの複雑なロジックに
        拡張する場合はここで計算ロジックを変更できます。

        Args:
            indices: 観測開始時刻のインデックス
            horizon_n: N本先
            timeframe: 時間足 (デフォルト "1h")

        Returns:
            pd.Series: t1 (ラベル終了時刻)
        """
        # タイムフレームに応じてtimedeltaを計算
        if timeframe == "1h":
            delta = pd.Timedelta(hours=horizon_n)
        elif timeframe == "4h":
            delta = pd.Timedelta(hours=4 * horizon_n)
        elif timeframe == "1d":
            delta = pd.Timedelta(days=horizon_n)
        elif timeframe == "15m":
            delta = pd.Timedelta(minutes=15 * horizon_n)
        else:
            # デフォルトは1hとみなす（簡易実装）
            delta = pd.Timedelta(hours=horizon_n)

        # t1を計算
        t1 = pd.Series(indices + delta, index=indices)
        return t1

    def get_hit_rate(self) -> float:
        """キャッシュヒット率を取得

        Returns:
            float: ヒット率（0-100%）
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100

    def get_miss_rate(self) -> float:
        """キャッシュミス率を取得

        Returns:
            float: ミス率（0-100%）
        """
        return 100.0 - self.get_hit_rate()

    def clear(self) -> None:
        """キャッシュをクリア"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("キャッシュをクリアしました")

    def get_stats(self) -> Dict[str, int]:
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
