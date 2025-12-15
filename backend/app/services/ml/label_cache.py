"""
ラベル生成キャッシュクラス

Optunaの最適化中に同じパラメータでラベルを再生成するコストを削減します。
ラベル生成パラメータの組み合わせをキーとしてメモ化します。
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from app.services.ml.label_generation.enums import ThresholdMethod
from app.services.ml.label_generation.presets import (
    trend_scanning_preset,
    triple_barrier_method_preset,
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
            pt_factor: トリプルバリア法のプロフィットテイキング乗数。
            sl_factor: トリプルバリア法のストップロス乗数。
            use_atr: トリプルバリア法でボラティリティにATRを使用するかどうか。
            atr_period: ATR計算期間。
            binary_label: バイナリ（0/1）ラベルを返すかどうか。
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
            raise ValueError(
                f"無効な閾値計算方法: {threshold_method}, 有効なメソッド: {valid_methods}"
            )

        if threshold_method_enum == ThresholdMethod.TRIPLE_BARRIER:
            # presets.py の実装を使用
            # min_ret=0.0001, volatility_window=24 は以前のLabelCache実装に合わせてハードコード
            labels = triple_barrier_method_preset(
                df=self.ohlcv_df,
                timeframe=timeframe,
                horizon_n=horizon_n,
                pt=pt_factor,
                sl=sl_factor,
                min_ret=0.0001,
                price_column=price_column,
                volatility_window=24,
                use_atr=use_atr,
                atr_period=atr_period,
                binary_label=binary_label,
                t_events=t_events,
            )

        elif threshold_method_enum == ThresholdMethod.TREND_SCANNING:
            # presets.py の実装を使用
            labels = trend_scanning_preset(
                df=self.ohlcv_df,
                timeframe=timeframe,
                horizon_n=horizon_n,
                threshold=threshold,
                min_window=min_window,
                window_step=window_step,
                price_column=price_column,
                binary_label=binary_label,
                t_events=t_events,
            )

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


