"""
ラベル生成キャッシュクラス

Optunaの最適化中に同じパラメータでラベルを再生成するコストを削減します。
ラベル生成パラメータの組み合わせをキーとしてメモ化します。
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from app.services.ml.label_generation.enums import ThresholdMethod
from app.services.ml.label_generation.triple_barrier import TripleBarrier
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

    Example:
        >>> cache = LabelCache(ohlcv_df)
        >>> labels = cache.get_labels(
        ...     horizon_n=4,
        ...     threshold_method="QUANTILE",
        ...     threshold=0.33,
        ...     timeframe="1h",
        ...     price_column="close"
        ... )
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
    ) -> pd.Series:
        """キャッシュを使ってラベルを取得

        パラメータの組み合わせをキーとしてキャッシュを検索し、
        見つかればキャッシュから返し、なければ生成してキャッシュに保存します。

        Args:
            horizon_n: N本先を見る
            threshold_method: 閾値計算方法（enum名の文字列）
            threshold: 閾値
            timeframe: 時間足
            price_column: 価格カラム名
            pt_factor: Profit Taking multiplier for Triple Barrier.
            sl_factor: Stop Loss multiplier for Triple Barrier.
            use_atr: Use ATR for volatility in Triple Barrier.
            atr_period: ATR calculation period.
            binary_label: Return binary (0/1) labels for Triple Barrier.

        Returns:
            pd.Series: ラベル（"UP"/"RANGE"/"DOWN" または 0/1）

        Raises:
            ValueError: threshold_methodが無効な場合
        """
        # キャッシュキーを生成（タプルは hashable なので辞書のキーに使える）
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
        )

        # キャッシュヒット
        if cache_key in self.cache:
            self.hit_count += 1
            logger.debug(
                f"キャッシュヒット: horizon_n={horizon_n}, "
                f"method={threshold_method}, threshold={threshold:.4f}, "
                f"timeframe={timeframe} "
                f"(hit率: {self.get_hit_rate():.1f}%)"
            )
            return self.cache[cache_key]

        # キャッシュミス -> 新規生成
        self.miss_count += 1
        logger.info(
            f"ラベル生成: horizon_n={horizon_n}, "
            f"method={threshold_method}, threshold={threshold:.4f}, "
            f"timeframe={timeframe} "
            f"(miss率: {self.get_miss_rate():.1f}%)"
        )

        # ThresholdMethod enum に変換
        try:
            threshold_method_enum = ThresholdMethod[threshold_method]
        except KeyError:
            valid_methods = [m.name for m in ThresholdMethod]
            raise ValueError(
                f"無効な閾値計算方法: {threshold_method}. "
                f"有効な値: {', '.join(valid_methods)}"
            )

        # ラベル生成
        if threshold_method_enum == ThresholdMethod.TRIPLE_BARRIER:
            close_prices = self.ohlcv_df[price_column]

            # ボラティリティ計算
            if use_atr:
                volatility = calculate_volatility_atr(
                    high=self.ohlcv_df["high"],
                    low=self.ohlcv_df["low"],
                    close=close_prices,
                    window=atr_period,
                    as_percentage=True,
                )
            else:
                # 従来の標準偏差
                returns = close_prices.pct_change(fill_method=None)
                volatility = calculate_volatility_std(returns=returns, window=24)

            # 垂直バリア (horizon_n 時間後)
            t1_vertical_barrier = pd.Series(
                close_prices.index, index=close_prices.index
            ).shift(-horizon_n)

            # Triple Barrier 実行
            tb = TripleBarrier(
                pt=pt_factor, sl=sl_factor, min_ret=0.0001
            )  # min_retは小さい値で固定

            events = tb.get_events(
                close=close_prices,
                t_events=close_prices.index,  # 全てのバーをイベント候補とする
                pt_sl=[pt_factor, sl_factor],
                target=volatility,
                min_ret=0.0001,
                vertical_barrier_times=t1_vertical_barrier,
            )

            labels_df = tb.get_bins(events, close_prices, binary_label=binary_label)

            if binary_label:
                labels = labels_df["bin"]
            else:
                # ラベルをSeriesに変換 (1, -1, 0) -> ("UP", "DOWN", "RANGE")
                labels_map = {1.0: "UP", -1.0: "DOWN", 0.0: "RANGE"}
                labels = labels_df["bin"].map(labels_map)

            # インデックスを合わせてNaN処理
            labels = labels.reindex(close_prices.index)

        else:
            raise NotImplementedError(
                f"閾値計算方法 {threshold_method} はサポートされていません。"
                "現在は TRIPLE_BARRIER のみサポートしています。"
            )

        # キャッシュに保存
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
