"""
ラベル生成キャッシュクラス

Optunaの最適化中に同じパラメータでラベルを再生成するコストを削減します。
ラベル生成パラメータの組み合わせをキーとしてメモ化します。
"""

import logging
from typing import Dict, Tuple

import pandas as pd

from app.utils.label_generation.enums import ThresholdMethod
from app.utils.label_generation.presets import forward_classification_preset

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

        Returns:
            pd.Series: ラベル（"UP"/"RANGE"/"DOWN"）

        Raises:
            ValueError: threshold_methodが無効な場合
        """
        # キャッシュキーを生成（タプルは hashable なので辞書のキーに使える）
        cache_key = (horizon_n, threshold_method, threshold, timeframe, price_column)

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
        labels = forward_classification_preset(
            df=self.ohlcv_df,
            timeframe=timeframe,
            horizon_n=horizon_n,
            threshold=threshold,
            price_column=price_column,
            threshold_method=threshold_method_enum,
        )

        # キャッシュに保存
        self.cache[cache_key] = labels

        return labels

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
