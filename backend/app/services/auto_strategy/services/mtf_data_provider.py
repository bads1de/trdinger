"""
マルチタイムフレームデータプロバイダー

異なるタイムフレームのOHLCVデータをキャッシュ・提供し、
GA戦略で複数タイムフレームの指標を使用可能にします。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config.constants import SUPPORTED_TIMEFRAMES

logger = logging.getLogger(__name__)


# タイムフレームを分単位に変換するマッピング
TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


class MultiTimeframeDataProvider:
    """
    マルチタイムフレームデータプロバイダー

    ベースタイムフレームのOHLCVデータから、より高いタイムフレームの
    データをリサンプリングして提供します。

    Attributes:
        base_timeframe: ベースタイムフレーム（最も細かい時間軸）
        base_data: ベースデータ（DataFrameまたはbacktesting.pyのDataオブジェクト）
        _cache: タイムフレームごとのリサンプリング済みデータキャッシュ
    """

    def __init__(
        self,
        base_data: Any,
        base_timeframe: str = "1h",
        available_timeframes: Optional[List[str]] = None,
    ):
        """
        初期化

        Args:
            base_data: ベースのOHLCVデータ（backtesting.pyのDataオブジェクトまたはDataFrame）
            base_timeframe: ベースタイムフレーム
            available_timeframes: 利用可能なタイムフレームのリスト
        """
        self.base_timeframe = base_timeframe
        self._cache: Dict[str, pd.DataFrame] = {}

        # base_dataをDataFrameに変換して保持
        if hasattr(base_data, "df"):
            # backtesting.pyのDataオブジェクト
            self.base_df = base_data.df.copy()
        elif isinstance(base_data, pd.DataFrame):
            self.base_df = base_data.copy()
        else:
            raise ValueError(
                f"サポートされていないデータ型: {type(base_data)}. "
                "DataFrameまたはbacktesting.pyのDataオブジェクトが必要です。"
            )

        # 利用可能なタイムフレームを設定
        self.available_timeframes = available_timeframes or SUPPORTED_TIMEFRAMES.copy()

        # ベースタイムフレームのデータをキャッシュに追加
        self._cache[base_timeframe] = self.base_df

        logger.debug(
            f"MTFデータプロバイダー初期化: base_timeframe={base_timeframe}, "
            f"data_rows={len(self.base_df)}"
        )

    def get_data(self, timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        指定されたタイムフレームのデータを取得

        Args:
            timeframe: タイムフレーム。Noneの場合はベースタイムフレームを返す。

        Returns:
            指定されたタイムフレームのOHLCVデータ
        """
        # Noneの場合はベースタイムフレームを返す
        if timeframe is None:
            timeframe = self.base_timeframe

        # キャッシュに存在する場合はそれを返す
        if timeframe in self._cache:
            return self._cache[timeframe]

        # タイムフレームの検証
        if timeframe not in self.available_timeframes:
            logger.warning(
                f"サポートされていないタイムフレーム: {timeframe}. "
                f"ベースタイムフレームを使用します。"
            )
            return self._cache[self.base_timeframe]

        # リサンプリングが可能か確認
        if not self._can_resample_to(timeframe):
            logger.warning(
                f"リサンプリング不可: {self.base_timeframe} -> {timeframe}. "
                f"ベースタイムフレームを使用します。"
            )
            return self._cache[self.base_timeframe]

        # リサンプリングを実行
        resampled_df = self._resample_ohlcv(timeframe)
        self._cache[timeframe] = resampled_df

        logger.debug(
            f"データリサンプリング完了: {self.base_timeframe} -> {timeframe}, "
            f"rows: {len(self.base_df)} -> {len(resampled_df)}"
        )

        return resampled_df

    def _can_resample_to(self, target_timeframe: str) -> bool:
        """
        目標タイムフレームへのリサンプリングが可能かチェック

        Args:
            target_timeframe: 目標タイムフレーム

        Returns:
            リサンプリング可能な場合True
        """
        base_minutes = TIMEFRAME_TO_MINUTES.get(self.base_timeframe, 60)
        target_minutes = TIMEFRAME_TO_MINUTES.get(target_timeframe, 60)

        # ターゲットはベース以上の時間軸である必要がある
        # かつ、ターゲットがベースの整数倍である必要がある
        if target_minutes < base_minutes:
            logger.warning(
                f"ダウンサンプリングはサポートしていません: "
                f"{self.base_timeframe}({base_minutes}m) -> "
                f"{target_timeframe}({target_minutes}m)"
            )
            return False

        if target_minutes % base_minutes != 0:
            logger.warning(
                f"非整数倍のリサンプリングはサポートしていません: "
                f"{target_minutes} % {base_minutes} != 0"
            )
            return False

        return True

    def _resample_ohlcv(self, target_timeframe: str) -> pd.DataFrame:
        """
        OHLCVデータを目標タイムフレームにリサンプリング

        pandasのresampleを使用して、Open/High/Low/Close/Volumeの各カラムを
        適切に集約します。インデックスがDatetimeIndexでない場合は変換を試みます。

        Args:
            target_timeframe: 目標タイムフレーム（'1h', '4h', '1d' など）

        Returns:
            リサンプリングされたDataFrame
        """
        # マッピングを統合して管理
        rule_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        rule = rule_map.get(target_timeframe, "1h")

        df = self.base_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        return (
            df.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache = {self.base_timeframe: self.base_df}
        logger.debug("MTFデータキャッシュをクリアしました")

    @property
    def cached_timeframes(self) -> List[str]:
        """キャッシュされているタイムフレームのリスト"""
        return list(self._cache.keys())
