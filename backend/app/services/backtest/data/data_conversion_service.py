"""
データ変換サービス

"""

import logging
from typing import List

import pandas as pd

from app.utils.error_handler import safe_operation
from database.models import (
    OHLCVData,
)

logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """データ変換エラー"""


class DataConversionService:
    """
    データ変換サービス

    データベースモデルからpandas.DataFrameへの変換を専門に担当します。
    """

    @safe_operation(context="OHLCV DataFrame変換", is_api_call=False, default_return=pd.DataFrame())
    def convert_ohlcv_to_dataframe(self, ohlcv_data: List[OHLCVData]) -> pd.DataFrame:
        """
        OHLCVDataリストをpandas.DataFrameに変換

        Args:
            ohlcv_data: OHLCVDataオブジェクトのリスト

        Returns:
            backtesting.py用のDataFrame

        Raises:
            DataConversionError: 変換に失敗した場合
        """
        if not ohlcv_data:
            return pd.DataFrame()

        # 効率的にDataFrameを作成
        data = {
            "timestamp": [r.timestamp for r in ohlcv_data],
            "open": [float(r.open) for r in ohlcv_data],
            "high": [float(r.high) for r in ohlcv_data],
            "low": [float(r.low) for r in ohlcv_data],
            "close": [float(r.close) for r in ohlcv_data],
            "volume": [float(r.volume) for r in ohlcv_data],
        }

        df = pd.DataFrame(data)

        # インデックスをdatetimeに設定
        df.index = pd.DatetimeIndex([r.timestamp for r in ohlcv_data])

        # データ型を最適化
        df = self._optimize_ohlcv_dtypes(df)

        return df

    @safe_operation(context="OHLCVデータ型最適化", is_api_call=False)
    def _optimize_ohlcv_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCVデータのデータ型を最適化

        Args:
            df: 最適化対象のDataFrame

        Returns:
            最適化されたDataFrame
        """
        # timestampはdatetime型を維持
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 価格データは高精度が必要なのでfloat64を維持
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype("float64")

        # ボリュームは整数でも可
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(
                "float64"
            )  # 小数点以下がある場合を考慮

        return df
