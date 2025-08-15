"""
データ変換サービス

"""

import logging
from typing import List

import pandas as pd

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
            raise DataConversionError("OHLCVデータが空です")

        try:
            # 効率的にDataFrameを作成
            data = {
                "Open": [float(r.open) for r in ohlcv_data],
                "High": [float(r.high) for r in ohlcv_data],
                "Low": [float(r.low) for r in ohlcv_data],
                "Close": [float(r.close) for r in ohlcv_data],
                "Volume": [float(r.volume) for r in ohlcv_data],
            }

            df = pd.DataFrame(data)

            # インデックスをdatetimeに設定
            df.index = pd.DatetimeIndex([r.timestamp for r in ohlcv_data])

            # データ型を最適化
            df = self._optimize_ohlcv_dtypes(df)

            return df

        except Exception as e:
            logger.error(f"OHLCV DataFrame変換エラー: {e}")
            raise DataConversionError(f"OHLCVデータの変換に失敗しました: {e}")

    def _optimize_ohlcv_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCVデータのデータ型を最適化

        Args:
            df: 最適化対象のDataFrame

        Returns:
            最適化されたDataFrame
        """
        try:
            # 価格データは高精度が必要なのでfloat64を維持
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col].astype("float64")

            # ボリュームは整数でも可
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].astype(
                    "float64"
                )  # 小数点以下がある場合を考慮

            return df

        except Exception as e:
            logger.warning(f"データ型最適化エラー: {e}")
            return df
