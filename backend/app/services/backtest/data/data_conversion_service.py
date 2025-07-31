"""
データ変換サービス

データベースモデルからpandas.DataFrameへの変換を専門に担当します。
"""

import logging
from typing import List


import pandas as pd

from database.models import (
    OHLCVData,
    OpenInterestData,
    FundingRateData,
    FearGreedIndexData,
)

logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """データ変換エラー"""

    pass


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

            logger.debug(f"OHLCV DataFrame変換完了: {len(df)}行")
            return df

        except Exception as e:
            logger.error(f"OHLCV DataFrame変換エラー: {e}")
            raise DataConversionError(f"OHLCVデータの変換に失敗しました: {e}")

    def convert_open_interest_to_dataframe(
        self, oi_data: List[OpenInterestData]
    ) -> pd.DataFrame:
        """
        OpenInterestDataリストをpandas.DataFrameに変換

        Args:
            oi_data: OpenInterestDataオブジェクトのリスト

        Returns:
            Open InterestのDataFrame
        """
        if not oi_data:
            logger.debug("Open Interestデータが空です")
            return pd.DataFrame(columns=["open_interest"])

        try:
            data = {"open_interest": [float(r.open_interest_value) for r in oi_data]}
            df = pd.DataFrame(data)
            df.index = pd.DatetimeIndex([r.data_timestamp for r in oi_data])

            # データ型を最適化
            df["open_interest"] = df["open_interest"].astype("float64")

            logger.debug(f"Open Interest DataFrame変換完了: {len(df)}行")
            return df

        except Exception as e:
            logger.warning(f"Open Interest DataFrame変換エラー: {e}")
            return pd.DataFrame(columns=["open_interest"])

    def convert_funding_rate_to_dataframe(
        self, fr_data: List[FundingRateData]
    ) -> pd.DataFrame:
        """
        FundingRateDataリストをpandas.DataFrameに変換

        Args:
            fr_data: FundingRateDataオブジェクトのリスト

        Returns:
            Funding RateのDataFrame
        """
        if not fr_data:
            logger.debug("Funding Rateデータが空です")
            return pd.DataFrame(columns=["funding_rate"])

        try:
            data = {"funding_rate": [float(r.funding_rate) for r in fr_data]}
            df = pd.DataFrame(data)
            df.index = pd.DatetimeIndex([r.funding_timestamp for r in fr_data])

            # データ型を最適化
            df["funding_rate"] = df["funding_rate"].astype("float64")

            logger.debug(f"Funding Rate DataFrame変換完了: {len(df)}行")
            return df

        except Exception as e:
            logger.warning(f"Funding Rate DataFrame変換エラー: {e}")
            return pd.DataFrame(columns=["funding_rate"])

    def convert_fear_greed_to_dataframe(
        self, fear_greed_data: List[FearGreedIndexData]
    ) -> pd.DataFrame:
        """
        FearGreedIndexDataリストをpandas.DataFrameに変換

        Args:
            fear_greed_data: FearGreedIndexDataオブジェクトのリスト

        Returns:
            Fear & GreedのDataFrame
        """
        if not fear_greed_data:
            logger.debug("Fear & Greedデータが空です")
            return pd.DataFrame(
                columns=["fear_greed_value", "fear_greed_classification"]
            )

        try:
            data = {
                "fear_greed_value": [float(r.value) for r in fear_greed_data],
                "fear_greed_classification": [
                    str(r.value_classification) for r in fear_greed_data
                ],
            }
            df = pd.DataFrame(data)
            df.index = pd.DatetimeIndex([r.data_timestamp for r in fear_greed_data])

            # データ型を最適化
            df["fear_greed_value"] = df["fear_greed_value"].astype("float64")
            df["fear_greed_classification"] = df["fear_greed_classification"].astype(
                "string"
            )

            logger.debug(f"Fear & Greed DataFrame変換完了: {len(df)}行")
            return df

        except Exception as e:
            logger.warning(f"Fear & Greed DataFrame変換エラー: {e}")
            return pd.DataFrame(
                columns=["fear_greed_value", "fear_greed_classification"]
            )

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

    def create_empty_dataframe(self, columns: List[str]) -> pd.DataFrame:
        """
        指定されたカラムを持つ空のDataFrameを作成

        Args:
            columns: カラム名のリスト

        Returns:
            空のDataFrame
        """
        df = pd.DataFrame(columns=columns)

        # DatetimeIndexを設定
        df.index = pd.DatetimeIndex([])

        return df

    def validate_dataframe_structure(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> bool:
        """
        DataFrameの構造を検証

        Args:
            df: 検証対象のDataFrame
            required_columns: 必須カラムのリスト

        Returns:
            構造が正しい場合True
        """
        try:
            # カラムの存在確認
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"必須カラムが不足: {missing_columns}")
                return False

            # インデックスがDatetimeIndexかどうか確認
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("インデックスがDatetimeIndexではありません")
                return False

            # データが空でないことを確認
            if df.empty:
                logger.warning("DataFrameが空です")
                return False

            return True

        except Exception as e:
            logger.error(f"DataFrame構造検証エラー: {e}")
            return False
