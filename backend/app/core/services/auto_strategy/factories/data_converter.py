"""
データ変換器

backtesting.py互換のデータ変換を担当するモジュール。
"""

import logging
import pandas as pd
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class DataConverter:
    """
    データ変換器

    backtesting.py互換のデータ変換を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def convert_to_series(self, data) -> pd.Series:
        """
        backtesting.pyの_ArrayをPandas Seriesに変換

        Args:
            data: 変換対象のデータ

        Returns:
            Pandas Series
        """
        try:
            if hasattr(data, "_data"):
                return pd.Series(data._data)
            elif hasattr(data, "values"):
                return pd.Series(data.values)
            elif isinstance(data, (list, np.ndarray)):
                return pd.Series(data)
            else:
                return pd.Series(data)
        except Exception as e:
            logger.error(f"データ変換エラー: {e}")
            return pd.Series([])

    def convert_to_backtesting_format(self, data: Any) -> Any:
        """
        データをbacktesting.py形式に変換

        Args:
            data: 変換対象のデータ

        Returns:
            backtesting.py互換形式のデータ
        """
        try:
            # 既にbacktesting.py形式の場合はそのまま返す
            if hasattr(data, "_data") or hasattr(data, "Close"):
                return data

            # DataFrameの場合
            if isinstance(data, pd.DataFrame):
                return self._convert_dataframe_to_backtesting(data)

            # Seriesの場合
            if isinstance(data, pd.Series):
                return self._convert_series_to_backtesting(data)

            # その他の場合はそのまま返す
            return data

        except Exception as e:
            logger.error(f"backtesting.py形式変換エラー: {e}")
            return data

    def _convert_dataframe_to_backtesting(self, df: pd.DataFrame) -> Any:
        """
        DataFrameをbacktesting.py形式に変換

        Args:
            df: 変換対象のDataFrame

        Returns:
            backtesting.py互換形式のデータ
        """
        try:
            # 必要な列が存在するかチェック
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"必要な列が不足: {missing_columns}")
                # 不足している列を補完
                for col in missing_columns:
                    if col == "Volume":
                        df[col] = 0  # ボリュームが無い場合は0で補完
                    else:
                        df[col] = df.get("Close", 0)  # 価格系は終値で補完

            # インデックスがDatetimeでない場合は変換
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index(
                        pd.to_datetime(df["timestamp"], unit="ms")
                        .dt.tz_localize(None)
                        .rename("time")
                    )
                elif "date" in df.columns:
                    df = df.set_index(
                        pd.to_datetime(df["date"], unit="ms")
                        .dt.tz_localize(None)
                        .rename("time")
                    )
                else:
                    # デフォルトの日付インデックスを作成
                    df.index = pd.date_range(
                        start="2024-01-01", periods=len(df), freq="D"
                    )

            return df

        except Exception as e:
            logger.error(f"DataFrame変換エラー: {e}")
            return df

    def _convert_series_to_backtesting(self, series: pd.Series) -> pd.Series:
        """
        SeriesをBacktesting.py形式に変換

        Args:
            series: 変換対象のSeries

        Returns:
            backtesting.py互換形式のSeries
        """
        try:
            # インデックスがDatetimeでない場合は変換
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.date_range(
                    start="2024-01-01", periods=len(series), freq="D"
                )

            return series

        except Exception as e:
            logger.error(f"Series変換エラー: {e}")
            return series

    def validate_data_format(self, data: Any) -> bool:
        """
        データ形式の妥当性をチェック

        Args:
            data: チェック対象のデータ

        Returns:
            妥当な場合True
        """
        try:
            # DataFrameの場合
            if isinstance(data, pd.DataFrame):
                required_columns = ["Open", "High", "Low", "Close"]
                return all(col in data.columns for col in required_columns)

            # backtesting.pyのデータオブジェクトの場合
            if (
                hasattr(data, "Close")
                and hasattr(data, "High")
                and hasattr(data, "Low")
            ):
                return True

            # その他の場合は無効
            return False

        except Exception as e:
            logger.error(f"データ形式チェックエラー: {e}")
            return False

    def extract_price_data(self, data: Any) -> dict:
        """
        価格データを抽出

        Args:
            data: データオブジェクト

        Returns:
            価格データの辞書
        """
        try:
            price_data = {}

            if isinstance(data, pd.DataFrame):
                price_data["open"] = data.get("Open", data.get("open"))
                price_data["high"] = data.get("High", data.get("high"))
                price_data["low"] = data.get("Low", data.get("low"))
                price_data["close"] = data.get("Close", data.get("close"))
                price_data["volume"] = data.get("Volume", data.get("volume"))
            elif hasattr(data, "Close"):
                price_data["open"] = getattr(data, "Open", None)
                price_data["high"] = getattr(data, "High", None)
                price_data["low"] = getattr(data, "Low", None)
                price_data["close"] = getattr(data, "Close", None)
                price_data["volume"] = getattr(data, "Volume", None)

            # Noneの値を除去
            price_data = {k: v for k, v in price_data.items() if v is not None}

            return price_data

        except Exception as e:
            logger.error(f"価格データ抽出エラー: {e}")
            return {}

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列名を正規化

        Args:
            df: 対象のDataFrame

        Returns:
            列名が正規化されたDataFrame
        """
        try:
            # 列名のマッピング
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "timestamp": "Timestamp",
                "date": "Date",
            }

            # 列名を正規化
            df_normalized = df.copy()
            df_normalized.columns = [
                column_mapping.get(col.lower(), col) for col in df_normalized.columns
            ]

            return df_normalized

        except Exception as e:
            logger.error(f"列名正規化エラー: {e}")
            return df

    def ensure_numeric_data(self, data: Any) -> Any:
        """
        データが数値型であることを保証

        Args:
            data: 対象のデータ

        Returns:
            数値型に変換されたデータ
        """
        try:
            if isinstance(data, pd.DataFrame):
                numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors="coerce")
                return data

            elif isinstance(data, pd.Series):
                return pd.to_numeric(data, errors="coerce")

            else:
                return data

        except Exception as e:
            logger.error(f"数値型変換エラー: {e}")
            return data
