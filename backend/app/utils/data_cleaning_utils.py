"""
データクリーナーユーティリティ

データ補間、クレンジング、最適化のロジックを提供します。
"""

import logging
from typing import List

import pandas as pd

from .data_preprocessing import data_preprocessor

logger = logging.getLogger(__name__)


class DataCleaner:
    """データクリーニングユーティリティ"""

    @staticmethod
    def interpolate_oi_fr_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        OI/FRデータの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        result_df = df.copy()

        # 共通補間ヘルパーで OI / FR を処理（ffill → 統計補完）
        target_cols = [c for c in ["open_interest", "funding_rate"] if c in result_df.columns]
        if target_cols:
            result_df = data_preprocessor.interpolate_columns(
                result_df,
                columns=target_cols,
                strategy="median",
                forward_fill=True,
                dtype="float64",
                default_fill_values=None,
                fit_if_needed=True,
            )

        return result_df

    @staticmethod
    def interpolate_fear_greed_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fear & Greedデータの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        result_df = df.copy()

        # Fear & Greed 数値列: 共通補間ヘルパー適用（デフォルト50.0）
        if "fear_greed_value" in result_df.columns:
            result_df = data_preprocessor.interpolate_columns(
                result_df,
                columns=["fear_greed_value"],
                strategy="median",
                forward_fill=True,
                dtype="float64",
                default_fill_values={"fear_greed_value": 50.0},
                fit_if_needed=True,
            )

        # Fear & Greed クラス列: 文字列列のため簡易前方補完＋既定値
        if "fear_greed_classification" in result_df.columns:
            fg_class_series = result_df["fear_greed_classification"].replace({pd.NA: None})
            fg_class_series = fg_class_series.astype("string")
            result_df["fear_greed_classification"] = fg_class_series.ffill().fillna("Neutral")

        return result_df

    @staticmethod
    def interpolate_all_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        全データの補間処理

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        logger.info("データ補間処理を開始")

        result_df = df.copy()

        # OI/FRデータの補間
        result_df = DataCleaner.interpolate_oi_fr_data(result_df)

        # Fear & Greedデータの補間
        result_df = DataCleaner.interpolate_fear_greed_data(result_df)

        logger.info("データ補間処理が完了")
        return result_df

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        データ型を最適化してメモリ使用量を削減

        Args:
            df: 最適化するDataFrame

        Returns:
            最適化されたDataFrame
        """
        try:
            optimized_df = df.copy()

            for col in optimized_df.columns:
                if col == "timestamp":
                    continue

                if optimized_df[col].dtype == "float64":
                    # float64をfloat32に変換（精度は十分）
                    optimized_df[col] = optimized_df[col].astype("float32")
                elif optimized_df[col].dtype == "int64":
                    # int64をint32に変換（範囲が十分な場合）
                    if (
                        optimized_df[col].min() >= -2147483648
                        and optimized_df[col].max() <= 2147483647
                    ):
                        optimized_df[col] = optimized_df[col].astype("int32")

            return optimized_df

        except Exception as e:
            logger.warning(f"データ型最適化エラー: {e}")
            return df

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> None:
        """
        OHLCVデータの整合性をチェック

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: DataFrameが無効な場合
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # 必須カラムの存在確認
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        # 空のDataFrameチェック
        if df.empty:
            raise ValueError("DataFrameが空です。")

        # インデックスがDatetimeIndexかチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("インデックスがDatetimeIndexではありません。")

        # NaN値のチェック（OHLCVにNaNがある場合はエラー）
        if df[required_columns].isnull().any().any():
            raise ValueError("OHLCVデータにNaN値が含まれています。")

        # 価格の論理チェック（High >= Low, Open/Close が High-Low 範囲内）
        if not (df["High"] >= df["Low"]).all():
            raise ValueError("High価格がLow価格より低い行があります。")

        if not ((df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])).all():
            raise ValueError("Open価格がHigh-Low範囲外の行があります。")

        if not ((df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])).all():
            raise ValueError("Close価格がHigh-Low範囲外の行があります。")

    @staticmethod
    def validate_extended_data(df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        拡張データの整合性をチェック

        Args:
            df: 検証対象のDataFrame
            required_columns: 必須カラムのリスト

        Raises:
            ValueError: DataFrameが無効な場合
        """
        # 基本的なOHLCVデータの検証
        DataCleaner.validate_ohlcv_data(df)

        # 追加カラムの存在確認
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")

        # 重複インデックスのチェック
        if df.index.duplicated().any():
            logger.warning("重複したタイムスタンプが検出されました。")

        # ソート確認
        if not df.index.is_monotonic_increasing:
            logger.warning("インデックスが時系列順にソートされていません。")

    @staticmethod
    def clean_and_validate_data(
        df: pd.DataFrame,
        required_columns: List[str],
        interpolate: bool = True,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """
        データのクリーニングと検証を一括実行

        Args:
            df: 対象のDataFrame
            required_columns: 必須カラムのリスト
            interpolate: 補間処理を実行するか
            optimize: データ型最適化を実行するか

        Returns:
            クリーニング済みのDataFrame
        """
        result_df = df.copy()

        # データ補間
        if interpolate:
            result_df = DataCleaner.interpolate_all_data(result_df)

        # データ型最適化
        if optimize:
            result_df = DataCleaner.optimize_dtypes(result_df)

        # データ検証
        DataCleaner.validate_extended_data(result_df, required_columns)

        # 時系列順にソート
        result_df = result_df.sort_index()

        return result_df
