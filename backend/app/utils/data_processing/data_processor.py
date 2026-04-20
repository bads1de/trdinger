"""
データプロセッサーの統合インターフェース

リファクタリング後の新しいモジュール構造を統合した高レベルAPIを提供。
transformers, pipelines, validatorsモジュールを統一的に操作可能。
"""

import logging
from typing import List, cast

import numpy as np
import pandas as pd

from .data_validator import (
    validate_data_integrity,
    validate_extended_data,
    validate_ohlcv_data,
)
from .dtype_optimizer import DtypeOptimizer

logger = logging.getLogger(__name__)


def _replace_inf_with_nan(series: pd.Series) -> pd.Series:
    """
    Series 内の inf を NaN に揃える。

    正の無限大と負の無限大をNaNに置換します。

    Args:
        series: 処理対象のpandas Series

    Returns:
        pd.Series: infをNaNに置換したSeries
    """
    return series.replace([np.inf, -np.inf], np.nan)


class DataProcessor:
    """
    統合データ処理クラス

    transformers, pipelines, validatorsモジュールを統合した高レベルAPIを提供します。
    データのクリーニング、検証、補間、型最適化を一括して実行できます。
    """

    def __init__(self):
        """
        DataProcessorを初期化します。

        現在の実装では初期化パラメータはありません。
        """
        pass

    def clean_and_validate_data(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        interpolate: bool = True,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """
        データのクリーニングと検証を一括実行

        以下の処理を順番に実行します：
        1. カラム名を小文字に統一
        2. 拡張データ（funding_rate等）の範囲クリップ
        3. データ補間（NaN/null値の処理）
        4. データ検証（OHLC関係、整合性等）
        5. データ型最適化
        6. 時系列順にソート

        Args:
            df: 処理対象のDataFrame
            required_columns: 必須カラムのリスト（検証に使用）
            interpolate: 補間処理を実行するか（デフォルト: True）
            optimize: データ型最適化を実行するか（デフォルト: True）

        Returns:
            pd.DataFrame: クリーニング済みのDataFrame

        Raises:
            ValueError: データ検証に失敗した場合
        """
        result_df = df.copy()

        # カラム名を小文字に統一（大文字小文字のケースを統一）
        result_df.columns = result_df.columns.str.lower()

        # 拡張データの範囲クリップ（funding_rateなど）
        result_df = self._clip_extended_data_ranges(result_df)

        # 必要なカラムを定義
        ohlcv_columns = ["open", "high", "low", "close", "volume"]

        # データ補間 (NaN/null値を先に処理)
        if interpolate:
            result_df = self._interpolate_data(result_df)

        # データ検証 (補間後に実行)
        try:
            # 空のデータは検証をスキップ
            if result_df.empty:
                logger.warning("データが空のため検証をスキップ")
                return result_df

            # 必要なカラムに基づいて検証を実行
            if not result_df.empty:
                if any(col in required_columns for col in ohlcv_columns):
                    validate_ohlcv_data(result_df)
                validate_extended_data(result_df)
                validate_data_integrity(result_df)
        except Exception as e:
            logger.error(f"データ検証でエラー: {e}")
            raise ValueError(f"データ検証に失敗しました: {e}")

        # データ型最適化
        if optimize:
            optimizer = DtypeOptimizer()
            result_df = optimizer.fit_transform(result_df)

        # 時系列順にソート
        if hasattr(result_df.index, "is_monotonic_increasing"):
            # Pandas Series/Index比較を安全に行う - boolに変換してから評価
            is_sorted = bool(result_df.index.is_monotonic_increasing)
            if not is_sorted:
                result_df = result_df.sort_index()

        return result_df

    def clear_cache(self):
        """
        キャッシュをクリアします。

        現在の実装ではキャッシュ機能はありませんが、
        将来的な拡張に備えてメソッドを提供しています。
        """
        logger.info("DataProcessorのキャッシュをクリアしました")

    def _interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ補間処理（ベクトル化版）

        数値カラムとカテゴリカルカラムの欠損値を補間します。
        - 数値カラム: inf→NaN変換後、前方補完→線形補完→ゼロ埋め
        - OHLCカラム: 補間後にlow <= open/close <= highの関係を修正
        - カテゴリカルカラム: 最頻値で補完

        Args:
            df: 補間対象のDataFrame

        Returns:
            pd.DataFrame: 補間済みのDataFrame
        """
        result_df = df.copy()

        # 数値カラムの補間（ベクトル化）
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # inf値をNaNに変換（補間前に）
            result_df[col] = _replace_inf_with_nan(cast(pd.Series, result_df[col]))

            # 欠損値がある場合のみ補間
            if bool(np.any(result_df[col].isnull().to_numpy())):
                # 前方補完 → 線形補完 → ゼロ埋め（未来データを使わない）
                result_df[col] = (
                    result_df[col].ffill().interpolate(method="linear").fillna(0)
                )

        # 特別なOHLCカラムの補間後検証と修正（ベクトル化）
        if all(col in result_df.columns for col in ["open", "high", "low", "close"]):
            # NaN値を含まない行のみ抽出
            ohlc_mask = result_df[["open", "high", "low", "close"]].notnull().all(axis=1)
            ohlc_df = result_df.loc[ohlc_mask, ["open", "high", "low", "close"]].copy()

            if not ohlc_df.empty:
                # numpy配列に変換して高速処理（.copy()で書き込み可能にする）
                open_vals = ohlc_df["open"].values.copy()
                high_vals = ohlc_df["high"].values.copy()
                low_vals = ohlc_df["low"].values.copy()
                close_vals = ohlc_df["close"].values.copy()

                # OHLC関係の検証（ベクトル化）
                valid_min = np.minimum(open_vals, close_vals)
                valid_max = np.maximum(open_vals, close_vals)

                # lowの修正
                low_invalid = low_vals > valid_min
                low_vals[low_invalid] = valid_min[low_invalid]

                # highの修正
                high_invalid = high_vals < valid_max
                high_vals[high_invalid] = valid_max[high_invalid]

                # 修正結果を反映
                result_df.loc[ohlc_mask, "low"] = low_vals
                result_df.loc[ohlc_mask, "high"] = high_vals

        # カテゴリカルカラムの補間（ベクトル化）
        categorical_columns = result_df.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_columns:
            if bool(np.any(result_df[col].isnull().to_numpy())):
                # 最頻値で補完
                mode_value = result_df[col].mode()
                if not mode_value.empty:
                    result_df[col] = result_df[col].fillna(mode_value.iloc[0])

        return result_df

    def _clip_extended_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拡張データの範囲クリップ処理

        以下の拡張データの範囲を制限します：
        - funding_rate: -1から1の範囲にクリップ
        - open_interest: 負値を0にクリップ

        Args:
            df: 処理対象のDataFrame

        Returns:
            pd.DataFrame: 範囲クリップ済みのDataFrame
        """
        result_df = df.copy()

        # funding_rateの範囲クリップ (-1から1)
        if "funding_rate" in result_df.columns:
            # NaNとinfを処理してからクリップ
            funding_rate_clean = _replace_inf_with_nan(cast(pd.Series, result_df["funding_rate"]))
            # Pandas Series比較を安全に行う
            below_min = (funding_rate_clean < -1).sum()
            above_max = (funding_rate_clean > 1).sum()
            before_count = below_min + above_max

            # 常にクリップを実行（範囲外値がなくてもNaN/infの処理のため）
            result_df["funding_rate"] = np.clip(funding_rate_clean.fillna(0), -1, 1)

            if before_count > 0:
                logger.info(f"範囲外値を修正: {before_count}件")

        # open_interestは負値にならないようにクリップ
        if "open_interest" in result_df.columns:
            oi_clean = _replace_inf_with_nan(cast(pd.Series, result_df["open_interest"]))
            # Pandas Series比較を安全に行う
            before_count = (oi_clean < 0).sum()
            if before_count > 0:
                result_df["open_interest"] = np.maximum(oi_clean.fillna(0), 0)

        return result_df


# グローバルインスタンス
data_processor = DataProcessor()
