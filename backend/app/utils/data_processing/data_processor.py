"""
データプロセッサーの統合インターフェース

リファクタリング後の新しいモジュール構造を統合した高レベルAPIを提供。
transformers, pipelines, validatorsモジュールを統一的に操作可能。
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from .transformers.dtype_optimizer import DtypeOptimizer
from .validators.data_validator import (
    validate_data_integrity,
    validate_extended_data,
    validate_ohlcv_data,
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    統合データ処理クラス

    transformers, pipelines, validatorsモジュールを統合した高レベルAPIを提供。
    既存のAPIとの後方互換性を維持。
    """

    def __init__(self):
        """初期化"""
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

        Args:
            df: 対象のDataFrame
            required_columns: 必須カラムのリスト
            interpolate: 補間処理を実行するか
            optimize: データ型最適化を実行するか

        Returns:
            クリーニング済みのDataFrame
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
        """キャッシュをクリア"""
        logger.info("DataProcessorのキャッシュをクリアしました")

    def _interpolate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ補間処理"""
        result_df = df.copy()

        # 数値カラムの補間
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            # inf値をNaNに変換（補間前に）
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)

            # Pandas Series比較を安全に行う - .item()でスカラー値を取得
            # .any()がSeriesを返す場合に備えて、明示的にitem()を使用
            try:
                has_null_check = result_df[col].isnull().any()
                # has_null_checkがSeriesの場合、.item()でスカラー値を取得
                if hasattr(has_null_check, "item"):
                    has_null = has_null_check.item()
                else:
                    has_null = bool(has_null_check)
            except (ValueError, AttributeError):
                # フォールバック: 要素ごとにチェック
                has_null = result_df[col].isnull().values.any()

            if has_null:
                # 前方補完 → 線形補完 → ゼロ埋め（未来データを使わない）
                # bfill() は未来のデータを使うためデータリークの原因となるため削除
                result_df[col] = (
                    result_df[col].ffill().interpolate(method="linear").fillna(0)
                )

        # 特別なOHLCカラムの補間後検証と修正
        if all(col in result_df.columns for col in ["open", "high", "low", "close"]):
            # OHLCカラムが全て存在する場合のみ検証
            for idx in result_df.index:
                row = result_df.loc[idx]

                # NaN値が含まれている行はスキップ
                ohlc_values = row[["open", "high", "low", "close"]]
                # Pandas Series比較を安全に行う - boolに変換してから評価
                has_null_ohlc = bool(ohlc_values.isnull().any())
                if has_null_ohlc:
                    continue

                # OHLC関係が崩れている場合は修正
                # Pandas Series比較を安全に行う
                low_val = float(row["low"])
                open_val = float(row["open"])
                high_val = float(row["high"])
                close_val = float(row["close"])

                ohlc_valid = (
                    (low_val <= open_val)
                    and (open_val <= high_val)
                    and (low_val <= close_val)
                    and (close_val <= high_val)
                )

                if not ohlc_valid:
                    # OHLC関係を強制的に修正
                    valid_min = min(open_val, close_val)
                    valid_max = max(open_val, close_val)

                    # lowが適切な最小値になるように修正
                    if low_val > valid_min:
                        result_df.loc[idx, "low"] = valid_min

                    # highが適切な最大値になるように修正
                    if high_val < valid_max:
                        result_df.loc[idx, "high"] = valid_max

        # カテゴリカルカラムの補間
        categorical_columns = result_df.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_columns:
            # Pandas Series比較を安全に行う - boolに変換してから評価
            has_null = bool(result_df[col].isnull().any())
            if has_null:
                # 最頻値で補完
                mode_value = result_df[col].mode()
                if not mode_value.empty:
                    result_df[col] = result_df[col].fillna(mode_value.iloc[0])

        return result_df

    def _clip_extended_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張データの範囲クリップ処理"""
        result_df = df.copy()

        # funding_rateの範囲クリップ (-1から1)
        if "funding_rate" in result_df.columns:
            # NaNとinfを処理してからクリップ
            funding_rate_clean = result_df["funding_rate"].replace(
                [np.inf, -np.inf], np.nan
            )
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
            oi_clean = result_df["open_interest"].replace([np.inf, -np.inf], np.nan)
            # Pandas Series比較を安全に行う
            before_count = (oi_clean < 0).sum()
            if before_count > 0:
                result_df["open_interest"] = np.maximum(oi_clean.fillna(0), 0)

        return result_df


# グローバルインスタンス（後方互換性維持）
data_processor = DataProcessor()


