"""
データバリデーションユーティリティ

特徴量データの妥当性チェックとクリーンアップ機能を提供します。
無限大値、NaN値、異常に大きな値の検出と処理を行います。
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class OHLCVDataModel(BaseModel):
    """
    OHLCV データのPydanticモデル

    宣言的なスキーマ定義により、データの妥当性を保証。
    """

    Open: float = Field(gt=0, description="始値（正の値）")
    High: float = Field(gt=0, description="高値（正の値）")
    Low: float = Field(gt=0, description="安値（正の値）")
    Close: float = Field(gt=0, description="終値（正の値）")
    Volume: float = Field(ge=0, description="出来高（非負の値）")

    @validator("High")
    def high_must_be_highest(cls, v, values):
        """高値は他の価格より高いか等しい必要がある"""
        if "Open" in values and v < values["Open"]:
            raise ValueError("高値は始値以上である必要があります")
        if "Low" in values and v < values["Low"]:
            raise ValueError("高値は安値以上である必要があります")
        if "Close" in values and v < values["Close"]:
            raise ValueError("高値は終値以上である必要があります")
        return v

    @validator("Low")
    def low_must_be_lowest(cls, v, values):
        """安値は他の価格より低いか等しい必要がある"""
        if "Open" in values and v > values["Open"]:
            raise ValueError("安値は始値以下である必要があります")
        if "Close" in values and v > values["Close"]:
            raise ValueError("安値は終値以下である必要があります")
        return v


# Panderaスキーマ定義
OHLCV_SCHEMA = DataFrameSchema(
    {
        "Open": Column(
            float,
            checks=[
                Check.greater_than(0),
                Check.less_than(1e6),  # 異常に大きな値を除外
            ],
        ),
        "High": Column(float, checks=[Check.greater_than(0), Check.less_than(1e6)]),
        "Low": Column(float, checks=[Check.greater_than(0), Check.less_than(1e6)]),
        "Close": Column(float, checks=[Check.greater_than(0), Check.less_than(1e6)]),
        "Volume": Column(
            float, checks=[Check.greater_than_or_equal_to(0), Check.less_than(1e12)]
        ),
    },
    index=pa.Index("datetime64[ns]", name="timestamp"),
)

EXTENDED_MARKET_DATA_SCHEMA = DataFrameSchema(
    {
        "Open": Column(float, checks=[Check.greater_than(0)]),
        "High": Column(float, checks=[Check.greater_than(0)]),
        "Low": Column(float, checks=[Check.greater_than(0)]),
        "Close": Column(float, checks=[Check.greater_than(0)]),
        "Volume": Column(float, checks=[Check.greater_than_or_equal_to(0)]),
        "open_interest": Column(
            float, checks=[Check.greater_than_or_equal_to(0)], nullable=True
        ),
        "funding_rate": Column(
            float,
            checks=[
                Check.greater_than(-1),  # -100%以上
                Check.less_than(1),  # 100%未満
            ],
            nullable=True,
        ),
        "fear_greed_value": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=True,
        ),
        "fear_greed_classification": Column(str, nullable=True),
    },
    index=pa.Index("datetime64[ns]", name="timestamp"),
)


def validate_dataframe_with_schema(
    df: pd.DataFrame, schema: DataFrameSchema, lazy: bool = True
) -> Tuple[bool, List[str]]:
    """
    Panderaスキーマを使用したDataFrameバリデーション（推奨アプローチ）

    Args:
        df: 検証対象のDataFrame
        schema: Panderaスキーマ
        lazy: 遅延評価（全エラーを収集）

    Returns:
        (検証成功フラグ, エラーメッセージリスト)
    """
    try:
        schema.validate(df, lazy=lazy)
        return True, []
    except pa.errors.SchemaErrors as e:
        error_messages = []
        for error in e.schema_errors:
            error_messages.append(str(error))
        return False, error_messages
    except Exception as e:
        return False, [f"予期しないエラー: {str(e)}"]


def clean_dataframe_with_schema(
    df: pd.DataFrame, schema: DataFrameSchema, drop_invalid_rows: bool = True
) -> pd.DataFrame:
    """
    スキーマに基づくDataFrameクリーニング（推奨アプローチ）

    Args:
        df: クリーニング対象のDataFrame
        schema: Panderaスキーマ
        drop_invalid_rows: 無効な行を削除するか

    Returns:
        クリーニング済みのDataFrame
    """
    try:
        # スキーマに基づく型変換とバリデーション
        cleaned_df = schema.validate(df, lazy=False)
        return cleaned_df
    except pa.errors.SchemaError as e:
        logger.warning(f"スキーマエラー: {e}")
        if drop_invalid_rows:
            # 無効な行を特定して削除
            try:
                # 各行を個別に検証
                valid_rows = []
                for idx, row in df.iterrows():
                    try:
                        schema.validate(pd.DataFrame([row]), lazy=False)
                        valid_rows.append(idx)
                    except Exception:
                        # 個別行の検証で予期しない例外が出ても次行へ
                        continue

                cleaned_df = df.loc[valid_rows]
                logger.info(f"無効な行を削除: {len(df) - len(cleaned_df)}行")
                return cleaned_df
            except Exception as inner_e:
                logger.error(f"行削除中にエラー: {inner_e}")
                return df
        else:
            return df


class DataValidator:
    """
    データバリデーションクラス

    特徴量データの妥当性チェックとクリーンアップを行います。
    """

    # 異常値の閾値設定（金融データに適した範囲に調整）
    MAX_VALUE_THRESHOLD = 1e6  # 最大値の閾値
    MIN_VALUE_THRESHOLD = -1e6  # 最小値の閾値

    def __init__(self):
        """初期化"""

    # Deprecated safe_* helper methods removed.
    # Use pandas built-in operations instead, e.g.:
    #  - division: (a / b).replace([np.inf, -np.inf], np.nan).fillna(default)
    #  - pct_change: series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    #  - rolling mean/std: series.rolling(window, min_periods=1).mean()/std()
    #  - normalization: (series - series.rolling(...).mean()) / series.rolling(...).std()
    # These helpers were removed to simplify the codebase and rely on pandas semantics.

    @classmethod
    def validate_dataframe(
        cls, df: pd.DataFrame, column_names: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        DataFrameの妥当性チェック（非推奨）

        注意: この方法は非推奨です。代わりにvalidate_dataframe_with_schema()を使用してください。

        Args:
            df: チェック対象のDataFrame
            column_names: チェック対象のカラム名（Noneの場合は全カラム）

        Returns:
            (妥当性フラグ, 問題のあるカラムの詳細)
        """
        warnings.warn(
            "validate_dataframe()は非推奨です。validate_dataframe_with_schema()を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        if df is None or df.empty:
            return False, {"empty": ["DataFrame is empty or None"]}

        issues = {"infinite": [], "nan": [], "too_large": [], "too_small": []}

        check_columns = (
            column_names
            if column_names
            else df.select_dtypes(include=[np.number]).columns
        )

        for col in check_columns:
            if col not in df.columns:
                continue

            series = df[col]

            # 数値型のカラムのみチェック
            if not pd.api.types.is_numeric_dtype(series):
                continue

            # 無限大値のチェック（数値型のみ）
            try:
                if np.isinf(series).any():
                    issues["infinite"].append(col)
            except (TypeError, ValueError):
                # データ型が対応していない場合はスキップ
                continue

            # NaN値のチェック
            if series.isna().any():
                issues["nan"].append(col)

            # 異常に大きな値のチェック
            try:
                if (series > cls.MAX_VALUE_THRESHOLD).any():
                    issues["too_large"].append(col)
            except (TypeError, ValueError):
                # 比較できない型の場合はスキップ
                pass

            # 異常に小さな値のチェック
            try:
                if (series < cls.MIN_VALUE_THRESHOLD).any():
                    issues["too_small"].append(col)
            except (TypeError, ValueError):
                # 比較できない型の場合はスキップ
                pass

        is_valid = not any(issues.values())
        return is_valid, issues

    @classmethod
    def clean_dataframe(
        cls,
        df: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        fill_method: str = "median",
    ) -> pd.DataFrame:
        """
        DataFrameのクリーンアップ（非推奨）

        注意: この方法は非推奨です。代わりにclean_dataframe_with_schema()を使用してください。

        Args:
            df: クリーンアップ対象のDataFrame
            column_names: クリーンアップ対象のカラム名（Noneの場合は全カラム）
            fill_method: 欠損値の補完方法（median, mean, zero, forward_fill）

        Returns:
            クリーンアップされたDataFrame
        """
        warnings.warn(
            "clean_dataframe()は非推奨です。clean_dataframe_with_schema()を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        if df is None or df.empty:
            return df

        result_df = df.copy()
        check_columns = (
            column_names
            if column_names
            else df.select_dtypes(include=[np.number]).columns
        )

        for col in check_columns:
            if col not in result_df.columns:
                continue

            series = result_df[col]

            # 数値型のカラムのみ処理
            if not pd.api.types.is_numeric_dtype(series):
                continue

            try:
                # 無限大値を NaN に変換
                series = series.replace([np.inf, -np.inf], np.nan)

                # 異常に大きな値を NaN に変換
                series = series.where(
                    (series <= cls.MAX_VALUE_THRESHOLD)
                    & (series >= cls.MIN_VALUE_THRESHOLD),
                    np.nan,
                )
            except (TypeError, ValueError) as e:
                logger.warning(f"カラム {col} の処理中にエラー: {e}")
                continue

            # 欠損値の補完
            if fill_method == "median":
                fill_value = series.median()
            elif fill_method == "mean":
                fill_value = series.mean()
            elif fill_method == "zero":
                fill_value = 0.0
            elif fill_method == "forward_fill":
                series = series.ffill()
                fill_value = 0.0  # 最初の値がNaNの場合のフォールバック
            else:
                fill_value = 0.0

            # まだNaNが残っている場合は指定された値で補完
            if fill_method != "forward_fill":
                series = series.fillna(fill_value)
            else:
                series = series.fillna(fill_value)

            result_df[col] = series

        return result_df

    @classmethod
    def log_validation_results(
        cls,
        validation_results: Tuple[bool, Dict[str, List[str]]],
        dataframe_name: str = "DataFrame",
    ) -> None:
        """
        バリデーション結果をログ出力

        Args:
            validation_results: validate_dataframeの結果
            dataframe_name: DataFrameの名前（ログ用）
        """
        is_valid, issues = validation_results

        if is_valid:
            logger.debug(f"{dataframe_name}: データは正常です")
        else:
            logger.warning(f"{dataframe_name}: データに問題があります")

            for issue_type, columns in issues.items():
                if columns:
                    logger.warning(f"  {issue_type}: {columns}")

    @classmethod
    def validate_and_clean(
        cls,
        df: pd.DataFrame,
        column_names: Optional[List[str]] = None,
        fill_method: str = "median",
        log_name: str = "DataFrame",
    ) -> pd.DataFrame:
        """
        バリデーションとクリーンアップを一括実行

        Args:
            df: 対象のDataFrame
            column_names: 対象カラム名
            fill_method: 補完方法
            log_name: ログ用の名前

        Returns:
            クリーンアップされたDataFrame
        """
        # バリデーション実行
        validation_results = cls.validate_dataframe(df, column_names)
        cls.log_validation_results(validation_results, log_name)

        # クリーンアップ実行
        cleaned_df = cls.clean_dataframe(df, column_names, fill_method)

        # クリーンアップ後の再バリデーション
        post_validation = cls.validate_dataframe(cleaned_df, column_names)
        if not post_validation[0]:
            logger.error(f"{log_name}: クリーンアップ後もデータに問題があります")
            cls.log_validation_results(
                post_validation, f"{log_name} (クリーンアップ後)"
            )

        return cleaned_df

    @classmethod
    def validate_ohlcv_data(
        cls, df: pd.DataFrame, strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        OHLCVデータの妥当性を検証（脆弱性修正）

        Args:
            df: OHLCVデータのDataFrame
            strict_mode: 厳密モード（デフォルト: False）

        Returns:
            検証結果の辞書
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "ohlc_violations": 0,
            "negative_volumes": 0,
            "missing_data_ratio": 0.0,
            "data_quality_score": 100.0,
        }

        try:
            if df.empty:
                validation_result["is_valid"] = False
                validation_result["errors"].append("データが空です")
                validation_result["data_quality_score"] = 0.0
                return validation_result

            # 必要なカラムの存在確認
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"必要なカラムが不足: {missing_columns}"
                )
                validation_result["data_quality_score"] -= 30.0

            # OHLC論理の検証
            if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                # High >= max(Open, Close) and Low <= min(Open, Close)
                high_violations = (
                    (df["High"] < df["Open"]) | (df["High"] < df["Close"])
                ).sum()

                low_violations = (
                    (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
                ).sum()

                total_violations = high_violations + low_violations
                validation_result["ohlc_violations"] = total_violations

                if total_violations > 0:
                    violation_ratio = total_violations / len(df)
                    if violation_ratio > 0.05:  # 5%以上の違反で無効
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(
                            f"OHLC論理違反が多すぎます: {total_violations}件 ({violation_ratio:.2%})"
                        )
                    else:
                        validation_result["warnings"].append(
                            f"OHLC論理違反: {total_violations}件"
                        )

                    validation_result["data_quality_score"] -= min(
                        50.0, violation_ratio * 100
                    )

            # ボリュームの検証
            if "Volume" in df.columns:
                negative_volumes = (df["Volume"] < 0).sum()
                validation_result["negative_volumes"] = negative_volumes

                if negative_volumes > 0:
                    neg_vol_ratio = negative_volumes / len(df)
                    if neg_vol_ratio > 0.01:  # 1%以上で警告
                        validation_result["warnings"].append(
                            f"負のボリューム: {negative_volumes}件"
                        )
                        validation_result["data_quality_score"] -= min(
                            20.0, neg_vol_ratio * 100
                        )

            # 欠損データの検証
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            validation_result["missing_data_ratio"] = missing_ratio

            if missing_ratio > 0.1:  # 10%以上の欠損で警告
                validation_result["warnings"].append(
                    f"欠損データが多い: {missing_ratio:.2%}"
                )
                validation_result["data_quality_score"] -= min(
                    30.0, missing_ratio * 100
                )

            # 厳密モードでの追加チェック
            if strict_mode:
                # 価格の異常値チェック
                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        extreme_values = (
                            (df[col] <= 0) | (df[col] > 1e6) | df[col].isnull()
                        ).sum()

                        if extreme_values > 0:
                            validation_result["warnings"].append(
                                f"{col}に異常値: {extreme_values}件"
                            )

            # 最終的な品質スコアの調整
            validation_result["data_quality_score"] = max(
                0.0, validation_result["data_quality_score"]
            )

            # 品質スコアが低すぎる場合は無効とする
            if validation_result["data_quality_score"] < 30.0:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"データ品質スコアが低すぎます: {validation_result['data_quality_score']:.1f}%"
                )

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"検証中にエラーが発生: {str(e)}")
            validation_result["data_quality_score"] = 0.0

        return validation_result
