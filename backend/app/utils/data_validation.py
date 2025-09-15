"""
データバリデーションユーティリティ

特徴量データの妥当性チェックとクリーンアップ機能を提供します。
無限大値、NaN値、異常に大きな値の検出と処理を行います。
"""

import logging
from datetime import datetime, timezone

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from typing import Dict as TypingDict
from pydantic import BaseModel, Field

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


# バリデーション設定
OHLCV_VALIDATION_CONFIG = {
    "Open": {"min": 0, "max": 1e6, "required": True},
    "High": {"min": 0, "max": 1e6, "required": True},
    "Low": {"min": 0, "max": 1e6, "required": True},
    "Close": {"min": 0, "max": 1e6, "required": True},
    "Volume": {"min": 0, "max": 1e12, "required": True},
}

EXTENDED_MARKET_DATA_VALIDATION_CONFIG = {
    "Open": {"min": 0, "max": 1e6, "required": True},
    "High": {"min": 0, "max": 1e6, "required": True},
    "Low": {"min": 0, "max": 1e6, "required": True},
    "Close": {"min": 0, "max": 1e6, "required": True},
    "Volume": {"min": 0, "max": 1e12, "required": True},
    "open_interest": {"min": 0, "required": False},
    "funding_rate": {"min": -1, "max": 1, "required": False},
}


def validate_dataframe_with_config(
    df: pd.DataFrame, config: TypingDict[str, TypingDict[str, Any]], lazy: bool = True
) -> Tuple[bool, List[str]]:
    """
    設定ベースのDataFrameバリデーション

    Args:
        df: 検証対象のDataFrame
        config: バリデーション設定

    Returns:
        (検証成功フラグ, エラーメッセージリスト)
    """
    try:
        error_messages = []

        # 必須カラムの存在確認
        for col_name, col_config in config.items():
            if col_config.get("required", True) and col_name not in df.columns:
                error_messages.append(f"必須カラム '{col_name}' が存在しません")

        if error_messages:
            return False, error_messages

        # 各カラムの値検証
        for col_name, col_config in config.items():
            if col_name not in df.columns:
                continue

            col_data = df[col_name]

            # 型チェック
            if col_config.get("type") == "str":
                if not pd.api.types.is_string_dtype(col_data):
                    error_messages.append(
                        f"'{col_name}' は文字列型である必要があります"
                    )
                continue

            # 数値チェック
            if not pd.api.types.is_numeric_dtype(col_data):
                error_messages.append(f"'{col_name}' は数値型である必要があります")
                continue

            # NaNチェック（必須カラムの場合）
            if col_config.get("required", True):
                nan_count = col_data.isna().sum()
                if nan_count > 0:
                    error_messages.append(
                        f"'{col_name}' にNaN値が{nan_count}個含まれています"
                    )

            # 範囲チェック
            if "min" in col_config:
                min_violations = (col_data < col_config["min"]).sum()
                if min_violations > 0:
                    error_messages.append(
                        f"'{col_name}' に最小値違反が{min_violations}個あります（最小: {col_config['min']}）"
                    )

            if "max" in col_config:
                max_violations = (col_data > col_config["max"]).sum()
                if max_violations > 0:
                    error_messages.append(
                        f"'{col_name}' に最大値違反が{max_violations}個あります（最大: {col_config['max']}）"
                    )

        return len(error_messages) == 0, error_messages

    except Exception as e:
        return False, [f"予期しないエラー: {str(e)}"]


def clean_dataframe_with_config(
    df: pd.DataFrame,
    config: TypingDict[str, TypingDict[str, Any]],
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    """
    設定ベースのDataFrameクリーニング

    Args:
        df: クリーニング対象のDataFrame
        config: バリデーション設定
        drop_invalid_rows: 無効な行を削除するか

    Returns:
        クリーニング済みのDataFrame
    """
    try:
        cleaned_df: pd.DataFrame = df.copy()

        # 各カラムのクリーニング
        for col_name, col_config in config.items():
            if col_name not in cleaned_df.columns:
                if col_config.get("required", True):
                    logger.warning(f"必須カラム '{col_name}' が存在しません")
                continue

            col_data = cleaned_df[col_name]

            # 型変換
            if col_config.get("type") == "str":
                cleaned_df.loc[:, col_name] = col_data.astype(str)
            else:
                # 数値型に変換（エラーの場合はNaN）
                numeric_data = pd.to_numeric(col_data, errors="coerce")
                # 明示的にfloat64型に変換
                cleaned_df.loc[:, col_name] = numeric_data.astype("float64")

            # 範囲外の値をクリップ
            if "min" in col_config:
                min_val = col_config["min"]
                cleaned_df.loc[:, col_name] = np.where(
                    cleaned_df[col_name] < min_val,
                    min_val,
                    cleaned_df[col_name],
                )

            if "max" in col_config:
                max_val = col_config["max"]
                cleaned_df.loc[:, col_name] = np.where(
                    cleaned_df[col_name] > max_val,
                    max_val,
                    cleaned_df[col_name],
                )

        # 無効な行の削除
        if drop_invalid_rows:
            # 必須カラムのNaNを含む行を削除
            valid_mask: pd.Series = pd.Series([True] * len(cleaned_df))

            for col_name, col_config in config.items():
                if col_config.get("required", True) and col_name in cleaned_df.columns:
                    valid_mask &= ~cleaned_df[col_name].isna()

            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.loc[valid_mask].reset_index(drop=True)

            if len(cleaned_df) < original_len:
                logger.info(f"無効な行を削除: {original_len - len(cleaned_df)}行")

        return cleaned_df

    except Exception as e:
        logger.error(f"DataFrameクリーニングエラー: {e}")
        return df


# 後方互換性のためのラッパー関数
def validate_dataframe_with_schema(
    df: pd.DataFrame, schema: TypingDict[str, TypingDict[str, Any]], lazy: bool = True
) -> Tuple[bool, List[str]]:
    """
    後方互換性のためのラッパー関数
    Panderaスキーマの代わりにバリデーション設定を使用
    """
    return validate_dataframe_with_config(df, schema, lazy)


def clean_dataframe_with_schema(
    df: pd.DataFrame,
    schema: TypingDict[str, TypingDict[str, Any]],
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    """
    後方互換性のためのラッパー関数
    Panderaスキーマの代わりにバリデーション設定を使用
    """
    return clean_dataframe_with_config(df, schema, drop_invalid_rows)


# 互換性維持のための定数
OHLCV_SCHEMA = OHLCV_VALIDATION_CONFIG
EXTENDED_MARKET_DATA_SCHEMA = EXTENDED_MARKET_DATA_VALIDATION_CONFIG


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
                    (df["High"] < np.maximum(df["Open"], df["Close"]))
                ).sum()

                low_violations = (
                    (df["Low"] > np.minimum(df["Open"], df["Close"]))
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

    @classmethod
    def validate_ohlcv_records_simple(cls, ohlcv_records: List[Dict[str, Any]]) -> bool:
        """
        OHLCVデータの検証（シンプル版 - DataSanitizer からの移行）

        Args:
            ohlcv_records: 検証するOHLCVデータのリスト

        Returns:
            データが有効な場合True、無効な場合False
        """
        if not ohlcv_records or not isinstance(ohlcv_records, list):
            return False

        try:
            for record in ohlcv_records:
                if not isinstance(record, dict):
                    return False

                # 必須フィールドの存在確認
                required_fields = [
                    "symbol",
                    "timeframe",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
                if not all(field in record for field in required_fields):
                    return False

                # 数値フィールドの検証
                for field in ["open", "high", "low", "close", "volume"]:
                    try:
                        float(record[field])
                    except (ValueError, TypeError):
                        return False

                # タイムスタンプの検証
                timestamp = record["timestamp"]
                if isinstance(timestamp, datetime):
                    pass  # OK
                elif isinstance(timestamp, str):
                    # strの場合はISO形式かどうかチェック
                    try:
                        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        return False
                elif isinstance(timestamp, (int, float)):
                    # 数値の場合は妥当な範囲かチェック
                    try:
                        datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                    except (ValueError, TypeError, OSError):
                        return False
                else:
                    return False

            return True

        except Exception as e:
            logger.error(f"OHLCVデータ検証エラー: {e}")
            return False


    @classmethod
    def sanitize_ohlcv_data(
        cls, ohlcv_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        OHLCVデータをサニタイズ（DataSanitizer からの移行）

        Args:
            ohlcv_records: サニタイズするOHLCVデータのリスト

        Returns:
            サニタイズされたOHLCVデータのリスト
        """
        sanitized_records = []

        try:
            for record in ohlcv_records:
                sanitized_record = {}

                # シンボルの正規化
                sanitized_record["symbol"] = str(record["symbol"]).strip().upper()

                # 時間軸の正規化
                sanitized_record["timeframe"] = str(record["timeframe"]).strip().lower()

                # タイムスタンプの変換
                timestamp = record["timestamp"]
                if isinstance(timestamp, str):
                    sanitized_record["timestamp"] = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                elif isinstance(timestamp, datetime):
                    sanitized_record["timestamp"] = timestamp
                else:
                    sanitized_record["timestamp"] = datetime.fromtimestamp(
                        float(timestamp), tz=timezone.utc
                    )

                # 数値データの変換
                for field in ["open", "high", "low", "close", "volume"]:
                    sanitized_record[field] = float(record[field])

                sanitized_records.append(sanitized_record)

            return sanitized_records

        except Exception as e:
            logger.error(f"OHLCVデータのサニタイズエラー: {e}")
            raise ValueError(f"OHLCVデータのサニタイズに失敗しました: {e}")
