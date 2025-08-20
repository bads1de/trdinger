"""
データバリデーションユーティリティ

特徴量データの妥当性チェックとクリーンアップ機能を提供します。
無限大値、NaN値、異常に大きな値の検出と処理を行います。
"""

import logging
from datetime import datetime, timezone

from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
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

    @classmethod
    def validate_ohlcv_records_simple(
        cls, ohlcv_records: List[Dict[str, Any]]
    ) -> bool:
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
                if not isinstance(timestamp, (datetime, str, int, float)):
                    return False

            return True

        except Exception as e:
            logger.error(f"OHLCVデータ検証エラー: {e}")
            return False

    @classmethod
    def validate_fear_greed_data(cls, fear_greed_records: List[Dict[str, Any]]) -> bool:
        """
        Fear & Greed Indexデータの検証（DataSanitizer からの移行）

        Args:
            fear_greed_records: 検証するFear & Greed Indexデータのリスト

        Returns:
            データが有効な場合True、無効な場合False
        """
        if not fear_greed_records or not isinstance(fear_greed_records, list):
            return False

        try:
            for record in fear_greed_records:
                if not isinstance(record, dict):
                    return False

                # 必須フィールドの存在確認
                required_fields = ["value", "value_classification", "data_timestamp"]
                if not all(field in record for field in required_fields):
                    return False

                # 値の検証
                try:
                    value = int(record["value"])
                    if not (0 <= value <= 100):
                        return False
                except (ValueError, TypeError):
                    return False

                # 分類の検証
                if not isinstance(record["value_classification"], str):
                    return False

                # タイムスタンプの検証
                timestamp = record["data_timestamp"]
                if not isinstance(timestamp, (datetime, str, int, float)):
                    return False

            return True

        except Exception as e:
            logger.error(f"Fear & Greed Indexデータ検証エラー: {e}")
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
