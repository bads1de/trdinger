"""
データバリデーションユーティリティ

特徴量データの妥当性チェックとクリーンアップ機能を提供します。
無限大値、NaN値、異常に大きな値の検出と処理を行います。
"""

import logging
from datetime import datetime, timezone

from typing import Any, Dict, List


logger = logging.getLogger(__name__)


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
