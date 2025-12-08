"""
レコード形式（辞書リスト）のデータバリデーション

OHLCVデータなどの辞書リスト形式のバリデーションとサニタイズを提供します。
DataFrame形式のバリデーションは data_validator.py を参照してください。
"""

import logging
from typing import Any, Dict, List

from app.utils.data_conversion import parse_timestamp_safe

logger = logging.getLogger(__name__)


class RecordValidator:
    """
    レコード形式（辞書リスト）のバリデーションクラス

    OHLCVデータなどの辞書リストの妥当性チェックとサニタイズを行います。
    """

    OHLCV_REQUIRED_FIELDS = [
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    OHLCV_NUMERIC_FIELDS = ["open", "high", "low", "close", "volume"]

    @classmethod
    def validate_ohlcv_records_simple(cls, ohlcv_records: List[Dict[str, Any]]) -> bool:
        """
        OHLCVデータの妥当性を検証

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
                if not all(field in record for field in cls.OHLCV_REQUIRED_FIELDS):
                    return False

                # 数値フィールドの検証
                for field in cls.OHLCV_NUMERIC_FIELDS:
                    try:
                        float(record[field])
                    except (ValueError, TypeError):
                        return False

                # タイムスタンプの検証
                if parse_timestamp_safe(record["timestamp"]) is None:
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
        OHLCVデータをサニタイズ

        Args:
            ohlcv_records: サニタイズするOHLCVデータのリスト

        Returns:
            サニタイズされたOHLCVデータのリスト

        Raises:
            ValueError: サニタイズに失敗した場合
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
                timestamp = parse_timestamp_safe(record["timestamp"])
                if timestamp is None:
                    raise ValueError(f"無効なタイムスタンプ: {record['timestamp']}")
                sanitized_record["timestamp"] = timestamp

                # 数値データの変換
                for field in cls.OHLCV_NUMERIC_FIELDS:
                    sanitized_record[field] = float(record[field])

                sanitized_records.append(sanitized_record)

            return sanitized_records

        except Exception as e:
            logger.error(f"OHLCVデータのサニタイズエラー: {e}")
            raise ValueError(f"OHLCVデータのサニタイズに失敗しました: {e}")


# 後方互換性のためのエイリアス
DataValidator = RecordValidator
