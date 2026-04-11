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
        """OHLCVデータの妥当性を検証する。

        各レコードが必須フィールド（timestamp, open, high, low, close, volume）
        を持ち、数値フィールドが有効な数値であることを確認します。

        検証項目:
            - 必須フィールドの存在（OHLCV_REQUIRED_FIELDS）
            - 数値フィールドの妥当性（OHLCV_NUMERIC_FIELDS）
            - タイムスタンプの有効性

        Args:
            ohlcv_records: 検証するOHLCVデータのリスト。
                各要素は辞書形式で、OHLCVレコードを表す。

        Returns:
            bool: 全てのレコードが有効な場合はTrue、
                1つでも無効なレコードがある場合はFalse。
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
        """OHLCVデータをサニタイズ（正規化・検証）する。

        入力レコードの各フィールドを以下の通り正規化します:
            - シンボル: 大文字に変換、空白を除去
            - 時間軸: 小文字に変換、空白を除去
            - タイムスタンプ: パースして datetime オブジェクトに変換
            - 数値フィールド（open, high, low, close, volume）: floatに変換

        Args:
            ohlcv_records: サニタイズするOHLCVデータのリスト。

        Returns:
            List[Dict[str, Any]]: サニタイズされたOHLCVデータのリスト。
                正規化されたフィールドを持つ辞書のリスト。

        Raises:
            ValueError: タイムスタンプのパースに失敗した場合、
                または数値フィールドの変換に失敗した場合。
        """
        sanitized_records = []

        try:
            for record in ohlcv_records:
                sanitized_record: Dict[str, Any] = {}

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


# 別名（data_collection.data_validator との区別用）
DataValidator = RecordValidator
