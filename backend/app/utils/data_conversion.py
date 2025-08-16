"""
データ変換ユーティリティ（簡素化版）

pandas標準機能を活用し、冗長なカスタム実装を削除。
必要最小限の変換ロジックのみを提供。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """データ変換エラー"""


class OHLCVDataConverter:
    """OHLCV データ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        ohlcv_data: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のOHLCVデータをデータベース形式に変換

        Args:
            ohlcv_data: CCXT形式のOHLCVデータ
            symbol: シンボル
            timeframe: 時間軸

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for candle in ohlcv_data:
            timestamp_ms, open_price, high, low, close, volume = candle

            # ミリ秒タイムスタンプをdatetimeに変換
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

            db_record = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }

            db_records.append(db_record)

        return db_records

    @staticmethod
    def db_to_api_format(ohlcv_records: List[Any]) -> List[List]:
        """
        データベース形式のOHLCVデータをAPI形式に変換

        Args:
            ohlcv_records: データベースのOHLCVレコード

        Returns:
            API形式のOHLCVデータ
        """
        api_data = []

        for record in ohlcv_records:
            api_data.append(
                [
                    int(
                        record.timestamp.timestamp() * 1000
                    ),  # タイムスタンプ（ミリ秒）
                    record.open,
                    record.high,
                    record.low,
                    record.close,
                    record.volume,
                ]
            )

        return api_data


class FundingRateDataConverter:
    """ファンディングレートデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        funding_rate_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のファンディングレートデータをデータベース形式に変換

        Args:
            funding_rate_data: CCXT形式のファンディングレートデータ
            symbol: シンボル

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for rate_data in funding_rate_data:
            # データタイムスタンプの処理
            data_timestamp = rate_data.get("datetime")
            if data_timestamp:
                if isinstance(data_timestamp, str):
                    data_timestamp = datetime.fromisoformat(
                        data_timestamp.replace("Z", "+00:00")
                    )
                elif isinstance(data_timestamp, (int, float)):
                    data_timestamp = datetime.fromtimestamp(
                        data_timestamp / 1000, tz=timezone.utc
                    )

            db_record = {
                "symbol": symbol,
                "funding_rate": float(rate_data.get("fundingRate", 0.0)),
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }

            # 次回ファンディング時刻の処理
            next_funding = rate_data.get("nextFundingDatetime")
            if next_funding:
                if isinstance(next_funding, str):
                    db_record["next_funding_timestamp"] = datetime.fromisoformat(
                        next_funding.replace("Z", "+00:00")
                    )
                elif isinstance(next_funding, (int, float)):
                    db_record["next_funding_timestamp"] = datetime.fromtimestamp(
                        next_funding / 1000, tz=timezone.utc
                    )

            db_records.append(db_record)

        return db_records


class OpenInterestDataConverter:
    """オープンインタレストデータ変換の共通ヘルパークラス"""

    @staticmethod
    def ccxt_to_db_format(
        open_interest_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のオープンインタレストデータをデータベース形式に変換

        Args:
            open_interest_data: CCXT形式のオープンインタレストデータ
            symbol: シンボル

        Returns:
            データベース挿入用の辞書リスト
        """
        db_records = []

        for oi_data in open_interest_data:
            # データタイムスタンプの処理
            data_timestamp = None
            timestamp_value = oi_data.get("timestamp")

            if timestamp_value:
                if isinstance(timestamp_value, (int, float)):
                    data_timestamp = datetime.fromtimestamp(
                        timestamp_value / 1000, tz=timezone.utc
                    )
                elif isinstance(timestamp_value, str):
                    data_timestamp = datetime.fromisoformat(
                        timestamp_value.replace("Z", "+00:00")
                    )

            # オープンインタレスト値の処理（複数のフィールドを確認）
            open_interest_value = None

            # 可能性のあるフィールド名を順番に確認
            for field_name in [
                "openInterestAmount",
                "openInterest",
                "openInterestValue",
            ]:
                if field_name in oi_data:
                    value = oi_data[field_name]
                    if value is not None and value != 0:
                        open_interest_value = value
                        break

            # 値が見つからない場合は警告を出してスキップ
            if open_interest_value is None:
                logger.warning(
                    f"オープンインタレスト値が取得できませんでした: {oi_data}"
                )
                continue

            db_record = {
                "symbol": symbol,
                "open_interest_value": float(open_interest_value),
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }

            db_records.append(db_record)

        return db_records


class DataSanitizer:
    """データ検証・サニタイズの共通ヘルパークラス（旧DataValidator）"""

    @staticmethod
    def validate_ohlcv_data(ohlcv_records: List[Dict[str, Any]]) -> bool:
        """
        OHLCVデータの検証

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

    @staticmethod
    def validate_fear_greed_data(fear_greed_records: List[Dict[str, Any]]) -> bool:
        """
        Fear & Greed Indexデータの検証

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

    @staticmethod
    def sanitize_ohlcv_data(
        ohlcv_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        OHLCVデータをサニタイズ

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
            raise DataConversionError(f"OHLCVデータのサニタイズに失敗しました: {e}")


def ensure_list(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> list:
    """
    データをlistに変換（簡素化版）

    Python標準のlist()コンストラクタを活用。
    """
    try:
        if isinstance(data, list):
            return data

        # pandas/numpyは.tolist()で効率的に変換
        if hasattr(data, "tolist"):
            return data.tolist()

        # backtesting._Arrayの特殊ケース
        if hasattr(data, "_data"):
            return list(data._data)

        # その他は全てPython標準で処理
        return list(data)

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"list変換に失敗: {e}")
        else:
            logger.warning(f"list変換に失敗: {e}")
            return []


# 後方互換性のためのエイリアス
DataValidator = DataSanitizer
