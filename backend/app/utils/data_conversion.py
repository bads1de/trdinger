"""
データ変換ユーティリティ

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
            # CCXTのOHLCVデータは [timestamp, open, high, low, close, volume] の形式
            try:
                timestamp_ms, open_price, high, low, close, volume = candle
            except ValueError:
                logger.warning(f"OHLCVデータの要素数が不正: {candle} - スキップ")
                continue

            # ミリ秒タイムスタンプをdatetimeに変換（UTCタイムゾーンを明示）
            try:
                if isinstance(timestamp_ms, (int, float)):
                    timestamp = datetime.fromtimestamp(
                        timestamp_ms / 1000, tz=timezone.utc
                    )
                else:
                    raise TypeError(
                        f"タイムスタンプは数値である必要があります: {type(timestamp_ms)}"
                    )
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"不正なタイムスタンプ: {timestamp_ms} - スキップ。エラー: {e}"
                )
                continue

            # データベースレコードの辞書を作成
            try:
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
            except (TypeError, ValueError) as e:
                logger.warning(f"不正なOHLCV値: {candle} - スキップ。エラー: {e}")
                continue

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
                    float(record.open),
                    float(record.high),
                    float(record.low),
                    float(record.close),
                    float(record.volume),
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
            # データタイムスタンプの処理（CCXTのdatetimeフィールドを解析）
            data_timestamp = rate_data.get("datetime")
            if data_timestamp:
                if isinstance(data_timestamp, str):
                    try:
                        # ISO形式の文字列をdatetimeに変換（Zulu時間は+00:00に置換）
                        data_timestamp = datetime.fromisoformat(
                            data_timestamp.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"不正なdatetime文字列: {data_timestamp} - エラー: {e}"
                        )
                        data_timestamp = None
                elif isinstance(data_timestamp, (int, float)):
                    try:
                        # ミリ秒タイムスタンプをdatetimeに変換
                        data_timestamp = datetime.fromtimestamp(
                            data_timestamp / 1000, tz=timezone.utc
                        )
                    except (ValueError, TypeError, OSError) as e:
                        logger.warning(
                            f"不正なタイムスタンプ数値: {data_timestamp} - エラー: {e}"
                        )
                        data_timestamp = None
                else:
                    logger.warning(f"不明なdatetime形式: {type(data_timestamp)}")
                    data_timestamp = None

            # データベースレコードの辞書を作成
            db_record = {
                "symbol": symbol,
                "funding_rate": float(rate_data.get("fundingRate", 0.0)),
                "data_timestamp": data_timestamp,
                # レコード挿入時のタイムスタンプ
                "timestamp": datetime.now(timezone.utc),
            }

            # 次回ファンディング時刻の処理（存在する場合）
            next_funding = rate_data.get("nextFundingDatetime")
            if next_funding:
                try:
                    if isinstance(next_funding, str):
                        # ISO形式の文字列をdatetimeに変換
                        db_record["next_funding_timestamp"] = datetime.fromisoformat(
                            next_funding.replace("Z", "+00:00")
                        )
                    elif isinstance(next_funding, (int, float)):
                        # ミリ秒タイムスタンプをdatetimeに変換
                        db_record["next_funding_timestamp"] = datetime.fromtimestamp(
                            next_funding / 1000, tz=timezone.utc
                        )
                except (ValueError, TypeError, OSError) as e:
                    logger.warning(
                        f"不正なnext_funding値: {next_funding} - エラー: {e}"
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
            # データタイムスタンプの処理（CCXTのtimestampフィールドを解析）
            data_timestamp = None
            timestamp_value = oi_data.get("timestamp")

            if timestamp_value:
                if isinstance(timestamp_value, (int, float)):
                    try:
                        # ミリ秒タイムスタンプをdatetimeに変換
                        data_timestamp = datetime.fromtimestamp(
                            timestamp_value / 1000, tz=timezone.utc
                        )
                    except (ValueError, TypeError, OSError) as e:
                        logger.warning(
                            f"不正なタイムスタンプ数値: {timestamp_value} - エラー: {e}"
                        )
                        data_timestamp = None
                elif isinstance(timestamp_value, str):
                    try:
                        # ISO形式の文字列をdatetimeに変換
                        data_timestamp = datetime.fromisoformat(
                            timestamp_value.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"不正なdatetime文字列: {timestamp_value} - エラー: {e}"
                        )
                        data_timestamp = None
                else:
                    logger.warning(f"不明なtimestamp形式: {type(timestamp_value)}")
                    data_timestamp = None

            # オープンインタレスト値の処理（複数のフィールドを確認して取得）
            open_interest_value = None

            # 可能性のあるフィールド名を順番に確認（取引所によってフィールド名が異なるため）
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

            # データベースレコードの辞書を作成
            db_record = {
                "symbol": symbol,
                "open_interest_value": float(open_interest_value),
                "data_timestamp": data_timestamp,
                # レコード挿入時のタイムスタンプ
                "timestamp": datetime.now(timezone.utc),
            }

            db_records.append(db_record)

        return db_records


def ensure_list(
    data: Union[pd.Series, list, np.ndarray, Any],
    raise_on_error: bool = True,
) -> list:
    """
    データをlistに変換（簡素化版）

    Python標準のlist()コンストラクタを活用。
    pandas/numpyオブジェクトはtolist()で効率的に変換。
    backtesting.lib._Arrayなどの特殊な配列型にも対応。
    """
    try:
        if isinstance(data, list):
            return data

        # pandas/numpyおよびbacktesting.lib._Arrayはtolist()で効率的に変換
        if hasattr(data, "tolist"):
            return data.tolist()

        # numpy互換のdata属性がある場合（例: backtesting.lib._Array）
        if hasattr(data, "data"):
            try:
                return list(data.data)
            except (AttributeError, TypeError):
                # data属性がリストに変換できない場合
                pass

        # その他は全てPython標準のlist()コンストラクタで処理
        if data is None:
            return []  # Noneの場合は空リストを返す

        try:
            return list(data)
        except TypeError:
            # イテレートできない場合、空リストを返す
            logger.warning(f"イテレートできないデータをlistに変換: {type(data)}")
            return []

    except Exception as e:
        if raise_on_error:
            raise DataConversionError(f"list変換に失敗: {e}")
        else:
            logger.warning(f"list変換に失敗: {e}")
            return []
