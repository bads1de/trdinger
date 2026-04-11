"""
データ変換ユーティリティ

pandas標準機能を活用し、冗長なカスタム実装を削除。
必要最小限の変換ロジックのみを提供。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.utils.datetime_utils import parse_timestamp_safe as _parse_timestamp_safe

logger = logging.getLogger(__name__)


class DataConversionError(Exception):
    """
    データ変換エラー

    データ変換処理中に発生したエラーを表す例外クラスです。
    """


def parse_timestamp_safe(value: Any) -> Optional[datetime]:
    """
    タイムスタンプを安全にパースする（警告ログ付き）。

    datetime_utils.parse_timestamp_safe のラッパー関数で、
    変換失敗時に警告ログを出力します。

    Args:
        value: 変換対象のタイムスタンプ値（数値、文字列、datetime等）

    Returns:
        Optional[datetime]: 変換されたdatetimeオブジェクト（失敗時はNone）
    """
    result = _parse_timestamp_safe(value)
    if result is None and value is not None:
        logger.warning("タイムスタンプ変換エラー: %s - 型: %s", value, type(value))
    return result


class OHLCVDataConverter:
    """
    OHLCV データ変換の共通ヘルパークラス

    CCXT形式とデータベース形式の間でOHLCVデータを変換する
    静的メソッドを提供します。
    """

    @staticmethod
    def ccxt_to_db_format(
        ohlcv_data: List[List], symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のOHLCVデータをデータベース形式に変換

        CCXTライブラリから取得した [timestamp, open, high, low, close, volume]
        形式のリストを、データベース挿入用の辞書リストに変換します。
        不正なデータはスキップされ、警告ログが出力されます。

        Args:
            ohlcv_data: CCXT形式のOHLCVデータ（[timestamp, open, high, low, close, volume]のリスト）
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）
            timeframe: 時間軸（例: '1h', '1d'）

        Returns:
            List[Dict[str, Any]]: データベース挿入用の辞書リスト。
                                   各辞書は以下のキーを含みます：
                                   - symbol: シンボル
                                   - timeframe: 時間軸
                                   - timestamp: datetimeオブジェクト
                                   - open, high, low, close, volume: 数値
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
            timestamp = parse_timestamp_safe(timestamp_ms)
            if timestamp is None:
                logger.warning(f"不正なタイムスタンプ: {timestamp_ms} - スキップ")
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

        データベースから取得したOHLCVレコードを、
        [timestamp, open, high, low, close, volume] 形式のリストに変換します。
        タイムスタンプはミリ秒単位の数値に変換されます。

        Args:
            ohlcv_records: データベースのOHLCVレコード（timestamp、open、high、low、close、volume属性を持つオブジェクトのリスト）

        Returns:
            List[List]: API形式のOHLCVデータ（[timestamp_ms, open, high, low, close, volume]のリスト）
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
    """
    ファンディングレートデータ変換の共通ヘルパークラス

    CCXT形式とデータベース形式の間でファンディングレートデータを変換する
    静的メソッドを提供します。
    """

    @staticmethod
    def ccxt_to_db_format(
        funding_rate_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のファンディングレートデータをデータベース形式に変換

        CCXTライブラリから取得したファンディングレートデータを、
        データベース挿入用の辞書リストに変換します。

        Args:
            funding_rate_data: CCXT形式のファンディングレートデータ（辞書のリスト）
                            各辞書は 'fundingRate', 'datetime', 'nextFundingDatetime' 等のキーを含みます
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）

        Returns:
            List[Dict[str, Any]]: データベース挿入用の辞書リスト。
                                   各辞書は以下のキーを含みます：
                                   - symbol: シンボル
                                   - funding_rate: ファンディングレート（数値）
                                   - data_timestamp: データタイムスタンプ（datetime）
                                   - timestamp: レコード挿入時のタイムスタンプ（datetime）
                                   - next_funding_timestamp: 次回ファンディング時刻（存在する場合）
        """
        db_records = []

        for rate_data in funding_rate_data:
            # データタイムスタンプの処理（CCXTのdatetimeフィールドを解析）
            data_timestamp = parse_timestamp_safe(rate_data.get("datetime"))

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
                next_timestamp = parse_timestamp_safe(next_funding)
                if next_timestamp:
                    db_record["next_funding_timestamp"] = next_timestamp

            db_records.append(db_record)

        return db_records


class OpenInterestDataConverter:
    """
    オープンインタレストデータ変換の共通ヘルパークラス

    CCXT形式とデータベース形式の間でオープンインタレストデータを変換する
    静的メソッドを提供します。
    """

    @staticmethod
    def ccxt_to_db_format(
        open_interest_data: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        CCXT形式のオープンインタレストデータをデータベース形式に変換

        CCXTライブラリから取得したオープンインタレストデータを、
        データベース挿入用の辞書リストに変換します。
        取引所によってフィールド名が異なるため、複数の候補フィールドを確認します。

        Args:
            open_interest_data: CCXT形式のオープンインタレストデータ（辞書のリスト）
                                各辞書は 'openInterestAmount', 'openInterest', 'openInterestValue',
                                'timestamp' 等のキーを含みます
            symbol: 取引ペアシンボル（例: 'BTC/USDT:USDT'）

        Returns:
            List[Dict[str, Any]]: データベース挿入用の辞書リスト。
                                   各辞書は以下のキーを含みます：
                                   - symbol: シンボル
                                   - open_interest_value: オープンインタレスト値（数値）
                                   - data_timestamp: データタイムスタンプ（datetime）
                                   - timestamp: レコード挿入時のタイムスタンプ（datetime）

        Note:
            オープンインタレスト値が見つからないレコードはスキップされ、警告ログが出力されます。
        """
        db_records = []

        for oi_data in open_interest_data:
            # データタイムスタンプの処理（CCXTのtimestampフィールドを解析）
            data_timestamp = parse_timestamp_safe(oi_data.get("timestamp"))

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


def normalize_market_symbol(symbol: Any) -> str:
    """
    市場シンボルを正規化する

    様々な形式のシンボル表記をBybitの標準形式（BTC/USDT:USDT）に正規化します。
    BybitService._normalize_symbol_for_ccxt および
    BaseDataCollectionOrchestrationService._normalize_derivative_symbol
    の共通化を目的とする。

    Args:
        symbol: 正規化するシンボル（例: "BTC/USDT", "BTCUSDT", "BTC/USDT:USDT"）

    Returns:
        str: 正規化されたシンボル（例: "BTC/USDT:USDT"）

    変換ルール:
        - Noneまたは空文字: 空文字を返す
        - 非文字列: str()で文字列化
        - 既にコロン付き（例: BTC/USDT:USDT）: そのまま返す
        - ハイフンを含む（例: BTC-USDT）: スラッシュに変換
        - スラッシュで終わるUSDT/USD/BUSD/USDC: コロン+同じ通貨を追加
        - スラッシュなし（例: BTCUSDT）: コロン+USDTを追加
    """
    if symbol is None:
        return ""

    # 非文字列入力はstr()変換
    if not isinstance(symbol, str):
        symbol = str(symbol)

    # 空文字列の場合はそのまま返す
    if not symbol:
        return symbol

    # 大文字に変換
    normalized = symbol.upper().strip()

    # すでにコロン付きの場合はそのまま返す（例: "BTC/USDT:USDT"）
    if ":" in normalized:
        return normalized

    # ハイフンをスラッシュに変換
    normalized = normalized.replace("-", "/")

    # スラッシュ付きシンボルの処理（例: "BTC/USDT" -> "BTC/USDT:USDT"）
    if "/" in normalized:
        if normalized.endswith("/USDT"):
            return f"{normalized}:USDT"
        elif normalized.endswith("/USD"):
            return f"{normalized}:USD"
        elif normalized.endswith("/BUSD"):
            return f"{normalized}:BUSD"
        elif normalized.endswith("/USDC"):
            return f"{normalized}:USDC"
        # その他のQuote通貨
        return normalized

    # スラッシュがない場合（例: "BTCUSDT" -> "BTCUSDT:USDT"）
    # その他のQuote通貨
    return f"{normalized}:USDT"
