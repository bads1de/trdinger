"""
データ変換処理の共通ユーティリティ
"""

import logging

from typing import List, Dict, Any
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


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
            # 必要なフィールドを抽出
            funding_timestamp = rate_data.get("datetime")
            if isinstance(funding_timestamp, str):
                funding_timestamp = datetime.fromisoformat(
                    funding_timestamp.replace("Z", "+00:00")
                )
            elif isinstance(funding_timestamp, (int, float)):
                funding_timestamp = datetime.fromtimestamp(
                    funding_timestamp / 1000, tz=timezone.utc
                )

            db_record = {
                "symbol": symbol,
                "funding_rate": float(rate_data.get("fundingRate", 0)),
                "funding_timestamp": funding_timestamp,
                "timestamp": datetime.now(timezone.utc),
                "next_funding_timestamp": None,  # 必要に応じて設定
                "mark_price": (
                    float(rate_data.get("markPrice", 0))
                    if rate_data.get("markPrice")
                    else None
                ),
                "index_price": (
                    float(rate_data.get("indexPrice", 0))
                    if rate_data.get("indexPrice")
                    else None
                ),
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

        logger.info(f"オープンインタレストデータ変換開始: {len(open_interest_data)}件")

        for oi_data in open_interest_data:
            # タイムスタンプの処理
            data_timestamp = oi_data.get("timestamp")
            if isinstance(data_timestamp, str):
                data_timestamp = datetime.fromisoformat(
                    data_timestamp.replace("Z", "+00:00")
                )
            elif isinstance(data_timestamp, (int, float)):
                data_timestamp = datetime.fromtimestamp(
                    data_timestamp / 1000, tz=timezone.utc
                )

            # オープンインタレスト値の取得（openInterestValueを優先）
            open_interest_value = oi_data.get("openInterestValue")

            # openInterestValueがNoneの場合、infoから取得を試行
            if open_interest_value is None and "info" in oi_data:
                info_data = oi_data["info"]
                if "openInterest" in info_data:
                    try:
                        open_interest_value = float(info_data["openInterest"])
                    except (ValueError, TypeError):
                        pass

            # 値が取得できない場合はスキップ
            if open_interest_value is None:
                logger.warning(
                    f"オープンインタレスト値が取得できませんでした: {oi_data}"
                )
                continue

            logger.info(
                f"オープンインタレストデータを変換中: {oi_data} -> value={open_interest_value}"
            )

            db_record = {
                "symbol": symbol,
                "open_interest_value": float(open_interest_value),
                "data_timestamp": data_timestamp,
                "timestamp": datetime.now(timezone.utc),
            }

            db_records.append(db_record)

        return db_records


class DataValidator:
    """データ検証・サニタイズの共通ヘルパークラス"""

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
            logger.error(f"OHLCVデータのサニタイズ中にエラーが発生しました: {e}")
            raise

    @staticmethod
    def validate_ohlcv_data(ohlcv_records: List[Dict[str, Any]]) -> bool:
        """
        OHLCVデータの妥当性を検証

        Args:
            ohlcv_records: 検証するOHLCVデータのリスト

        Returns:
            全て有効な場合True、無効なデータがある場合False
        """
        try:
            for record in ohlcv_records:
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
                for field in required_fields:
                    if field not in record:
                        logger.error(
                            f"OHLCVデータに必須フィールド '{field}' が不足しています。"
                        )
                        return False

                # 価格データの妥当性確認
                open_price = float(record["open"])
                high = float(record["high"])
                low = float(record["low"])
                close = float(record["close"])
                volume = float(record["volume"])

                # 価格関係の検証
                if high < max(open_price, close) or low > min(open_price, close):
                    logger.error(
                        f"OHLCVデータの価格関係が無効です: open={open_price}, high={high}, low={low}, close={close}"
                    )
                    return False

                # 負の値の検証
                if any(x < 0 for x in [open_price, high, low, close, volume]):
                    logger.error("OHLCVデータに負の値が含まれています。")
                    return False

            return True

        except Exception as e:
            logger.error(f"OHLCVデータの検証中にエラーが発生しました: {e}")
            return False

    @staticmethod
    def validate_fear_greed_data(fear_greed_records: List[Dict[str, Any]]) -> bool:
        """
        Fear & Greed Index データの妥当性を検証

        Args:
            fear_greed_records: 検証するFear & Greed Indexデータのリスト

        Returns:
            全て有効な場合True、無効なデータがある場合False
        """
        try:
            for record in fear_greed_records:
                # 必須フィールドの存在確認
                required_fields = [
                    "value",
                    "value_classification",
                    "data_timestamp",
                    "timestamp",
                ]
                for field in required_fields:
                    if field not in record:
                        logger.error(
                            f"Fear & Greed Indexデータに必須フィールド '{field}' が不足しています。"
                        )
                        return False

                # 値の妥当性確認
                value = int(record["value"])
                value_classification = record["value_classification"]

                # 値の範囲確認（0-100）
                if not (0 <= value <= 100):
                    logger.error(
                        f"Fear & Greed Index値が範囲外です: {value} (0-100の範囲である必要があります)"
                    )
                    return False

                # 分類の妥当性確認
                valid_classifications = [
                    "Extreme Fear",
                    "Fear",
                    "Neutral",
                    "Greed",
                    "Extreme Greed",
                ]
                if value_classification not in valid_classifications:
                    logger.error(
                        f"Fear & Greed Index分類が無効です: {value_classification} "
                        f"(有効な値: {', '.join(valid_classifications)})"
                    )
                    return False

                # タイムスタンプの妥当性確認
                from datetime import datetime

                try:
                    if isinstance(record["data_timestamp"], str):
                        datetime.fromisoformat(
                            record["data_timestamp"].replace("Z", "+00:00")
                        )
                    if isinstance(record["timestamp"], str):
                        datetime.fromisoformat(
                            record["timestamp"].replace("Z", "+00:00")
                        )
                except (ValueError, TypeError) as e:
                    logger.error(
                        f"Fear & Greed Indexデータのタイムスタンプが無効です: {e}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Fear & Greed Indexデータの検証中にエラーが発生しました: {e}")
            return False

    @staticmethod
    def validate_external_market_data(
        external_market_records: List[Dict[str, Any]],
    ) -> bool:
        """
        外部市場データの妥当性を検証

        Args:
            external_market_records: 検証する外部市場データのリスト

        Returns:
            全て有効な場合True、無効なデータがある場合False
        """
        try:
            for record in external_market_records:
                # 必須フィールドの存在確認
                required_fields = [
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "data_timestamp",
                    "timestamp",
                ]
                for field in required_fields:
                    if field not in record:
                        logger.error(
                            f"外部市場データに必須フィールド '{field}' が不足しています。"
                        )
                        return False

                # 数値フィールドの妥当性確認
                try:
                    open_price = float(record["open"])
                    high_price = float(record["high"])
                    low_price = float(record["low"])
                    close_price = float(record["close"])

                    # volumeは任意フィールド（VIXなどでは存在しない場合がある）
                    volume = None
                    if "volume" in record and record["volume"] is not None:
                        volume = float(record["volume"])

                except (ValueError, TypeError) as e:
                    logger.error(f"外部市場データの数値変換エラー: {e}")
                    return False

                # 価格の論理的妥当性確認
                if high_price < max(open_price, close_price, low_price):
                    logger.error(
                        f"高値が他の価格より低いです: high={high_price}, "
                        f"open={open_price}, close={close_price}, low={low_price}"
                    )
                    return False

                if low_price > min(open_price, close_price, high_price):
                    logger.error(
                        f"安値が他の価格より高いです: low={low_price}, "
                        f"open={open_price}, close={close_price}, high={high_price}"
                    )
                    return False

                # 負の値の検証（価格は正の値である必要がある）
                if any(
                    x <= 0 for x in [open_price, high_price, low_price, close_price]
                ):
                    logger.error("外部市場データの価格に0以下の値が含まれています。")
                    return False

                # 出来高の検証（存在する場合）
                if volume is not None and volume < 0:
                    logger.error("外部市場データの出来高に負の値が含まれています。")
                    return False

                # シンボルの妥当性確認
                symbol = str(record["symbol"]).strip()
                if not symbol:
                    logger.error("外部市場データのシンボルが空です。")
                    return False

                # 有効なシンボルの確認（計画書に記載されたシンボル）
                valid_symbols = ["^GSPC", "^IXIC", "DX-Y.NYB", "^VIX"]
                if symbol not in valid_symbols:
                    logger.warning(
                        f"未知のシンボルです: {symbol} "
                        f"(有効なシンボル: {', '.join(valid_symbols)})"
                    )
                    # 警告のみで処理は継続

                # タイムスタンプの妥当性確認
                from datetime import datetime

                try:
                    if isinstance(record["data_timestamp"], str):
                        datetime.fromisoformat(
                            record["data_timestamp"].replace("Z", "+00:00")
                        )
                    if isinstance(record["timestamp"], str):
                        datetime.fromisoformat(
                            record["timestamp"].replace("Z", "+00:00")
                        )
                except (ValueError, TypeError) as e:
                    logger.error(f"外部市場データのタイムスタンプが無効です: {e}")
                    return False

            return True

        except Exception as e:
            logger.error(f"外部市場データの検証中にエラーが発生しました: {e}")
            return False
