"""
データ変換処理の共通ユーティリティ
"""

from typing import List, Dict, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class OHLCVDataConverter:
    """OHLCV データ変換の共通ヘルパークラス"""
    
    @staticmethod
    def ccxt_to_db_format(
        ohlcv_data: List[List],
        symbol: str,
        timeframe: str
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
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'open': float(open_price),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': float(volume)
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
            api_data.append([
                int(record.timestamp.timestamp() * 1000),  # タイムスタンプ（ミリ秒）
                record.open,
                record.high,
                record.low,
                record.close,
                record.volume,
            ])
        
        return api_data


class FundingRateDataConverter:
    """ファンディングレートデータ変換の共通ヘルパークラス"""
    
    @staticmethod
    def ccxt_to_db_format(
        funding_rate_data: List[Dict[str, Any]],
        symbol: str
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
            funding_timestamp = rate_data.get('datetime')
            if isinstance(funding_timestamp, str):
                funding_timestamp = datetime.fromisoformat(
                    funding_timestamp.replace('Z', '+00:00')
                )
            elif isinstance(funding_timestamp, (int, float)):
                funding_timestamp = datetime.fromtimestamp(
                    funding_timestamp / 1000, tz=timezone.utc
                )
            
            db_record = {
                'symbol': symbol,
                'funding_rate': float(rate_data.get('fundingRate', 0)),
                'funding_timestamp': funding_timestamp,
                'timestamp': datetime.now(timezone.utc),
                'next_funding_timestamp': None,  # 必要に応じて設定
                'mark_price': float(rate_data.get('markPrice', 0)) if rate_data.get('markPrice') else None,
                'index_price': float(rate_data.get('indexPrice', 0)) if rate_data.get('indexPrice') else None,
            }
            
            # 次回ファンディング時刻の処理
            next_funding = rate_data.get('nextFundingDatetime')
            if next_funding:
                if isinstance(next_funding, str):
                    db_record['next_funding_timestamp'] = datetime.fromisoformat(
                        next_funding.replace('Z', '+00:00')
                    )
                elif isinstance(next_funding, (int, float)):
                    db_record['next_funding_timestamp'] = datetime.fromtimestamp(
                        next_funding / 1000, tz=timezone.utc
                    )
            
            db_records.append(db_record)
        
        return db_records
    
    @staticmethod
    def db_to_api_format(funding_rate_records: List[Any]) -> List[Dict[str, Any]]:
        """
        データベース形式のファンディングレートデータをAPI形式に変換
        
        Args:
            funding_rate_records: データベースのファンディングレートレコード
            
        Returns:
            API形式のファンディングレートデータ
        """
        api_data = []
        
        for record in funding_rate_records:
            api_data.append({
                "symbol": record.symbol,
                "funding_rate": record.funding_rate,
                "funding_timestamp": record.funding_timestamp.isoformat(),
                "timestamp": record.timestamp.isoformat(),
                "next_funding_timestamp": (
                    record.next_funding_timestamp.isoformat()
                    if record.next_funding_timestamp
                    else None
                ),
                "mark_price": record.mark_price,
                "index_price": record.index_price,
            })
        
        return api_data


class OpenInterestDataConverter:
    """オープンインタレストデータ変換の共通ヘルパークラス"""
    
    @staticmethod
    def ccxt_to_db_format(
        open_interest_data: List[Dict[str, Any]],
        symbol: str
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
            data_timestamp = oi_data.get('timestamp')
            if isinstance(data_timestamp, str):
                data_timestamp = datetime.fromisoformat(
                    data_timestamp.replace('Z', '+00:00')
                )
            elif isinstance(data_timestamp, (int, float)):
                data_timestamp = datetime.fromtimestamp(
                    data_timestamp / 1000, tz=timezone.utc
                )

            # オープンインタレスト値の取得（openInterestValueを優先）
            open_interest_value = oi_data.get("openInterestValue")
            open_interest_amount = oi_data.get("openInterestAmount")

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
                logger.warning(f"オープンインタレスト値が取得できません: {oi_data}")
                continue

            logger.info(
                f"オープンインタレストデータ変換: {oi_data} -> value={open_interest_value}"
            )

            db_record = {
                'symbol': symbol,
                'open_interest': float(open_interest_value),
                'open_interest_amount': (
                    float(open_interest_amount)
                    if open_interest_amount is not None
                    else None
                ),
                'data_timestamp': data_timestamp,
                'timestamp': datetime.now(timezone.utc),
            }

            db_records.append(db_record)

        return db_records
    
    @staticmethod
    def db_to_api_format(open_interest_records: List[Any]) -> List[Dict[str, Any]]:
        """
        データベース形式のオープンインタレストデータをAPI形式に変換
        
        Args:
            open_interest_records: データベースのオープンインタレストレコード
            
        Returns:
            API形式のオープンインタレストデータ
        """
        api_data = []
        
        for record in open_interest_records:
            api_data.append({
                "symbol": record.symbol,
                "open_interest": record.open_interest,
                "data_timestamp": record.data_timestamp.isoformat(),
                "timestamp": record.timestamp.isoformat(),
            })
        
        return api_data


class DataValidator:
    """データ検証・サニタイズの共通ヘルパークラス"""

    @staticmethod
    def sanitize_ohlcv_data(ohlcv_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            logger.error(f"データサニタイズエラー: {e}")
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
                    "symbol", "timeframe", "timestamp",
                    "open", "high", "low", "close", "volume"
                ]
                for field in required_fields:
                    if field not in record:
                        logger.error(f"必須フィールド '{field}' が不足しています")
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
                        f"価格関係が無効です: open={open_price}, high={high}, low={low}, close={close}"
                    )
                    return False
                
                # 負の値の検証
                if any(x < 0 for x in [open_price, high, low, close, volume]):
                    logger.error("負の値が含まれています")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"データ検証エラー: {e}")
            return False
