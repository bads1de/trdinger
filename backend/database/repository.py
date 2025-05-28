"""
データアクセス層（リポジトリパターン）
"""

from typing import List, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, func
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
import logging

from .models import OHLCVData, DataCollectionLog, FundingRateData
from .connection import get_db, ensure_db_initialized

logger = logging.getLogger(__name__)


class OHLCVRepository:
    """OHLCV データのリポジトリクラス"""

    def __init__(self, db: Session):
        self.db = db

    def insert_ohlcv_data(self, ohlcv_records: List[dict]) -> int:
        """
        OHLCV データを一括挿入（重複は無視）

        Args:
            ohlcv_records: OHLCV データのリスト

        Returns:
            挿入された件数
        """
        if not ohlcv_records:
            return 0

        try:
            # データベースタイプに応じて重複処理を切り替え
            from database.connection import DATABASE_URL

            if "sqlite" in DATABASE_URL.lower():
                # SQLiteの場合は一件ずつINSERT OR IGNOREで処理
                inserted_count = 0
                for record in ohlcv_records:
                    try:
                        ohlcv_obj = OHLCVData(**record)
                        self.db.add(ohlcv_obj)
                        self.db.commit()
                        inserted_count += 1
                    except Exception:
                        # 重複エラーの場合はロールバックして続行
                        self.db.rollback()
                        continue
            else:
                # PostgreSQL の ON CONFLICT を使用して重複を無視
                stmt = insert(OHLCVData).values(ohlcv_records)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["symbol", "timeframe", "timestamp"]
                )
                result = self.db.execute(stmt)
                self.db.commit()
                inserted_count = result.rowcount

            logger.info(f"OHLCV データを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"OHLCV データ挿入エラー: {e}")
            raise

    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[OHLCVData]:
        """
        OHLCV データを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            OHLCV データのリスト
        """
        try:
            query = self.db.query(OHLCVData).filter(
                and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
            )

            if start_time:
                query = query.filter(OHLCVData.timestamp >= start_time)

            if end_time:
                query = query.filter(OHLCVData.timestamp <= end_time)

            # 時系列順でソート
            query = query.order_by(asc(OHLCVData.timestamp))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"OHLCV データ取得エラー: {e}")
            raise

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボルと時間軸の最新タイムスタンプを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            最新のタイムスタンプ、データが存在しない場合はNone
        """
        try:
            result = (
                self.db.query(func.max(OHLCVData.timestamp))
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .scalar()
            )

            return result

        except Exception as e:
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            raise

    def count_records(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボルと時間軸のレコード数をカウント

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            レコード数
        """
        try:
            count = (
                self.db.query(OHLCVData)
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .count()
            )

            return count

        except Exception as e:
            logger.error(f"レコード数カウントエラー: {e}")
            raise

    def get_data_count(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボルと時間軸のデータ件数を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            データ件数
        """
        return self.count_records(symbol, timeframe)

    def get_oldest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボルと時間軸の最古タイムスタンプを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            最古のタイムスタンプ、データが存在しない場合はNone
        """
        try:
            result = (
                self.db.query(func.min(OHLCVData.timestamp))
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .scalar()
            )

            return result

        except Exception as e:
            logger.error(f"最古タイムスタンプ取得エラー: {e}")
            raise

    def validate_ohlcv_data(self, ohlcv_records: List[dict]) -> bool:
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
                    logger.error(f"負の値が含まれています")
                    return False

            return True

        except Exception as e:
            logger.error(f"データ検証エラー: {e}")
            return False

    def sanitize_ohlcv_data(self, ohlcv_records: List[dict]) -> List[dict]:
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

    def get_ohlcv_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        OHLCV データをDataFrameとして取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            OHLCV データのDataFrame
        """
        records = self.get_ohlcv_data(symbol, timeframe, start_time, end_time, limit)

        if not records:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        data = []
        for record in records:
            data.append(
                {
                    "timestamp": record.timestamp,
                    "open": record.open,
                    "high": record.high,
                    "low": record.low,
                    "close": record.close,
                    "volume": record.volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボル・時間軸の最新タイムスタンプを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            最新のタイムスタンプ（データがない場合はNone）
        """
        try:
            result = (
                self.db.query(func.max(OHLCVData.timestamp))
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .scalar()
            )

            return result

        except Exception as e:
            logger.error(f"最新タイムスタンプ取得エラー: {e}")
            raise

    def get_data_count(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボル・時間軸のデータ件数を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            データ件数
        """
        try:
            count = (
                self.db.query(OHLCVData)
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .count()
            )

            return count

        except Exception as e:
            logger.error(f"データ件数取得エラー: {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """
        利用可能なシンボルのリストを取得

        Returns:
            シンボルのリスト
        """
        try:
            symbols = self.db.query(OHLCVData.symbol).distinct().all()
            return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"利用可能シンボル取得エラー: {e}")
            raise

    def get_oldest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボル・時間軸の最古タイムスタンプを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            最古のタイムスタンプ（データがない場合はNone）
        """
        try:
            result = (
                self.db.query(func.min(OHLCVData.timestamp))
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .scalar()
            )

            return result

        except Exception as e:
            logger.error(f"最古タイムスタンプ取得エラー: {e}")
            raise

    def get_date_range(
        self, symbol: str, timeframe: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        指定されたシンボル・時間軸のデータ期間を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        try:
            result = (
                self.db.query(
                    func.min(OHLCVData.timestamp), func.max(OHLCVData.timestamp)
                )
                .filter(
                    and_(OHLCVData.symbol == symbol, OHLCVData.timeframe == timeframe)
                )
                .first()
            )

            return result if result else (None, None)

        except Exception as e:
            logger.error(f"データ期間取得エラー: {e}")
            raise


class DataCollectionLogRepository:
    """データ収集ログのリポジトリクラス"""

    def __init__(self, db: Session):
        self.db = db

    def log_collection(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        records_collected: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> DataCollectionLog:
        """
        データ収集ログを記録

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 収集開始時刻
            end_time: 収集終了時刻
            records_collected: 収集件数
            status: ステータス
            error_message: エラーメッセージ

        Returns:
            作成されたログレコード
        """
        try:
            log_record = DataCollectionLog(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                records_collected=records_collected,
                status=status,
                error_message=error_message,
            )

            self.db.add(log_record)
            self.db.commit()
            self.db.refresh(log_record)

            logger.info(f"データ収集ログを記録しました: {symbol} {timeframe} {status}")
            return log_record

        except Exception as e:
            self.db.rollback()
            logger.error(f"データ収集ログ記録エラー: {e}")
            raise

    def get_recent_logs(self, limit: int = 100) -> List[DataCollectionLog]:
        """
        最近のデータ収集ログを取得

        Args:
            limit: 取得件数制限

        Returns:
            データ収集ログのリスト
        """
        try:
            return (
                self.db.query(DataCollectionLog)
                .order_by(desc(DataCollectionLog.created_at))
                .limit(limit)
                .all()
            )

        except Exception as e:
            logger.error(f"データ収集ログ取得エラー: {e}")
            raise


class FundingRateRepository:
    """ファンディングレートデータのリポジトリクラス"""

    def __init__(self, db: Session):
        self.db = db

    def insert_funding_rate_data(self, funding_rate_records: List[dict]) -> int:
        """
        ファンディングレートデータを一括挿入

        Args:
            funding_rate_records: ファンディングレートデータのリスト

        Returns:
            挿入された件数
        """
        if not funding_rate_records:
            logger.warning("挿入するファンディングレートデータがありません")
            return 0

        try:
            # データベースタイプに応じて重複処理を切り替え
            from database.connection import DATABASE_URL

            if "sqlite" in DATABASE_URL.lower():
                # SQLiteの場合は一件ずつINSERT OR IGNOREで処理
                inserted_count = 0
                for record in funding_rate_records:
                    try:
                        funding_rate_obj = FundingRateData(**record)
                        self.db.add(funding_rate_obj)
                        self.db.commit()
                        inserted_count += 1
                    except Exception:
                        # 重複エラーの場合はロールバックして続行
                        self.db.rollback()
                        continue
            else:
                # PostgreSQL の ON CONFLICT を使用して重複を無視
                stmt = insert(FundingRateData).values(funding_rate_records)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["symbol", "funding_timestamp"]
                )
                result = self.db.execute(stmt)
                self.db.commit()
                inserted_count = result.rowcount

            logger.info(f"ファンディングレートデータを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"ファンディングレートデータ挿入エラー: {e}")
            raise

    def get_funding_rate_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[FundingRateData]:
        """
        ファンディングレートデータを取得

        Args:
            symbol: 取引ペア
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            ファンディングレートデータのリスト
        """
        try:
            query = self.db.query(FundingRateData).filter(
                FundingRateData.symbol == symbol
            )

            if start_time:
                query = query.filter(FundingRateData.funding_timestamp >= start_time)

            if end_time:
                query = query.filter(FundingRateData.funding_timestamp <= end_time)

            # 時系列順でソート
            query = query.order_by(asc(FundingRateData.funding_timestamp))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"ファンディングレートデータ取得エラー: {e}")
            raise

    def get_latest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最新ファンディングタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最新のファンディングタイムスタンプ（データがない場合はNone）
        """
        try:
            result = (
                self.db.query(func.max(FundingRateData.funding_timestamp))
                .filter(FundingRateData.symbol == symbol)
                .scalar()
            )

            return result

        except Exception as e:
            logger.error(f"最新ファンディングタイムスタンプ取得エラー: {e}")
            raise

    def get_funding_rate_count(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータ件数を取得

        Args:
            symbol: 取引ペア

        Returns:
            データ件数
        """
        try:
            count = (
                self.db.query(FundingRateData)
                .filter(FundingRateData.symbol == symbol)
                .count()
            )

            return count

        except Exception as e:
            logger.error(f"ファンディングレートデータ件数取得エラー: {e}")
            raise

    def get_available_funding_symbols(self) -> List[str]:
        """
        利用可能なファンディングレートシンボルのリストを取得

        Returns:
            シンボルのリスト
        """
        try:
            symbols = self.db.query(FundingRateData.symbol).distinct().all()
            return [symbol[0] for symbol in symbols]

        except Exception as e:
            logger.error(f"利用可能ファンディングレートシンボル取得エラー: {e}")
            raise

    def get_funding_rate_dataframe(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        ファンディングレートデータをDataFrameとして取得

        Args:
            symbol: 取引ペア
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            ファンディングレートデータのDataFrame
        """
        records = self.get_funding_rate_data(symbol, start_time, end_time, limit)

        if not records:
            return pd.DataFrame(
                columns=[
                    "funding_timestamp",
                    "funding_rate",
                    "mark_price",
                    "index_price",
                ]
            )

        data = []
        for record in records:
            data.append(
                {
                    "funding_timestamp": record.funding_timestamp,
                    "funding_rate": record.funding_rate,
                    "mark_price": record.mark_price,
                    "index_price": record.index_price,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("funding_timestamp", inplace=True)
        return df
