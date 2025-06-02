"""
OHLCV データのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, asc
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import OHLCVData
from app.core.utils.data_converter import DataValidator

logger = logging.getLogger(__name__)


class OHLCVRepository(BaseRepository):
    """OHLCV データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, OHLCVData)

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
            # データの検証
            if not DataValidator.validate_ohlcv_data(ohlcv_records):
                raise ValueError("無効なOHLCVデータが含まれています")

            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                ohlcv_records, ["symbol", "timeframe", "timestamp"]
            )

            logger.info(f"OHLCV データを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
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
        return super().get_latest_timestamp(
            "timestamp", {"symbol": symbol, "timeframe": timeframe}
        )

    def get_oldest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        指定されたシンボルと時間軸の最古タイムスタンプを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            最古のタイムスタンプ、データが存在しない場合はNone
        """
        return super().get_oldest_timestamp(
            "timestamp", {"symbol": symbol, "timeframe": timeframe}
        )

    def get_data_count(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボルと時間軸のデータ件数を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            データ件数
        """
        return super().get_record_count({"symbol": symbol, "timeframe": timeframe})

    def get_date_range(self, symbol: str, timeframe: str):
        """
        指定されたシンボルと時間軸のデータ期間を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        return super().get_date_range(
            "timestamp", {"symbol": symbol, "timeframe": timeframe}
        )

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

    def validate_ohlcv_data(self, ohlcv_data: List[dict]) -> bool:
        """
        OHLCV データの妥当性を検証

        Args:
            ohlcv_data: 検証するOHLCVデータのリスト

        Returns:
            有効な場合True、無効な場合False
        """
        return DataValidator.validate_ohlcv_data(ohlcv_data)

    def sanitize_ohlcv_data(self, ohlcv_data: List[dict]) -> List[dict]:
        """
        OHLCV データをサニタイズ

        Args:
            ohlcv_data: サニタイズするOHLCVデータのリスト

        Returns:
            サニタイズされたOHLCVデータのリスト
        """
        return DataValidator.sanitize_ohlcv_data(ohlcv_data)

    def count_records(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボルと時間軸のレコード数を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸

        Returns:
            レコード数
        """
        return self.get_data_count(symbol, timeframe)


# sanitize_ohlcv_data メソッドは app.core.utils.data_converter.DataValidator に移動されました
