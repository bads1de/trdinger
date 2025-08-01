"""
OHLCV データのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import OHLCVData
from app.utils.data_converter import DataValidator


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
                raise ValueError(
                    "挿入しようとしているOHLCVデータに無効なものが含まれています。"
                )

            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                ohlcv_records, ["symbol", "timeframe", "timestamp"]
            )

            logger.info(f"OHLCV データを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            logger.error(f"OHLCVデータの挿入中にエラーが発生しました: {e}")
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
        filters = {"symbol": symbol, "timeframe": timeframe}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> List[OHLCVData]:
        """
        最新のOHLCV データを取得（降順）

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            limit: 取得件数制限

        Returns:
            最新のOHLCV データのリスト（新しい順）
        """
        filters = {"symbol": symbol, "timeframe": timeframe}
        return self.get_latest_records(
            filters=filters,
            timestamp_column="timestamp",
            limit=limit,
        )

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

        column_mapping = {
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        return self.to_dataframe(
            records=records,
            column_mapping=column_mapping,
            index_column="timestamp",
        )

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

    def clear_all_ohlcv_data(self) -> int:
        """
        全てのOHLCVデータを削除

        Returns:
            削除された件数
        """
        deleted_count = self._delete_all_records()
        logger.info(f"全てのOHLCVデータを削除しました: {deleted_count}件")
        return deleted_count

    def clear_ohlcv_data_by_symbol(self, symbol: str) -> int:
        """
        指定されたシンボルのOHLCVデータを削除

        Args:
            symbol: 削除対象のシンボル

        Returns:
            削除された件数
        """
        deleted_count = self._delete_records_by_filter("symbol", symbol)
        logger.info(
            f"シンボル '{symbol}' のOHLCVデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_ohlcv_data_by_timeframe(self, timeframe: str) -> int:
        """
        指定された時間足のOHLCVデータを削除

        Args:
            timeframe: 削除対象の時間足

        Returns:
            削除された件数
        """
        deleted_count = self._delete_records_by_filter("timeframe", timeframe)
        logger.info(
            f"時間足 '{timeframe}' のOHLCVデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_ohlcv_data_by_date_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        指定された期間のOHLCVデータを削除

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_time: 削除開始時刻
            end_time: 削除終了時刻

        Returns:
            削除された件数
        """
        additional_filters = {"symbol": symbol, "timeframe": timeframe}
        deleted_count = self.delete_by_date_range(
            timestamp_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            additional_filters=additional_filters,
        )
        logger.info(
            f"期間指定でOHLCVデータを削除しました ({symbol}, {timeframe}): {deleted_count}件"
        )
        return deleted_count

    def get_available_timeframes(self, symbol: str) -> List[str]:
        """
        指定されたシンボルで利用可能な時間軸のリストを取得

        Args:
            symbol: 取引ペア

        Returns:
            利用可能な時間軸のリスト
        """
        try:
            result = (
                self.db.query(OHLCVData.timeframe)
                .filter(OHLCVData.symbol == symbol)
                .distinct()
                .all()
            )
            return [row[0] for row in result]

        except Exception as e:
            logger.error(f"利用可能な時間軸の取得中にエラーが発生しました: {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """
        利用可能なシンボルのリストを取得

        Returns:
            利用可能なシンボルのリスト
        """
        return super().get_available_symbols("symbol")
