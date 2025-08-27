"""
OHLCV データのリポジトリクラス
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import OHLCVData
from app.utils.data_validation import DataValidator


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

        from app.utils.error_handler import safe_operation

        @safe_operation(context="OHLCVデータ挿入", is_api_call=False)
        def _insert_data():
            # データの検証
            if not DataValidator.validate_ohlcv_records_simple(ohlcv_records):
                raise ValueError(
                    "挿入しようとしているOHLCVデータに無効なものが含まれています。"
                )

            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                ohlcv_records, ["symbol", "timeframe", "timestamp"]
            )

            logger.info(f"OHLCV データを {inserted_count} 件挿入しました")
            return inserted_count

        return _insert_data()

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

    def get_latest_timestamp(
        self,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最新タイムスタンプを取得（BaseRepositoryのメソッドをオーバーライド）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最新のタイムスタンプ、データが存在しない場合はNone
        """
        # OHLCVデータの場合は常にtimestampカラムを使用
        if timestamp_column != "timestamp":
            logger.warning(f"OHLCVデータではtimestampカラムを使用してください: {timestamp_column}")

        return super().get_latest_timestamp(timestamp_column, filter_conditions)

    def get_oldest_timestamp(
        self,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Optional[datetime]:
        """
        最古タイムスタンプを取得（BaseRepositoryのメソッドをオーバーライド）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            最古のタイムスタンプ、データが存在しない場合はNone
        """
        # OHLCVデータの場合は常にtimestampカラムを使用
        if timestamp_column != "timestamp":
            logger.warning(f"OHLCVデータではtimestampカラムを使用してください: {timestamp_column}")

        return super().get_oldest_timestamp(timestamp_column, filter_conditions)

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

    def get_date_range(
        self,
        timestamp_column: str,
        filter_conditions: Optional[Dict[str, Any]] = None
    ):
        """
        データ期間を取得（BaseRepositoryのメソッドをオーバーライド）

        Args:
            timestamp_column: タイムスタンプカラム名
            filter_conditions: フィルター条件

        Returns:
            (最古のタイムスタンプ, 最新のタイムスタンプ)
        """
        # OHLCVデータの場合は常にtimestampカラムを使用
        if timestamp_column != "timestamp":
            logger.warning(f"OHLCVデータではtimestampカラムを使用してください: {timestamp_column}")

        return super().get_date_range(timestamp_column, filter_conditions)

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



    def sanitize_ohlcv_data(self, ohlcv_data: List[dict]) -> List[dict]:
        """
        OHLCV データをサニタイズ

        Args:
            ohlcv_data: サニタイズするOHLCVデータのリスト

        Returns:
            サニタイズされたOHLCVデータのリスト
        """
        return DataValidator.sanitize_ohlcv_data(ohlcv_data)

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

    def clear_ohlcv_data_by_symbol_and_timeframe(self, symbol: str, timeframe: str) -> int:
        """
        指定されたシンボルと時間軸のOHLCVデータを削除

        Args:
            symbol: 削除対象のシンボル
            timeframe: 削除対象の時間軸

        Returns:
            削除された件数
        """
        from sqlalchemy import delete

        try:
            # SQLAlchemy 2.0の標準的なdelete文を使用
            stmt = delete(self.model_class).where(
                self.model_class.symbol == symbol,
                self.model_class.timeframe == timeframe
            )
            result = self.db.execute(stmt)
            deleted_count = getattr(result, "rowcount", 0)
            self.db.commit()

            logger.info(
                f"シンボル '{symbol}' 時間軸 '{timeframe}' のOHLCVデータを削除しました: {deleted_count}件"
            )
            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(
                f"シンボル '{symbol}' 時間軸 '{timeframe}' のOHLCVデータ削除エラー: {e}"
            )
            raise

    def get_available_symbols(self, symbol_column: str = "symbol") -> List[str]:
        """
        利用可能なシンボルのリストを取得（BaseRepositoryのメソッドをオーバーライド）

        Args:
            symbol_column: シンボルカラム名

        Returns:
            利用可能なシンボルのリスト
        """
        return super().get_available_symbols(symbol_column)
