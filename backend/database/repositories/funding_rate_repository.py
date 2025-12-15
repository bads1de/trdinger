"""
ファンディングレートデータのリポジトリクラス
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.models import FundingRateData

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class FundingRateRepository(BaseRepository):
    """ファンディングレートデータのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FundingRateData)

    def insert_funding_rate_data(self, funding_rate_records: List[dict]) -> int:
        if not funding_rate_records:
            logger.warning("挿入するファンディングレートデータがありません")
            return 0

        processed_records = []
        for record in funding_rate_records:
            # --- 1. データ抽出 ---
            # Noneとの比較を明示的に行い、0.0がスキップされないようにする
            info = record.get("info", {})

            rate = record.get("funding_rate")
            if rate is None:
                rate = record.get("fundingRate")
            if rate is None:
                rate = info.get("fundingRate")

            funding_ts = record.get("funding_timestamp")
            if funding_ts is None:
                funding_ts = record.get("data_timestamp")
            if funding_ts is None:
                funding_ts = record.get("fundingTimestamp")
            if funding_ts is None:
                funding_ts = info.get("fundingRateTimestamp")
            if funding_ts is None:
                funding_ts = record.get("timestamp")  # 最終フォールバック

            ts = record.get("timestamp")

            # --- 2. スキーマに基づく検証 ---
            if (
                "symbol" not in record
                or rate is None
                or funding_ts is None
                or ts is None
            ):
                logger.debug(f"必須項目が不足しているためレコードをスキップ: {record}")
                continue

            # --- 3. データ変換と構築 ---
            try:
                new_record = {
                    "symbol": record.get("symbol"),
                    "funding_rate": float(rate),
                    "mark_price": record.get("mark_price") or info.get("markPrice"),
                    "index_price": record.get("index_price") or info.get("indexPrice"),
                }

                # タイムスタンプをdatetimeオブジェクトに変換
                if isinstance(funding_ts, datetime):
                    new_record["funding_timestamp"] = funding_ts
                else:
                    new_record["funding_timestamp"] = datetime.fromtimestamp(
                        float(funding_ts) / 1000, tz=timezone.utc
                    )

                if isinstance(ts, datetime):
                    new_record["timestamp"] = ts
                else:
                    new_record["timestamp"] = datetime.fromtimestamp(
                        float(ts) / 1000, tz=timezone.utc
                    )

                next_ts = record.get("next_funding_timestamp") or info.get(
                    "nextFundingTime"
                )
                if next_ts is not None:
                    if isinstance(next_ts, datetime):
                        new_record["next_funding_timestamp"] = next_ts
                    else:
                        new_record["next_funding_timestamp"] = datetime.fromtimestamp(
                            float(next_ts) / 1000, tz=timezone.utc
                        )

                processed_records.append(new_record)

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"レコードの変換に失敗しました（スキップ）: {record}, エラー: {e}"
                )
                continue

        if not processed_records:
            logger.warning("処理後に有効なレコードがありませんでした。")
            return 0

        from app.utils.error_handler import safe_operation

        @safe_operation(context="ファンディングレートデータ挿入", is_api_call=False)
        def _insert_data():
            inserted_count = self.bulk_insert_with_conflict_handling(
                processed_records, ["symbol", "funding_timestamp"]
            )
            logger.info(f"ファンディングレートデータを {inserted_count} 件挿入しました")
            return inserted_count

        return _insert_data()

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
        filters = {"symbol": symbol}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="funding_timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="funding_timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_all_by_symbol(self, symbol: str) -> List[FundingRateData]:
        """
        指定されたシンボルの全ファンディングレートデータを取得

        Args:
            symbol: 取引ペア

        Returns:
            全ファンディングレートデータのリスト（時系列順）
        """
        filters = {"symbol": symbol}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="funding_timestamp",
            start_time=None,
            end_time=None,
            order_by_column="funding_timestamp",
            order_asc=True,
            limit=None,
        )

    def get_latest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最新ファンディングタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最新のファンディングタイムスタンプ（データがない場合はNone）
        """
        return super().get_latest_timestamp("funding_timestamp", {"symbol": symbol})

    def get_oldest_funding_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        指定されたシンボルの最古ファンディングタイムスタンプを取得

        Args:
            symbol: 取引ペア

        Returns:
            最古のファンディングタイムスタンプ（データがない場合はNone）
        """
        return super().get_oldest_timestamp("funding_timestamp", {"symbol": symbol})

    def get_funding_rate_count(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータ件数を取得

        Args:
            symbol: 取引ペア

        Returns:
            データ件数
        """
        return super().get_record_count({"symbol": symbol})

    def clear_all_funding_rate_data(self) -> int:
        """
        全てのファンディングレートデータを削除

        Returns:
            削除された件数
        """
        deleted_count = self._delete_all_records()
        logger.info(
            f"全てのファンディングレートデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_funding_rate_data_by_symbol(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータを削除

        Args:
            symbol: 削除対象のシンボル

        Returns:
            削除された件数
        """
        deleted_count = self._delete_records_by_filter("symbol", symbol)
        logger.info(
            f"シンボル '{symbol}' のファンディングレートデータを削除しました: {deleted_count}件"
        )
        return deleted_count

    def clear_funding_rate_data_by_date_range(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        指定された期間のファンディングレートデータを削除

        Args:
            symbol: 取引ペア
            start_time: 削除開始時刻
            end_time: 削除終了時刻

        Returns:
            削除された件数
        """
        additional_filters = {"symbol": symbol}
        deleted_count = self.delete_by_date_range(
            timestamp_column="funding_timestamp",
            start_time=start_time,
            end_time=end_time,
            additional_filters=additional_filters,
        )
        logger.info(
            f"期間指定でファンディングレートデータを削除しました ({symbol}): {deleted_count}件"
        )
        return deleted_count

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

        column_mapping = {
            "funding_timestamp": "funding_timestamp",
            "funding_rate": "funding_rate",
            "mark_price": "mark_price",
            "index_price": "index_price",
        }

        return self.to_dataframe(
            records=records,
            column_mapping=column_mapping,
            index_column="funding_timestamp",
        )


