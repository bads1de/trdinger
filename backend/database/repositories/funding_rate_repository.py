"""
ファンディングレートデータのリポジトリクラス
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import asc
import pandas as pd
import logging

from .base_repository import BaseRepository
from database.models import FundingRateData

logger = logging.getLogger(__name__)


class FundingRateRepository(BaseRepository):
    """ファンディングレートデータのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, FundingRateData)

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
            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                funding_rate_records, ["symbol", "funding_timestamp"]
            )

            logger.info(f"ファンディングレートデータを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
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
        return super().get_latest_timestamp("funding_timestamp", {"symbol": symbol})

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
        try:
            # 削除前の件数を取得
            self.db.query(FundingRateData).count()

            # 全てのファンディングレートデータを削除
            deleted_count = self.db.query(FundingRateData).delete()

            # コミット
            self.db.commit()

            logger.info(
                f"全てのファンディングレートデータを削除しました: {deleted_count}件"
            )
            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"ファンディングレートデータ全削除エラー: {e}")
            raise

    def clear_funding_rate_data_by_symbol(self, symbol: str) -> int:
        """
        指定されたシンボルのファンディングレートデータを削除

        Args:
            symbol: 削除対象のシンボル

        Returns:
            削除された件数
        """
        try:
            # 指定シンボルのデータを削除
            deleted_count = (
                self.db.query(FundingRateData)
                .filter(FundingRateData.symbol == symbol)
                .delete()
            )

            # コミット
            self.db.commit()

            logger.info(
                f"シンボル '{symbol}' のファンディングレートデータを削除しました: {deleted_count}件"
            )
            return deleted_count

        except Exception as e:
            self.db.rollback()
            logger.error(
                f"シンボル '{symbol}' のファンディングレートデータ削除エラー: {e}"
            )
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
