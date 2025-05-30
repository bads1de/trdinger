"""
テクニカル指標データリポジトリ

テクニカル指標データのデータベースアクセス機能を提供します。
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc
import logging

from database.models import TechnicalIndicatorData
from database.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TechnicalIndicatorRepository(BaseRepository):
    """テクニカル指標データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, TechnicalIndicatorData)

    def insert_technical_indicator_data(
        self, technical_indicator_records: List[dict]
    ) -> int:
        """
        テクニカル指標データを一括挿入

        Args:
            technical_indicator_records: テクニカル指標データのリスト

        Returns:
            挿入された件数
        """
        if not technical_indicator_records:
            logger.warning("挿入するテクニカル指標データがありません")
            return 0

        try:
            # 重複処理付き一括挿入
            inserted_count = self.bulk_insert_with_conflict_handling(
                technical_indicator_records,
                ["symbol", "indicator_type", "period", "timestamp"],
            )

            logger.info(f"テクニカル指標データを {inserted_count} 件挿入しました")
            return inserted_count

        except Exception as e:
            logger.error(f"テクニカル指標データ挿入エラー: {e}")
            raise

    def get_technical_indicator_data(
        self,
        symbol: str,
        indicator_type: Optional[str] = None,
        period: Optional[int] = None,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[TechnicalIndicatorData]:
        """
        テクニカル指標データを取得

        Args:
            symbol: 取引ペア
            indicator_type: 指標タイプ（SMA, EMA, RSI等）
            period: 期間
            timeframe: 時間枠
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            テクニカル指標データのリスト
        """
        try:
            query = self.db.query(TechnicalIndicatorData).filter(
                TechnicalIndicatorData.symbol == symbol
            )

            if indicator_type:
                query = query.filter(
                    TechnicalIndicatorData.indicator_type == indicator_type
                )

            if period:
                query = query.filter(TechnicalIndicatorData.period == period)

            if timeframe:
                query = query.filter(TechnicalIndicatorData.timeframe == timeframe)

            if start_time:
                query = query.filter(TechnicalIndicatorData.timestamp >= start_time)

            if end_time:
                query = query.filter(TechnicalIndicatorData.timestamp <= end_time)

            # 時系列順でソート
            query = query.order_by(asc(TechnicalIndicatorData.timestamp))

            if limit:
                query = query.limit(limit)

            return query.all()

        except Exception as e:
            logger.error(f"テクニカル指標データ取得エラー: {e}")
            raise

    def get_latest_technical_indicator_timestamp(
        self, symbol: str, indicator_type: str, period: int, timeframe: str
    ) -> Optional[datetime]:
        """
        指定された条件の最新テクニカル指標タイムスタンプを取得

        Args:
            symbol: 取引ペア
            indicator_type: 指標タイプ
            period: 期間
            timeframe: 時間枠

        Returns:
            最新のデータタイムスタンプ（データがない場合はNone）
        """
        return super().get_latest_timestamp(
            "timestamp",
            {
                "symbol": symbol,
                "indicator_type": indicator_type,
                "period": period,
                "timeframe": timeframe,
            },
        )

    def get_technical_indicator_count(
        self,
        symbol: str,
        indicator_type: Optional[str] = None,
        period: Optional[int] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        """
        指定された条件のテクニカル指標データ件数を取得

        Args:
            symbol: 取引ペア
            indicator_type: 指標タイプ
            period: 期間
            timeframe: 時間枠

        Returns:
            データ件数
        """
        filter_conditions = {"symbol": symbol}
        if indicator_type:
            filter_conditions["indicator_type"] = indicator_type
        if period:
            filter_conditions["period"] = period
        if timeframe:
            filter_conditions["timeframe"] = timeframe

        return super().get_record_count(filter_conditions)

    def get_available_indicators(self, symbol: str) -> List[Dict[str, Any]]:
        """
        指定されたシンボルで利用可能な指標の一覧を取得

        Args:
            symbol: 取引ペア

        Returns:
            利用可能な指標の一覧（indicator_type, period, timeframe）
        """
        try:
            query = (
                self.db.query(
                    TechnicalIndicatorData.indicator_type,
                    TechnicalIndicatorData.period,
                    TechnicalIndicatorData.timeframe,
                )
                .filter(TechnicalIndicatorData.symbol == symbol)
                .distinct()
                .order_by(
                    TechnicalIndicatorData.indicator_type,
                    TechnicalIndicatorData.period,
                    TechnicalIndicatorData.timeframe,
                )
            )

            results = query.all()
            return [
                {
                    "indicator_type": result.indicator_type,
                    "period": result.period,
                    "timeframe": result.timeframe,
                }
                for result in results
            ]

        except Exception as e:
            logger.error(f"利用可能な指標取得エラー: {e}")
            raise

    def get_current_technical_indicator_values(
        self, symbol: str, timeframe: str
    ) -> List[Dict[str, Any]]:
        """
        指定されたシンボルと時間枠の最新テクニカル指標値を取得

        Args:
            symbol: 取引ペア
            timeframe: 時間枠

        Returns:
            最新のテクニカル指標値のリスト
        """
        try:
            # 各指標タイプ・期間の組み合わせで最新のデータを取得
            subquery = (
                self.db.query(
                    TechnicalIndicatorData.indicator_type,
                    TechnicalIndicatorData.period,
                    desc(TechnicalIndicatorData.timestamp).label("max_timestamp"),
                )
                .filter(
                    TechnicalIndicatorData.symbol == symbol,
                    TechnicalIndicatorData.timeframe == timeframe,
                )
                .group_by(
                    TechnicalIndicatorData.indicator_type,
                    TechnicalIndicatorData.period,
                )
                .subquery()
            )

            query = (
                self.db.query(TechnicalIndicatorData)
                .join(
                    subquery,
                    (TechnicalIndicatorData.indicator_type == subquery.c.indicator_type)
                    & (TechnicalIndicatorData.period == subquery.c.period)
                    & (TechnicalIndicatorData.timestamp == subquery.c.max_timestamp),
                )
                .filter(
                    TechnicalIndicatorData.symbol == symbol,
                    TechnicalIndicatorData.timeframe == timeframe,
                )
                .order_by(
                    TechnicalIndicatorData.indicator_type,
                    TechnicalIndicatorData.period,
                )
            )

            results = query.all()
            return [result.to_dict() for result in results]

        except Exception as e:
            logger.error(f"現在のテクニカル指標値取得エラー: {e}")
            raise
