"""
ロング/ショート比率データのリポジトリクラス
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.models import LongShortRatioData
from database.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class LongShortRatioRepository(BaseRepository):
    """ロング/ショート比率データのリポジトリクラス"""

    def __init__(self, db: Session):
        super().__init__(db, LongShortRatioData)

    def insert_long_short_ratio_data(self, ratio_records: List[dict]) -> int:
        """
        ロング/ショート比率データを一括挿入

        Args:
            ratio_records: 挿入するデータのリスト（辞書形式）

        Returns:
            挿入された件数
        """
        if not ratio_records:
            logger.warning("挿入するロング/ショート比率データがありません")
            return 0

        processed_records = []
        for record in ratio_records:
            try:
                # --- 1. データ抽出とバリデーション ---
                # Bybit API形式: {symbol, buyRatio, sellRatio, timestamp, ...}
                symbol = record.get("symbol")
                timestamp_val = record.get("timestamp")
                buy_ratio = record.get("buyRatio")
                sell_ratio = record.get("sellRatio")
                
                # period は APIレスポンスには含まれない場合があるため、
                # 呼び出し元が補完して渡してくることを想定するか、
                # record内に 'period' キーが存在することを確認する
                period = record.get("period")

                if not all([symbol, timestamp_val, buy_ratio, sell_ratio, period]):
                    logger.debug(f"必須項目が不足しているためレコードをスキップ: {record}")
                    continue

                # --- 2. データ変換 ---
                new_record = {
                    "symbol": symbol,
                    "period": period,
                    "buy_ratio": float(buy_ratio),
                    "sell_ratio": float(sell_ratio),
                }

                # タイムスタンプ処理
                if isinstance(timestamp_val, datetime):
                    new_record["timestamp"] = timestamp_val
                else:
                    # ミリ秒単位のUnixタイムスタンプを想定
                    new_record["timestamp"] = datetime.fromtimestamp(
                        int(timestamp_val) / 1000, tz=timezone.utc
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

        @safe_operation(context="ロング/ショート比率データ挿入", is_api_call=False)
        def _insert_data():
            # 重複チェック対象のカラム
            conflict_columns = ["symbol", "period", "timestamp"]
            
            inserted_count = self.bulk_insert_with_conflict_handling(
                processed_records, conflict_columns
            )
            logger.info(f"ロング/ショート比率データを {inserted_count} 件挿入しました")
            return inserted_count

        return _insert_data()

    def get_long_short_ratio_data(
        self,
        symbol: str,
        period: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[LongShortRatioData]:
        """
        ロング/ショート比率データを取得

        Args:
            symbol: 取引ペア
            period: 期間（例: '5min', '1h'）
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            ロング/ショート比率データのリスト
        """
        filters = {"symbol": symbol, "period": period}
        return self.get_filtered_data(
            filters=filters,
            time_range_column="timestamp",
            start_time=start_time,
            end_time=end_time,
            order_by_column="timestamp",
            order_asc=True,
            limit=limit,
        )

    def get_latest_ratio(self, symbol: str, period: str) -> Optional[LongShortRatioData]:
        """
        最新のロング/ショート比率データを取得

        Args:
            symbol: 取引ペア
            period: 期間

        Returns:
            最新のLongShortRatioDataオブジェクト、またはNone
        """
        records = self.get_latest_records(
            filters={"symbol": symbol, "period": period},
            timestamp_column="timestamp",
            limit=1,
        )
        return records[0] if records else None

    def get_oldest_ratio_timestamp(self, symbol: str, period: str) -> Optional[datetime]:
        """
        最古のデータタイムスタンプを取得

        Args:
            symbol: 取引ペア
            period: 期間

        Returns:
            最古のタイムスタンプ
        """
        return self.get_oldest_timestamp(
            "timestamp", 
            filter_conditions={"symbol": symbol, "period": period}
        )

    def get_ratio_dataframe(
        self,
        symbol: str,
        period: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        ロング/ショート比率データをDataFrameとして取得

        Args:
            symbol: 取引ペア
            period: 期間
            start_time: 開始時刻
            end_time: 終了時刻
            limit: 取得件数制限

        Returns:
            DataFrame
        """
        records = self.get_long_short_ratio_data(
            symbol, period, start_time, end_time, limit
        )

        column_mapping = {
            "timestamp": "timestamp",
            "buy_ratio": "buy_ratio",
            "sell_ratio": "sell_ratio",
            # symbol と period はフィルタリング済みなので必須ではないが、明示的に含めることも可能
        }

        df = self.to_dataframe(
            records=records,
            column_mapping=column_mapping,
            index_column="timestamp",
        )
        
        # 追加で計算カラムを入れると便利（例: LS比）
        if not df.empty:
             # ゼロ除算回避
            df["ls_ratio"] = df["buy_ratio"] / df["sell_ratio"].replace(0, float('nan'))
            
        return df

    def clear_data(self, symbol: str, period: str) -> int:
        """
        指定されたシンボルと期間のデータを全削除

        Args:
            symbol: 取引ペア
            period: 期間

        Returns:
            削除件数
        """
        filters = {"symbol": symbol, "period": period}
        deleted_count = self.delete_by_date_range(
            timestamp_column="timestamp",
            additional_filters=filters
        )
        logger.info(f"ロング/ショート比率データを削除しました ({symbol}, {period}): {deleted_count}件")
        return deleted_count



