"""
Open Interest データマージャー

Open Interest データのマージロジックを提供します。
"""

import logging
from datetime import datetime
from typing import List

import pandas as pd

from database.models import OpenInterestData
from database.repositories.open_interest_repository import OpenInterestRepository

logger = logging.getLogger(__name__)


class OIMerger:
    """Open Interest データマージャー"""

    def __init__(self, oi_repo: OpenInterestRepository):
        """
        初期化

        Args:
            oi_repo: Open Interest リポジトリ
        """
        self.oi_repo = oi_repo

    def merge_oi_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Open Interest データをマージ

        Args:
            df: ベースとなるDataFrame
            symbol: 取引ペア
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            OIデータがマージされたDataFrame
        """
        try:
            oi_data = self.oi_repo.get_open_interest_data(
                symbol=symbol, start_time=start_date, end_time=end_date
            )

            logger.info(f"取得したOIデータ件数: {len(oi_data) if oi_data else 0}")

            if oi_data:
                oi_df = self._convert_oi_to_dataframe(oi_data)
                logger.info(
                    f"OI DataFrame: {len(oi_df)}行, "
                    f"期間: {oi_df.index.min()} - {oi_df.index.max()}"
                )

                # toleranceを設定（1日以内のデータのみ使用）
                df = pd.merge_asof(
                    df.sort_index(),
                    oi_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                    tolerance=pd.Timedelta(days=1),
                )

                valid_oi_count = df["open_interest"].notna().sum()
                logger.info(
                    f"OIデータマージ完了: {valid_oi_count}/{len(df)}行に値あり "
                    f"({valid_oi_count/len(df)*100:.1f}%)"
                )
            else:
                logger.warning(
                    f"シンボル {symbol} のOpen Interestデータが見つかりませんでした。"
                )
                df["open_interest"] = pd.NA

        except Exception as e:
            logger.warning(f"Open Interestデータのマージ中にエラーが発生しました: {e}")
            df["open_interest"] = pd.NA

        return df

    def _convert_oi_to_dataframe(self, oi_data: List[OpenInterestData]) -> pd.DataFrame:
        """
        OpenInterestDataリストをpandas.DataFrameに変換

        Args:
            oi_data: OpenInterestDataオブジェクトのリスト

        Returns:
            Open InterestのDataFrame
        """
        data = {"open_interest": [r.open_interest_value for r in oi_data]}
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.data_timestamp for r in oi_data])
        return df
