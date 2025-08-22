"""
Funding Rate データマージャー

Funding Rate データのマージロジックを提供します。
"""

import logging
from datetime import datetime
from typing import List

import pandas as pd

from database.models import FundingRateData
from database.repositories.funding_rate_repository import FundingRateRepository
from app.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class FRMerger:
    """Funding Rate データマージャー"""

    def __init__(self, fr_repo: FundingRateRepository):
        """
        初期化

        Args:
            fr_repo: Funding Rate リポジトリ
        """
        self.fr_repo = fr_repo

    def merge_fr_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Funding Rate データをマージ

        Args:
            df: ベースとなるDataFrame
            symbol: 取引ペア
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            FRデータがマージされたDataFrame
        """
        try:
            fr_data = self.fr_repo.get_funding_rate_data(
                symbol=symbol, start_time=start_date, end_time=end_date
            )

            if fr_data:
                fr_df = self._convert_fr_to_dataframe(fr_data)
                # logger.info(
                #     f"FR DataFrame: {len(fr_df)}行, "
                #     f"期間: {fr_df.index.min()} - {fr_df.index.max()}"
                # )

                # toleranceを設定（8時間以内のデータのみ使用）
                df = pd.merge_asof(
                    df.sort_index(),
                    fr_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                    tolerance=pd.Timedelta(hours=8),
                )

                valid_fr_count = df["funding_rate"].notna().sum()
                # logger.info(
                #     f"FRデータマージ完了: {valid_fr_count}/{len(df)}行に値あり "
                #     f"({valid_fr_count/len(df)*100:.1f}%)"
                # )
            else:
                logger.warning(
                    f"シンボル {symbol} のFunding Rateデータが見つかりませんでした。"
                )
                df["funding_rate"] = pd.NA

        except Exception as e:
            ErrorHandler.handle_model_error(e, context="merge_fr_data")

        return df

    def _convert_fr_to_dataframe(self, fr_data: List[FundingRateData]) -> pd.DataFrame:
        """
        FundingRateDataリストをpandas.DataFrameに変換

        Args:
            fr_data: FundingRateDataオブジェクトのリスト

        Returns:
            Funding RateのDataFrame
        """
        data = {"funding_rate": [r.funding_rate for r in fr_data]}
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.funding_timestamp for r in fr_data])
        return df
