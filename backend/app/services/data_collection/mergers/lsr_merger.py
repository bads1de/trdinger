"""
Long Short Ratio データマージャー

Long Short Ratio データのマージロジックを提供します。
"""

import logging
from datetime import datetime
from typing import List

import pandas as pd

from app.utils.error_handler import ErrorHandler
from database.models import LongShortRatioData
from database.repositories.long_short_ratio_repository import LongShortRatioRepository

logger = logging.getLogger(__name__)


class LSRMerger:
    """Long Short Ratio データマージャー"""

    def __init__(self, lsr_repo: LongShortRatioRepository):
        """
        初期化

        Args:
            lsr_repo: Long Short Ratio リポジトリ
        """
        self.lsr_repo = lsr_repo

    def merge_lsr_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        period: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Long Short Ratio データをマージ

        Args:
            df: ベースとなるDataFrame
            symbol: 取引ペア
            period: 期間（例: '5min', '1h'）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            LSRデータがマージされたDataFrame
        """
        try:
            lsr_data = self.lsr_repo.get_long_short_ratio_data(
                symbol=symbol, period=period, start_time=start_date, end_time=end_date
            )

            if lsr_data:
                lsr_df = self._convert_lsr_to_dataframe(lsr_data)

                # toleranceを設定（デフォルト4時間）
                # 期間に応じて調整することも検討可能だが、一旦固定値とする
                tolerance = pd.Timedelta(hours=4)

                df = pd.merge_asof(
                    df.sort_index(),
                    lsr_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                    tolerance=tolerance,
                )

            else:
                logger.warning(
                    f"シンボル {symbol} (period={period}) のLong Short Ratioデータが見つかりませんでした。"
                )
                self._fill_default_values(df)

        except Exception as e:
            ErrorHandler.handle_model_error(e, context="merge_lsr_data")
            self._fill_default_values(df)

        return df

    def _convert_lsr_to_dataframe(
        self, lsr_data: List[LongShortRatioData]
    ) -> pd.DataFrame:
        """
        LongShortRatioDataリストをpandas.DataFrameに変換

        Args:
            lsr_data: LongShortRatioDataオブジェクトのリスト

        Returns:
            LSRのDataFrame
        """
        data = {
            "lsr_buy_ratio": [r.buy_ratio for r in lsr_data],
            "lsr_sell_ratio": [r.sell_ratio for r in lsr_data],
        }
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.timestamp for r in lsr_data])
        return df

    def _fill_default_values(self, df: pd.DataFrame) -> None:
        """デフォルト値（0.5 = ニュートラル）で埋める"""
        df["lsr_buy_ratio"] = 0.5
        df["lsr_sell_ratio"] = 0.5
