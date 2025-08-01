"""
Fear & Greed Index データマージャー

Fear & Greed Index データのマージロジックを提供します。
"""

import logging
import pandas as pd
from datetime import datetime
from typing import List
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.models import FearGreedIndexData

logger = logging.getLogger(__name__)


class FearGreedMerger:
    """Fear & Greed Index データマージャー"""

    def __init__(self, fear_greed_repo: FearGreedIndexRepository):
        """
        初期化

        Args:
            fear_greed_repo: Fear & Greed Index リポジトリ
        """
        self.fear_greed_repo = fear_greed_repo

    def merge_fear_greed_data(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        detailed_logging: bool = False,
    ) -> pd.DataFrame:
        """
        Fear & Greed Index データをマージ

        Args:
            df: ベースとなるDataFrame
            start_date: 開始日時
            end_date: 終了日時
            detailed_logging: 詳細ログを出力するかどうか

        Returns:
            Fear & GreedデータがマージされたDataFrame
        """
        try:
            fear_greed_data = self.fear_greed_repo.get_fear_greed_data(
                start_time=start_date, end_time=end_date
            )

            if detailed_logging:
                logger.info(
                    f"取得したFear & Greedデータ件数: {len(fear_greed_data) if fear_greed_data else 0}"
                )

            if fear_greed_data:
                fear_greed_df = self._convert_fear_greed_to_dataframe(fear_greed_data)

                if detailed_logging:
                    logger.info(
                        f"Fear & Greed DataFrame: {len(fear_greed_df)}行, "
                        f"期間: {fear_greed_df.index.min()} - {fear_greed_df.index.max()}"
                    )

                # toleranceを設定（詳細ログ時は3日、通常時は制限なし）
                tolerance = pd.Timedelta(days=3) if detailed_logging else None

                df = pd.merge_asof(
                    df.sort_index(),
                    fear_greed_df.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction="backward",
                    tolerance=tolerance,
                )

                if detailed_logging:
                    valid_fg_count = df["fear_greed_value"].notna().sum()
                    logger.info(
                        f"Fear & Greedデータマージ完了: {valid_fg_count}/{len(df)}行に値あり "
                        f"({valid_fg_count/len(df)*100:.1f}%)"
                    )
            else:
                warning_msg = "Fear & Greedデータが見つかりませんでした。"
                if detailed_logging:
                    logger.warning(f"⚠️ {warning_msg}")
                else:
                    logger.warning(warning_msg)
                df["fear_greed_value"] = pd.NA
                df["fear_greed_classification"] = pd.NA

        except Exception as e:
            logger.warning(f"Fear & Greedデータのマージ中にエラーが発生しました: {e}")
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA

        return df

    def _convert_fear_greed_to_dataframe(
        self, fear_greed_data: List[FearGreedIndexData]
    ) -> pd.DataFrame:
        """
        FearGreedIndexDataリストをpandas.DataFrameに変換

        Args:
            fear_greed_data: FearGreedIndexDataオブジェクトのリスト

        Returns:
            Fear & GreedのDataFrame
        """
        data = {
            "fear_greed_value": [r.value for r in fear_greed_data],
            "fear_greed_classification": [
                r.value_classification for r in fear_greed_data
            ],
        }
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.data_timestamp for r in fear_greed_data])
        return df
