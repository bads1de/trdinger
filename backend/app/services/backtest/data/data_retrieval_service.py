"""
データ取得サービス

"""

import logging
from datetime import datetime
from typing import List, Optional

from database.models import (
    FearGreedIndexData,
    FundingRateData,
    OHLCVData,
    OpenInterestData,
)
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

logger = logging.getLogger(__name__)


class DataRetrievalError(Exception):
    """データ取得エラー"""


class DataRetrievalService:
    """
    データ取得サービス

    データベースからの各種データ取得を専門に担当します。
    """

    def __init__(
        self,
        ohlcv_repo: Optional[OHLCVRepository] = None,
        oi_repo: Optional[OpenInterestRepository] = None,
        fr_repo: Optional[FundingRateRepository] = None,
        fear_greed_repo: Optional[FearGreedIndexRepository] = None,
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVデータリポジトリ
            oi_repo: Open Interestデータリポジトリ
            fr_repo: Funding Rateデータリポジトリ
            fear_greed_repo: Fear & Greedインデックスリポジトリ
        """
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo
        self.fear_greed_repo = fear_greed_repo

    def get_ohlcv_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> List[OHLCVData]:
        """
        OHLCVデータを取得

        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            OHLCVデータのリスト

        Raises:
            DataRetrievalError: データ取得に失敗した場合
        """
        if self.ohlcv_repo is None:
            raise DataRetrievalError("OHLCVRepositoryが初期化されていません")

        try:
            data = self.ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )

            if not data:
                raise DataRetrievalError(
                    f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした"
                )

            logger.debug(f"OHLCVデータ取得完了: {len(data)}件")
            return data

        except Exception as e:
            logger.error(f"OHLCVデータ取得エラー: {e}")
            raise DataRetrievalError(f"OHLCVデータの取得に失敗しました: {e}")

    def get_open_interest_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[OpenInterestData]:
        """
        Open Interestデータを取得

        Args:
            symbol: 取引ペア
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            Open Interestデータのリスト
        """
        if self.oi_repo is None:
            logger.warning("Open InterestRepositoryが初期化されていません")
            return []

        try:
            data = self.oi_repo.get_open_interest_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Open Interestデータ取得完了: {len(data)}件")
            return data

        except Exception as e:
            logger.warning(f"Open Interestデータ取得エラー: {e}")
            return []

    def get_funding_rate_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[FundingRateData]:
        """
        Funding Rateデータを取得

        Args:
            symbol: 取引ペア
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            Funding Rateデータのリスト
        """
        if self.fr_repo is None:
            logger.warning("Funding RateRepositoryが初期化されていません")
            return []

        try:
            data = self.fr_repo.get_funding_rate_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Funding Rateデータ取得完了: {len(data)}件")
            return data

        except Exception as e:
            logger.warning(f"Funding Rateデータ取得エラー: {e}")
            return []

    def get_fear_greed_data(
        self, start_date: datetime, end_date: datetime
    ) -> List[FearGreedIndexData]:
        """
        Fear & Greedインデックスデータを取得

        Args:
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            Fear & Greedインデックスデータのリスト
        """
        if self.fear_greed_repo is None:
            logger.warning("Fear & GreedRepositoryが初期化されていません")
            return []

        try:
            data = self.fear_greed_repo.get_fear_greed_data(
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Fear & Greedデータ取得完了: {len(data)}件")
            return data

        except Exception as e:
            logger.warning(f"Fear & Greedデータ取得エラー: {e}")
            return []

    def validate_repositories(self) -> bool:
        """
        リポジトリの初期化状態を検証

        Returns:
            すべてのリポジトリが初期化されている場合True
        """
        missing_repos = []

        if self.ohlcv_repo is None:
            missing_repos.append("OHLCV")
        if self.oi_repo is None:
            missing_repos.append("Open Interest")
        if self.fr_repo is None:
            missing_repos.append("Funding Rate")
        if self.fear_greed_repo is None:
            missing_repos.append("Fear & Greed")

        if missing_repos:
            logger.warning(f"未初期化のリポジトリ: {', '.join(missing_repos)}")
            return False

        return True
