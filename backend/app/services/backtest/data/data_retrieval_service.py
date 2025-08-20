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

        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="OHLCVデータ取得",
            is_api_call=False,
            default_return=DataRetrievalError("OHLCVデータの取得に失敗しました"),
        )
        def _get_ohlcv_data():
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

        return _get_ohlcv_data()

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

        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="Open Interestデータ取得", is_api_call=False, default_return=[]
        )
        def _get_open_interest_data():
            data = self.oi_repo.get_open_interest_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Open Interestデータ取得完了: {len(data)}件")
            return data

        return _get_open_interest_data()

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

        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="Funding Rateデータ取得", is_api_call=False, default_return=[]
        )
        def _get_funding_rate_data():
            data = self.fr_repo.get_funding_rate_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Funding Rateデータ取得完了: {len(data)}件")
            return data

        return _get_funding_rate_data()

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

        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="Fear & Greedデータ取得", is_api_call=False, default_return=[]
        )
        def _get_fear_greed_data():
            data = self.fear_greed_repo.get_fear_greed_data(
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Fear & Greedデータ取得完了: {len(data)}件")
            return data

        return _get_fear_greed_data()
