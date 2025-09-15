"""
データ取得サービス

"""

import logging
from datetime import datetime
from typing import List, Optional

from database.models import (
    FundingRateData,
    OHLCVData,
    OpenInterestData,
)
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
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVデータリポジトリ
            oi_repo: Open Interestデータリポジトリ
            fr_repo: Funding Rateデータリポジトリ
        """
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo

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
            OHLCVデータのリスト。リポジトリが初期化されていない場合は空のリストを返します。
        """
        if self.ohlcv_repo is None:
            logger.warning("OHLCVRepositoryが初期化されていません")
            return []

        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="OHLCVデータ取得",
            is_api_call=False,
            default_return=[],  # Return empty list instead of exception for consistency
        )
        def _get_ohlcv_data():
            # Type assertion to help type checker understand ohlcv_repo is not None
            assert self.ohlcv_repo is not None, "OHLCVRepositoryが初期化されていません"

            data = self.ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )

            if not data:
                logger.error(f"DataRetrievalService - No OHLCV data found for {symbol} {timeframe}")
                raise DataRetrievalError(
                    f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした"
                )

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
            # Type assertion to help type checker understand oi_repo is not None
            assert self.oi_repo is not None, "Open InterestRepositoryが初期化されていません"

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
            # Type assertion to help type checker understand fr_repo is not None
            assert self.fr_repo is not None, "Funding RateRepositoryが初期化されていません"

            data = self.fr_repo.get_funding_rate_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

            logger.debug(f"Funding Rateデータ取得完了: {len(data)}件")
            return data

        return _get_funding_rate_data()

