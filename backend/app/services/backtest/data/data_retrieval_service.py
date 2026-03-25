"""
データ取得サービス

"""

import logging
from datetime import datetime
from typing import Any, Callable, List, Optional, cast

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

    def _fetch_with_safe_operation(
        self,
        *,
        context: str,
        default_return: list[Any],
        query: Callable[[], Any],
        raise_on_empty: bool = False,
        empty_error_message: Optional[str] = None,
    ) -> list[Any]:
        """repository 呼び出しを安全に実行する共通ラッパー。"""
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context=context,
            is_api_call=False,
            default_return=default_return,
        )
        def _run_query() -> list[Any]:
            data = query()
            if data is None:
                if raise_on_empty:
                    logger.error(
                        empty_error_message or f"{context}のデータが見つかりませんでした"
                    )
                    raise DataRetrievalError(
                        empty_error_message
                        or f"{context}のデータが見つかりませんでした"
                    )
                return default_return
            if raise_on_empty and not data:
                logger.error(empty_error_message or f"{context}のデータが見つかりませんでした")
                raise DataRetrievalError(
                    empty_error_message or f"{context}のデータが見つかりませんでした"
                )
            return data

        return _run_query()

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

        def query() -> List[OHLCVData]:
            ohlcv_repo = cast(OHLCVRepository, self.ohlcv_repo)
            return ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )

        return self._fetch_with_safe_operation(
            context="OHLCVデータ取得",
            default_return=[],
            query=query,
            raise_on_empty=True,
            empty_error_message=f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした",
        )

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

        def query() -> List[OpenInterestData]:
            oi_repo = cast(OpenInterestRepository, self.oi_repo)
            return oi_repo.get_open_interest_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

        data = self._fetch_with_safe_operation(
            context="Open Interestデータ取得",
            default_return=[],
            query=query,
        )
        logger.debug(f"Open Interestデータ取得完了: {len(data)}件")
        return data

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

        def query() -> List[FundingRateData]:
            fr_repo = cast(FundingRateRepository, self.fr_repo)
            return fr_repo.get_funding_rate_data(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
            )

        data = self._fetch_with_safe_operation(
            context="Funding Rateデータ取得",
            default_return=[],
            query=query,
        )
        logger.debug(f"Funding Rateデータ取得完了: {len(data)}件")
        return data



