"""
バックテスト用データサービス

"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from .data.data_conversion_service import DataConversionService
from .data.data_integration_service import DataIntegrationError, DataIntegrationService
from .data.data_retrieval_service import DataRetrievalService

logger = logging.getLogger(__name__)


class BacktestDataService:
    """
    リファクタリング後のバックテスト用データサービス

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
        # 専門サービスを初期化
        self._retrieval_service = DataRetrievalService(
            ohlcv_repo=ohlcv_repo,
            oi_repo=oi_repo,
            fr_repo=fr_repo,
            fear_greed_repo=fear_greed_repo,
        )
        self._conversion_service = DataConversionService()
        self._integration_service = DataIntegrationService(
            retrieval_service=self._retrieval_service,
            conversion_service=self._conversion_service,
        )

        # 後方互換性のためにリポジトリも保持
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo
        self.fear_greed_repo = fear_greed_repo

    def get_data_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        OHLCV、OI、FRデータを統合してbacktesting.py形式に変換

        リファクタリング後の実装では、専門サービスに処理を委譲します。

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            backtesting.py用のDataFrame（Open, High, Low, Close, Volume, open_interest, funding_rateカラム）

        Raises:
            DataIntegrationError: データ統合に失敗した場合
        """
        try:
            return self._integration_service.create_backtest_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_oi=True,
                include_fr=True,
                include_fear_greed=False,
            )
        except DataIntegrationError as e:
            logger.error(f"バックテスト用データ作成エラー: {e}")
            raise ValueError(f"バックテスト用データの作成に失敗しました: {e}")

    def get_ml_training_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        MLトレーニング用にOHLCV、OI、FR、Fear & Greedデータを統合

        リファクタリング後の実装では、専門サービスに処理を委譲します。

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            統合されたDataFrame（Open, High, Low, Close, Volume, open_interest, funding_rate, fear_greed_value）

        Raises:
            DataIntegrationError: データ統合に失敗した場合
        """
        try:
            return self._integration_service.create_ml_training_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        except DataIntegrationError as e:
            logger.error(f"MLトレーニング用データ作成エラー: {e}")
            raise ValueError(f"MLトレーニング用データの作成に失敗しました: {e}")

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        return self._integration_service.get_data_summary(df)
