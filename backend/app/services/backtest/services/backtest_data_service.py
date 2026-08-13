"""
バックテスト用データサービス

"""

import logging
from datetime import datetime

import pandas as pd

from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.long_short_ratio_repository import (
    LongShortRatioRepository,
)
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import (
    OpenInterestRepository,
)

from ..data.data_conversion_service import DataConversionService
from ..data.data_integration_service import (
    DataIntegrationError,
    DataIntegrationService,
)
from ..data.data_retrieval_service import DataRetrievalService

logger = logging.getLogger(__name__)


class BacktestDataService:
    """
    バックテスト用データの供給を一元管理するファサードサービス

    `DataRetrieval`（取得）、`DataConversion`（変換）、`DataIntegration`（統合）
    といった低層サービスを組み合わせ、シミュレーション実行に必要な
    価格データを生成します。
    """

    def __init__(
        self,
        ohlcv_repo: OHLCVRepository | None = None,
        oi_repo: OpenInterestRepository | None = None,
        fr_repo: FundingRateRepository | None = None,
        lsr_repo: LongShortRatioRepository | None = None,
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVデータリポジトリ
            oi_repo: Open Interestデータリポジトリ
            fr_repo: Funding Rateデータリポジトリ
            lsr_repo: Long/Short Ratioデータリポジトリ
        """
        # 専門サービスを初期化
        self._retrieval_service = DataRetrievalService(
            ohlcv_repo=ohlcv_repo,
            oi_repo=oi_repo,
            fr_repo=fr_repo,
            lsr_repo=lsr_repo,
        )
        self._conversion_service = DataConversionService()
        self._integration_service = DataIntegrationService(
            retrieval_service=self._retrieval_service,
            conversion_service=self._conversion_service,
        )

    def get_data_for_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        OHLCV、OI、FR データを統合してシミュレーター形式に変換

        指定された期間の全市場データを収集・結合し、
        backtesting.py 等のライブラリが期待するカラム構成の DataFrame を返します。

        Args:
            symbol: 取引ペア（例: BTC/USDT:USDT）
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            Open, High, Low, Close, Volume, open_interest, funding_rate を含む DataFrame
        """
        try:
            result = self._integration_service.create_backtest_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_oi=True,
                include_fr=True,
                include_lsr=True,
            )
            return result
        except DataIntegrationError as e:
            logger.error(f"バックテスト用データ作成エラー: {e}")
            raise ValueError(f"バックテスト用データの作成に失敗しました: {e}")

    def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        指定された期間と銘柄のOHLCVデータを取得

        リポジトリ経由で生データを取得し、タイムスタンプをインデックスとする
        クリーンなDataFrameに整理します。

        Args:
            symbol: 通貨ペア
            timeframe: 時間足
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            Open, High, Low, Close, Volume を含むDataFrame。
            取得失敗時は空のDataFrameを返します。
        """
        raw_data = self._retrieval_service.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        df = self._conversion_service.convert_ohlcv_to_dataframe(raw_data)
        df = df.drop(columns=["timestamp"], errors="ignore")

        for column in ("open", "high", "low", "close", "volume"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        if df.empty:
            logger.warning(
                "OHLCVデータが取得できなかったため空のDataFrameを返します: %s %s",
                symbol,
                timeframe,
            )

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        return self._integration_service.get_data_summary(df)
