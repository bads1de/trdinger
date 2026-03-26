"""
バックテスト用データサービス

"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from app.services.ml.label_generation import EventDrivenLabelGenerator
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

from .data.data_conversion_service import DataConversionService
from .data.data_integration_service import DataIntegrationError, DataIntegrationService
from .data.data_retrieval_service import DataRetrievalService

logger = logging.getLogger(__name__)


class BacktestDataService:
    """
    バックテストおよび ML 学習用データの供給を一元管理するファサードサービス

    `DataRetrieval`（取得）、`DataConversion`（変換）、`DataIntegration`（統合）
    といった低層サービスを組み合わせ、シミュレーション実行に必要な
    価格データや、イベントラベル（HRHP/LRLP 等）付きの学習データを生成します。
    """

    def __init__(
        self,
        ohlcv_repo: Optional[OHLCVRepository] = None,
        oi_repo: Optional[OpenInterestRepository] = None,
        fr_repo: Optional[FundingRateRepository] = None,
        event_label_generator: Optional[EventDrivenLabelGenerator] = None,
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVデータリポジトリ
            oi_repo: Open Interestデータリポジトリ
            fr_repo: Funding Rateデータリポジトリ
            event_label_generator: イベントドリブンラベル生成器
        """
        # 専門サービスを初期化
        self._retrieval_service = DataRetrievalService(
            ohlcv_repo=ohlcv_repo,
            oi_repo=oi_repo,
            fr_repo=fr_repo,
        )
        self._conversion_service = DataConversionService()
        self._integration_service = DataIntegrationService(
            retrieval_service=self._retrieval_service,
            conversion_service=self._conversion_service,
        )
        self._event_label_generator = (
            event_label_generator or EventDrivenLabelGenerator()
        )

        # 後方互換性のためにリポジトリも保持
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo

    def get_data_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
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
                include_oi=False,
                include_fr=False,
            )
            return result
        except DataIntegrationError as e:
            logger.error(f"バックテスト用データ作成エラー: {e}")
            raise ValueError(f"バックテスト用データの作成に失敗しました: {e}")

    def get_ohlcv_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
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

    def get_ml_training_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        ML モデル学習用に、正規化済みの市場統合データを取得

        価格データに加えて OI (建玉) や FR (金利) をインデックスで整列させ、
        欠損値を補完した状態で返します。特徴量エンジニアリングの入力として使用されます。

        Args:
            symbol: 通貨ペア
            timeframe: 時間足
            start_date: 開始日
            end_date: 終了日

        Returns:
            ML 学習のベースとして利用可能な統合済み DataFrame
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

    def get_event_labeled_training_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        ボラティリティ等の特定イベントに基づきラベリングされた学習データを取得

        `EventDrivenLabelGenerator` と連携し、価格変動リスクに応じた
        HRHP (High Reward High Probability) 等のラベルを付与します。
        これにより、単なる価格予測ではなく「収益機会の有無」を学習させることが可能になります。

        Args:
            symbol: 通貨ペア
            timeframe: 時間足
            start_date: 開始開始日
            end_date: 終了日

        Returns:
            (ラベル付き DataFrame, プロファイル情報等のメタデータ辞書)
        """
        try:
            market_df = self._integration_service.create_ml_training_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
        except DataIntegrationError as exc:
            logger.error(f"イベントラベル用データ作成エラー: {exc}")
            raise ValueError(f"イベントラベル付きデータの作成に失敗しました: {exc}")

        if market_df.empty:
            logger.warning("取得データが空のためイベントラベリングをスキップします")
            return market_df, {"regime_profiles": {}, "label_distribution": {}}

        labels_df, profile_info = self._event_label_generator.generate_hrhp_lrlp_labels(
            market_df,
            regime_labels=None,
        )

        aligned_market = market_df.loc[labels_df.index].copy()
        labeled_df = aligned_market.join(labels_df)

        return labeled_df, profile_info

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        return self._integration_service.get_data_summary(df)
