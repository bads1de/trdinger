"""
バックテスト用データサービス

"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pandas as pd

from app.utils.label_generation import EventDrivenLabelGenerator
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository

# 循環インポート回避のため、型チェック時のみインポート
if TYPE_CHECKING:
    from app.services.auto_strategy.services.regime_detector import RegimeDetector

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
        regime_detector: Optional["RegimeDetector"] = None,
        event_label_generator: Optional[EventDrivenLabelGenerator] = None,
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVデータリポジトリ
            oi_repo: Open Interestデータリポジトリ
            fr_repo: Funding Rateデータリポジトリ
            regime_detector: レジーム検知器
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
        self._regime_detector: Optional["RegimeDetector"] = regime_detector
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
            result = self._integration_service.create_backtest_dataframe(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                include_oi=True,
                include_fr=True,
            )
            if "funding_rate" in result.columns:
                result["funding_rate"].describe()
            return result
        except DataIntegrationError as e:
            logger.error(f"バックテスト用データ作成エラー: {e}")
            raise ValueError(f"バックテスト用データの作成に失敗しました: {e}")

    def get_ohlcv_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """OHLCVデータをDataFrameとして取得"""

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
        MLトレーニング用にOHLCV、OI、FRデータを統合

        リファクタリング後の実装では、専門サービスに処理を委譲します。

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            統合されたDataFrame（Open, High, Low, Close, Volume, open_interest, funding_rate）

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

    def get_event_labeled_training_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """イベントドリブンラベル付きの学習データを取得"""

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

        regime_labels = None
        detector = self._ensure_regime_detector()
        if detector is not None:
            try:
                regime_labels = detector.detect_regimes(market_df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"レジーム検知に失敗したためフォールバックします: {exc}")
                regime_labels = None

        labels_df, profile_info = self._event_label_generator.generate_hrhp_lrlp_labels(
            market_df,
            regime_labels=regime_labels,
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

    def _ensure_regime_detector(self) -> Optional["RegimeDetector"]:
        if self._regime_detector is None:
            try:
                # 循環インポート回避のため、ここでインポート
                from app.services.auto_strategy.services.regime_detector import (
                    RegimeDetector,
                )

                self._regime_detector = RegimeDetector()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"RegimeDetector初期化に失敗したため無効化します: {exc}")
                self._regime_detector = None
        return self._regime_detector
