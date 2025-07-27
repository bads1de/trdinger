"""
データ統合サービス

複数のデータソースを統合してバックテスト用のDataFrameを作成します。
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .data_retrieval_service import DataRetrievalService, DataRetrievalError
from .data_conversion_service import DataConversionService, DataConversionError
from ...data_mergers import OIMerger, FRMerger, FearGreedMerger
from app.utils.data_cleaning_utils import DataCleaner

logger = logging.getLogger(__name__)


class DataIntegrationError(Exception):
    """データ統合エラー"""
    pass


class DataIntegrationService:
    """
    データ統合サービス
    
    複数のデータソースを統合してバックテスト用のDataFrameを作成します。
    """
    
    def __init__(
        self,
        retrieval_service: DataRetrievalService,
        conversion_service: Optional[DataConversionService] = None,
    ):
        """
        初期化

        Args:
            retrieval_service: データ取得サービス
            conversion_service: データ変換サービス（Noneの場合は新規作成）
        """
        self.retrieval_service = retrieval_service
        self.conversion_service = conversion_service or DataConversionService()

        # データマージャーを初期化（リポジトリが利用可能な場合のみ）
        self.oi_merger = None
        self.fr_merger = None
        self.fear_greed_merger = None

        # リポジトリが利用可能な場合はマージャーを初期化
        if hasattr(retrieval_service, 'oi_repo') and retrieval_service.oi_repo:
            self.oi_merger = OIMerger(retrieval_service.oi_repo)

        if hasattr(retrieval_service, 'fr_repo') and retrieval_service.fr_repo:
            self.fr_merger = FRMerger(retrieval_service.fr_repo)

        if hasattr(retrieval_service, 'fear_greed_repo') and retrieval_service.fear_greed_repo:
            self.fear_greed_merger = FearGreedMerger(retrieval_service.fear_greed_repo)
    
    def create_backtest_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        include_oi: bool = True,
        include_fr: bool = True,
        include_fear_greed: bool = False,
    ) -> pd.DataFrame:
        """
        バックテスト用のDataFrameを作成
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時
            include_oi: Open Interestデータを含めるか
            include_fr: Funding Rateデータを含めるか
            include_fear_greed: Fear & Greedデータを含めるか
            
        Returns:
            統合されたDataFrame
            
        Raises:
            DataIntegrationError: データ統合に失敗した場合
        """
        try:
            # 1. OHLCVデータを取得・変換
            df = self._get_base_ohlcv_dataframe(symbol, timeframe, start_date, end_date)
            
            # 2. 追加データを統合
            if include_oi:
                df = self._integrate_open_interest_data(df, symbol, start_date, end_date)
            else:
                df["open_interest"] = 0.0
            
            if include_fr:
                df = self._integrate_funding_rate_data(df, symbol, start_date, end_date)
            else:
                df["funding_rate"] = 0.0
            
            if include_fear_greed:
                df = self._integrate_fear_greed_data(df, start_date, end_date)
            
            # 3. データクリーニングと最適化
            df = self._clean_and_optimize_dataframe(df, include_fear_greed)
            
            logger.info(f"データ統合完了: {len(df)}行, カラム: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"データ統合エラー: {e}")
            raise DataIntegrationError(f"データ統合に失敗しました: {e}")
    
    def create_ml_training_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        ML訓練用のDataFrameを作成（すべてのデータを含む）
        
        Args:
            symbol: 取引ペア
            timeframe: 時間軸
            start_date: 開始日時
            end_date: 終了日時
            
        Returns:
            ML訓練用の統合されたDataFrame
        """
        return self.create_backtest_dataframe(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            include_oi=True,
            include_fr=True,
            include_fear_greed=True,
        )
    
    def _get_base_ohlcv_dataframe(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """ベースとなるOHLCVデータのDataFrameを取得"""
        try:
            ohlcv_data = self.retrieval_service.get_ohlcv_data(
                symbol, timeframe, start_date, end_date
            )
            return self.conversion_service.convert_ohlcv_to_dataframe(ohlcv_data)
            
        except (DataRetrievalError, DataConversionError) as e:
            raise DataIntegrationError(f"OHLCVデータの取得・変換に失敗: {e}")
    
    def _integrate_open_interest_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Open Interestデータを統合"""
        try:
            if self.oi_merger:
                df = self.oi_merger.merge_oi_data(df, symbol, start_date, end_date)
            else:
                df["open_interest"] = pd.NA
            
            return df
            
        except Exception as e:
            logger.warning(f"Open Interestデータ統合エラー: {e}")
            df["open_interest"] = pd.NA
            return df
    
    def _integrate_funding_rate_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Funding Rateデータを統合"""
        try:
            if self.fr_merger:
                df = self.fr_merger.merge_fr_data(df, symbol, start_date, end_date)
            else:
                df["funding_rate"] = pd.NA
            
            return df
            
        except Exception as e:
            logger.warning(f"Funding Rateデータ統合エラー: {e}")
            df["funding_rate"] = pd.NA
            return df
    
    def _integrate_fear_greed_data(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fear & Greedデータを統合"""
        try:
            if self.fear_greed_merger:
                df = self.fear_greed_merger.merge_fear_greed_data(
                    df, start_date, end_date, detailed_logging=True
                )
            else:
                df["fear_greed_value"] = pd.NA
                df["fear_greed_classification"] = pd.NA
            
            return df
            
        except Exception as e:
            logger.warning(f"Fear & Greedデータ統合エラー: {e}")
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA
            return df
    
    def _clean_and_optimize_dataframe(
        self, df: pd.DataFrame, include_fear_greed: bool = False
    ) -> pd.DataFrame:
        """DataFrameのクリーニングと最適化"""
        try:
            # 必須カラムを定義
            required_columns = [
                "Open", "High", "Low", "Close", "Volume",
                "open_interest", "funding_rate"
            ]
            
            if include_fear_greed:
                required_columns.extend(["fear_greed_value", "fear_greed_classification"])
            
            # データクリーニングと検証
            df = DataCleaner.clean_and_validate_data(
                df,
                required_columns=required_columns,
                interpolate=True,
                optimize=True,
            )
            
            return df
            
        except Exception as e:
            logger.error(f"データクリーニングエラー: {e}")
            raise DataIntegrationError(f"データクリーニングに失敗しました: {e}")
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得
        
        Args:
            df: 対象のDataFrame
            
        Returns:
            データ概要の辞書
        """
        if df.empty:
            return {"error": "データがありません"}
        
        try:
            summary = {
                "total_records": len(df),
                "start_date": df.index.min().isoformat(),
                "end_date": df.index.max().isoformat(),
                "columns": list(df.columns),
                "price_range": {
                    "min": float(df["Low"].min()),
                    "max": float(df["High"].max()),
                    "first_close": float(df["Close"].iloc[0]),
                    "last_close": float(df["Close"].iloc[-1]),
                },
                "volume_stats": {
                    "total": float(df["Volume"].sum()),
                    "average": float(df["Volume"].mean()),
                    "max": float(df["Volume"].max()),
                },
            }
            
            # 追加データの統計情報
            if "open_interest" in df.columns:
                summary["open_interest_stats"] = {
                    "average": float(df["open_interest"].mean()),
                    "min": float(df["open_interest"].min()),
                    "max": float(df["open_interest"].max()),
                }
            
            if "funding_rate" in df.columns:
                summary["funding_rate_stats"] = {
                    "average": float(df["funding_rate"].mean()),
                    "min": float(df["funding_rate"].min()),
                    "max": float(df["funding_rate"].max()),
                }
            
            if "fear_greed_value" in df.columns:
                summary["fear_greed_stats"] = {
                    "average": float(df["fear_greed_value"].mean()),
                    "min": float(df["fear_greed_value"].min()),
                    "max": float(df["fear_greed_value"].max()),
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"データ概要取得エラー: {e}")
            return {"error": f"データ概要の取得に失敗しました: {e}"}
