"""
データ統合サービス

"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from app.utils.data_processing import data_processor
from app.utils.error_handler import safe_operation

from ...data_collection.mergers import FearGreedMerger, FRMerger, OIMerger
from .data_conversion_service import DataConversionService
from .data_retrieval_service import DataRetrievalService

logger = logging.getLogger(__name__)


class DataIntegrationError(Exception):
    """データ統合エラー"""


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
        if hasattr(retrieval_service, "oi_repo") and retrieval_service.oi_repo:
            self.oi_merger = OIMerger(retrieval_service.oi_repo)

        if hasattr(retrieval_service, "fr_repo") and retrieval_service.fr_repo:
            self.fr_merger = FRMerger(retrieval_service.fr_repo)

        if (
            hasattr(retrieval_service, "fear_greed_repo")
            and retrieval_service.fear_greed_repo
        ):
            self.fear_greed_merger = FearGreedMerger(retrieval_service.fear_greed_repo)

    @safe_operation(
        context="バックテスト用DataFrame作成",
        is_api_call=False,
        default_return=pd.DataFrame(),
    )
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
        """
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

        return df

    @safe_operation(
        context="ML訓練用DataFrame作成",
        is_api_call=False,
        default_return=pd.DataFrame(),
    )
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

    @safe_operation(context="ベースOHLCVデータ取得", is_api_call=False)
    def _get_base_ohlcv_dataframe(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """ベースとなるOHLCVデータのDataFrameを取得"""
        ohlcv_data = self.retrieval_service.get_ohlcv_data(
            symbol, timeframe, start_date, end_date
        )
        df = self.conversion_service.convert_ohlcv_to_dataframe(ohlcv_data)
        return df

    @safe_operation(context="Open Interestデータ統合", is_api_call=False)
    def _integrate_open_interest_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Open Interestデータを統合"""
        if self.oi_merger:
            df = self.oi_merger.merge_oi_data(df, symbol, start_date, end_date)
        else:
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
                df["funding_rate"] = 0.0
        except Exception as e:
            logger.error(f"Funding Rateデータ統合エラー: {e}")
            df["funding_rate"] = 0.0

        return df

    @safe_operation(context="Fear & Greedデータ統合", is_api_call=False)
    def _integrate_fear_greed_data(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Fear & Greedデータを統合"""
        if self.fear_greed_merger:
            df = self.fear_greed_merger.merge_fear_greed_data(
                df, start_date, end_date, detailed_logging=True
            )
        else:
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA

        return df

    @safe_operation(
        context="データクリーニングと最適化",
        is_api_call=False,
        default_return=pd.DataFrame(),
    )
    def _clean_and_optimize_dataframe(
        self, df: pd.DataFrame, include_fear_greed: bool = False
    ) -> pd.DataFrame:
        """DataFrameのクリーニングと最適化"""
        # 必須カラムを定義
        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
            "funding_rate",
        ]

        if include_fear_greed:
            required_columns.extend(["fear_greed_value", "fear_greed_classification"])

        # データクリーニングと検証
        df = data_processor.clean_and_validate_data(
            df,
            required_columns=required_columns,
            interpolate=True,
            optimize=True,
        )

        return df

    @safe_operation(context="データ概要取得", is_api_call=False, default_return={})
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

        # 安全に日付を取得
        start_date_val: pd.Timestamp = pd.to_datetime(df.index.min(), errors='coerce')  # type: ignore
        start_date = start_date_val.isoformat() if pd.notna(start_date_val) else None

        end_date_val: pd.Timestamp = pd.to_datetime(df.index.max(), errors='coerce')  # type: ignore
        end_date = end_date_val.isoformat() if pd.notna(end_date_val) else None

        # カラム名のケースをチェックして適切なものを選択
        low_col = "low" if "low" in df.columns else "Low"
        high_col = "high" if "high" in df.columns else "High"
        close_col = "close" if "close" in df.columns else "Close"
        volume_col = "volume" if "volume" in df.columns else "Volume"

        summary = {
            "total_records": len(df),
            "start_date": start_date,
            "end_date": end_date,
            "columns": list(df.columns),
            "price_range": {
                "min": float(df[low_col].min()) if not df.empty else None,
                "max": float(df[high_col].max()) if not df.empty else None,
                "first_close": float(df[close_col].iloc[0]) if not df.empty else None,
                "last_close": float(df[close_col].iloc[-1]) if not df.empty else None,
            },
            "volume_stats": {
                "total": float(df[volume_col].sum()) if not df.empty else 0.0,
                "average": float(df[volume_col].mean()) if not df.empty else 0.0,
                "max": float(df[volume_col].max()) if not df.empty else 0.0,
            }
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
