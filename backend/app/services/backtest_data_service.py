"""
バックテスト用データ変換サービス

backtesting.pyライブラリで使用するためのデータ変換機能を提供します。
Open Interest (OI) と Funding Rate (FR) データの統合機能を含みます。
"""

import logging
import pandas as pd
from datetime import datetime
from typing import List, Optional
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.repositories.fear_greed_repository import FearGreedIndexRepository
from database.models import (
    OHLCVData,
    OpenInterestData,
    FundingRateData,
    FearGreedIndexData,
)
from .data_mergers import OIMerger, FRMerger, FearGreedMerger
from app.utils.data_cleaning_utils import DataCleaner


logger = logging.getLogger(__name__)


class BacktestDataService:
    """
    backtesting.py用のデータ変換サービス

    OHLCVデータにOpen Interest (OI)とFunding Rate (FR)データを統合し、
    backtesting.pyライブラリで使用可能なpandas.DataFrame形式に変換します。
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
            ohlcv_repo: OHLCVリポジトリ（テスト時にモックを注入可能）
            oi_repo: Open Interestリポジトリ（オプション）
            fr_repo: Funding Rateリポジトリ（オプション）
            fear_greed_repo: Fear & Greedリポジトリ（オプション）
        """
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo
        self.fear_greed_repo = fear_greed_repo

        # データマージャーの初期化
        self.oi_merger = OIMerger(oi_repo) if oi_repo else None
        self.fr_merger = FRMerger(fr_repo) if fr_repo else None
        self.fear_greed_merger = (
            FearGreedMerger(fear_greed_repo) if fear_greed_repo else None
        )

    def get_data_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        OHLCV、OI、FRデータを統合してbacktesting.py形式に変換

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            backtesting.py用のDataFrame（Open, High, Low, Close, Volume, open_interest, funding_rateカラム）

        Raises:
            ValueError: データが見つからない場合
        """
        if self.ohlcv_repo is None:
            raise ValueError("OHLCVRepositoryが初期化されていません。")

        try:
            # 1. OHLCVデータを取得
            ohlcv_data = self.ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )

            if not ohlcv_data:
                raise ValueError(
                    f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした。"
                )
        except Exception as e:
            logger.error(f"OHLCVデータの取得中にエラーが発生しました: {e}")
            raise

        # 2. OHLCVデータをDataFrameに変換
        df = self._convert_to_dataframe(ohlcv_data)

        # 3. OI/FRデータを統合
        df = self._merge_additional_data(df, symbol, start_date, end_date)

        # 4. データクリーニングと検証
        df = DataCleaner.clean_and_validate_data(
            df,
            required_columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "open_interest",
                "funding_rate",
            ],
            interpolate=False,  # 既に補間済み
            optimize=True,
        )

        return df

    def _convert_to_dataframe(self, ohlcv_data: List[OHLCVData]) -> pd.DataFrame:
        """
        OHLCVDataリストをpandas.DataFrameに変換

        Args:
            ohlcv_data: OHLCVDataオブジェクトのリスト

        Returns:
            backtesting.py用のDataFrame
        """
        # 効率的にDataFrameを作成
        data = {
            "Open": [r.open for r in ohlcv_data],
            "High": [r.high for r in ohlcv_data],
            "Low": [r.low for r in ohlcv_data],
            "Close": [r.close for r in ohlcv_data],
            "Volume": [r.volume for r in ohlcv_data],
        }

        df = pd.DataFrame(data)

        # インデックスをdatetimeに設定
        df.index = pd.DatetimeIndex([r.timestamp for r in ohlcv_data])

        return df

    def _merge_additional_data(
        self, df: pd.DataFrame, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        OHLCVデータにOI/FRデータをマージ

        Args:
            df: OHLCVデータのDataFrame
            symbol: 取引ペア
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            OI/FRデータがマージされたDataFrame
        """
        # Open Interestデータをマージ
        if self.oi_merger:
            df = self.oi_merger.merge_oi_data(df, symbol, start_date, end_date)
        else:
            df["open_interest"] = pd.NA

        # Funding Rateデータをマージ
        if self.fr_merger:
            df = self.fr_merger.merge_fr_data(df, symbol, start_date, end_date)
        else:
            df["funding_rate"] = pd.NA

        # データクリーニング（補間処理）
        df = DataCleaner.interpolate_oi_fr_data(df)

        return df

    def _merge_fear_greed_data(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fear & Greedデータをマージ

        Args:
            df: 既存のDataFrame
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            Fear & GreedデータがマージされたDataFrame
        """
        if self.fear_greed_merger:
            df = self.fear_greed_merger.merge_fear_greed_data(
                df, start_date, end_date, detailed_logging=True
            )
        else:
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA

        # Fear & Greedデータの補間
        df = DataCleaner.interpolate_fear_greed_data(df)
        return df

    def _improve_data_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ補間の改善

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        return DataCleaner.interpolate_all_data(df)

    def get_ml_training_data(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        MLトレーニング用にOHLCV、OI、FR、Fear & Greedデータを統合

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            統合されたDataFrame（Open, High, Low, Close, Volume, open_interest, funding_rate, fear_greed_value）

        Raises:
            ValueError: データが見つからない場合
        """
        if self.ohlcv_repo is None:
            raise ValueError("OHLCVRepositoryが初期化されていません。")

        try:
            # 1. OHLCVデータを取得
            ohlcv_data = self.ohlcv_repo.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=end_date,
            )

            if not ohlcv_data:
                raise ValueError(
                    f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした。"
                )
        except Exception as e:
            logger.error(
                f"MLトレーニング用OHLCVデータの取得中にエラーが発生しました: {e}"
            )
            raise

        # 2. OHLCVデータをDataFrameに変換
        df = self._convert_to_dataframe(ohlcv_data)

        # 3. OI/FRデータを統合
        df = self._merge_additional_data(df, symbol, start_date, end_date)

        # 4. Fear & Greedデータを統合
        df = self._merge_fear_greed_data(df, start_date, end_date)

        # 5. データ補間の改善
        df = self._improve_data_interpolation(df)

        # 6. データクリーニングと検証
        df = DataCleaner.clean_and_validate_data(
            df,
            required_columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "open_interest",
                "funding_rate",
                "fear_greed_value",
                "fear_greed_classification",
            ],
            interpolate=False,  # 既に補間済み
            optimize=True,
        )

        return df

    def _convert_fear_greed_to_dataframe(
        self, fear_greed_data: List[FearGreedIndexData]
    ) -> pd.DataFrame:
        """
        FearGreedIndexDataリストをpandas.DataFrameに変換

        Args:
            fear_greed_data: FearGreedIndexDataオブジェクトのリスト

        Returns:
            Fear & GreedのDataFrame
        """
        data = {
            "fear_greed_value": [r.value for r in fear_greed_data],
            "fear_greed_classification": [
                r.value_classification for r in fear_greed_data
            ],
        }
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.data_timestamp for r in fear_greed_data])
        return df

    def _validate_ml_training_dataframe(self, df: pd.DataFrame) -> None:
        """
        MLトレーニング用DataFrameの整合性をチェック

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: DataFrameが無効な場合
        """
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "open_interest",
            "funding_rate",
            "fear_greed_value",
            "fear_greed_classification",
        ]
        DataCleaner.validate_extended_data(df, required_columns)

    def _convert_oi_to_dataframe(self, oi_data: List[OpenInterestData]) -> pd.DataFrame:
        """
        OpenInterestDataリストをpandas.DataFrameに変換

        Args:
            oi_data: OpenInterestDataオブジェクトのリスト

        Returns:
            Open InterestのDataFrame
        """
        data = {"open_interest": [r.open_interest_value for r in oi_data]}
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.data_timestamp for r in oi_data])
        return df

    def _convert_fr_to_dataframe(self, fr_data: List[FundingRateData]) -> pd.DataFrame:
        """
        FundingRateDataリストをpandas.DataFrameに変換

        Args:
            fr_data: FundingRateDataオブジェクトのリスト

        Returns:
            Funding RateのDataFrame
        """
        data = {"funding_rate": [r.funding_rate for r in fr_data]}
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.funding_timestamp for r in fr_data])
        return df

    def _validate_extended_dataframe(self, df: pd.DataFrame) -> None:
        """
        拡張されたDataFrame（OI/FR含む）の整合性をチェック

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: DataFrameが無効な場合
        """
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "open_interest",
            "funding_rate",
        ]
        DataCleaner.validate_extended_data(df, required_columns)

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得（OI/FR含む）

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        if df.empty:
            return {"error": "データがありません。"}

        summary = {
            "total_records": len(df),
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
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

        # OI/FRデータが含まれている場合は追加情報を含める
        if "open_interest" in df.columns:
            summary["open_interest_stats"] = {
                "average": float(df["open_interest"].mean()),
                "min": float(df["open_interest"].min()),
                "max": float(df["open_interest"].max()),
                "first": float(df["open_interest"].iloc[0]),
                "last": float(df["open_interest"].iloc[-1]),
            }

        if "funding_rate" in df.columns:
            summary["funding_rate_stats"] = {
                "average": float(df["funding_rate"].mean()),
                "min": float(df["funding_rate"].min()),
                "max": float(df["funding_rate"].max()),
                "first": float(df["funding_rate"].iloc[0]),
                "last": float(df["funding_rate"].iloc[-1]),
            }

        # Fear & Greedデータが含まれている場合は追加情報を含める
        if "fear_greed_value" in df.columns:
            summary["fear_greed_stats"] = {
                "average": float(df["fear_greed_value"].mean()),
                "min": float(df["fear_greed_value"].min()),
                "max": float(df["fear_greed_value"].max()),
                "first": float(df["fear_greed_value"].iloc[0]),
                "last": float(df["fear_greed_value"].iloc[-1]),
            }

        if "fear_greed_classification" in df.columns:
            # 分類の分布を取得
            classification_counts = (
                df["fear_greed_classification"].value_counts().to_dict()
            )
            summary["fear_greed_classification_distribution"] = {
                str(k): int(v) for k, v in classification_counts.items()
            }

        return summary
