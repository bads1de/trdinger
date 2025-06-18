"""
バックテスト用データ変換サービス

backtesting.pyライブラリで使用するためのデータ変換機能を提供します。
Open Interest (OI) と Funding Rate (FR) データの統合機能を含みます。
"""

import pandas as pd
from datetime import datetime
from typing import List, Optional
from database.repositories.ohlcv_repository import OHLCVRepository
from database.repositories.open_interest_repository import OpenInterestRepository
from database.repositories.funding_rate_repository import FundingRateRepository
from database.models import OHLCVData, OpenInterestData, FundingRateData
import logging

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
    ):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVリポジトリ（テスト時にモックを注入可能）
            oi_repo: Open Interestリポジトリ（オプション）
            fr_repo: Funding Rateリポジトリ（オプション）
        """
        self.ohlcv_repo = ohlcv_repo
        self.oi_repo = oi_repo
        self.fr_repo = fr_repo

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
            backtesting.py用のDataFrame（Open, High, Low, Close, Volume, OpenInterest, FundingRateカラム）

        Raises:
            ValueError: データが見つからない場合
        """
        if self.ohlcv_repo is None:
            raise ValueError("OHLCVRepository is not initialized.")
        # 1. OHLCVデータを取得
        ohlcv_data = self.ohlcv_repo.get_ohlcv_data(
            symbol=symbol, timeframe=timeframe, start_time=start_date, end_time=end_date
        )

        if not ohlcv_data:
            raise ValueError(f"No OHLCV data found for {symbol} {timeframe}")

        # 2. OHLCVデータをDataFrameに変換
        df = self._convert_to_dataframe(ohlcv_data)

        # 3. OI/FRデータを統合
        df = self._merge_additional_data(df, symbol, start_date, end_date)

        # 4. データの整合性チェックとソート
        self._validate_extended_dataframe(df)
        df = df.sort_index()  # 時系列順にソート

        return df

    def get_ohlcv_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        OHLCVデータのみをbacktesting.py形式に変換（後方互換性のため）

        Args:
            symbol: 取引ペア（例: BTC/USDT）
            timeframe: 時間軸（例: 1h, 4h, 1d）
            start_date: 開始日時
            end_date: 終了日時

        Returns:
            backtesting.py用のDataFrame（Open, High, Low, Close, Volumeカラム）

        Raises:
            ValueError: データが見つからない場合
        """
        logger.warning(
            "get_ohlcv_for_backtest is deprecated. Use get_data_for_backtest instead."
        )

        if self.ohlcv_repo is None:
            raise ValueError("OHLCVRepository is not initialized.")
        # 1. 既存のリポジトリからデータ取得
        ohlcv_data = self.ohlcv_repo.get_ohlcv_data(
            symbol=symbol, timeframe=timeframe, start_time=start_date, end_time=end_date
        )

        if not ohlcv_data:
            raise ValueError(f"No data found for {symbol} {timeframe}")

        # 2. DataFrameに変換
        df = self._convert_to_dataframe(ohlcv_data)

        # 3. データの整合性チェックとソート
        self._validate_dataframe(df)
        df = df.sort_index()  # 時系列順にソート

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
        if self.oi_repo:
            try:
                oi_data = self.oi_repo.get_open_interest_data(
                    symbol=symbol, start_time=start_date, end_time=end_date
                )
                if oi_data:
                    oi_df = self._convert_oi_to_dataframe(oi_data)
                    df = pd.merge_asof(
                        df.sort_index(),
                        oi_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                    logger.info(f"Merged {len(oi_data)} Open Interest records")
                else:
                    logger.warning(f"No Open Interest data found for {symbol}")
                    df["OpenInterest"] = 0.0  # デフォルト値
            except Exception as e:
                logger.warning(f"Failed to merge Open Interest data: {e}")
                df["OpenInterest"] = 0.0  # デフォルト値
        else:
            df["OpenInterest"] = 0.0  # リポジトリが無い場合のデフォルト値

        # Funding Rateデータをマージ
        if self.fr_repo:
            try:
                fr_data = self.fr_repo.get_funding_rate_data(
                    symbol=symbol, start_time=start_date, end_time=end_date
                )
                if fr_data:
                    fr_df = self._convert_fr_to_dataframe(fr_data)
                    df = pd.merge_asof(
                        df.sort_index(),
                        fr_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                    logger.info(f"Merged {len(fr_data)} Funding Rate records")
                else:
                    logger.warning(f"No Funding Rate data found for {symbol}")
                    df["FundingRate"] = 0.0  # デフォルト値
            except Exception as e:
                logger.warning(f"Failed to merge Funding Rate data: {e}")
                df["FundingRate"] = 0.0  # デフォルト値
        else:
            df["FundingRate"] = 0.0  # リポジトリが無い場合のデフォルト値

        return df

    def _convert_oi_to_dataframe(self, oi_data: List[OpenInterestData]) -> pd.DataFrame:
        """
        OpenInterestDataリストをpandas.DataFrameに変換

        Args:
            oi_data: OpenInterestDataオブジェクトのリスト

        Returns:
            Open InterestのDataFrame
        """
        data = {"OpenInterest": [r.open_interest_value for r in oi_data]}
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
        data = {"FundingRate": [r.funding_rate for r in fr_data]}
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([r.funding_timestamp for r in fr_data])
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        DataFrameの整合性をチェック

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: DataFrameが無効な場合
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        # 必要なカラムの存在確認
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # データ型の確認
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        # NaN値のチェック
        if df.isnull().any().any():
            raise ValueError("DataFrame contains NaN values")

    def _validate_extended_dataframe(self, df: pd.DataFrame) -> None:
        """
        拡張されたDataFrame（OI/FR含む）の整合性をチェック

        Args:
            df: 検証対象のDataFrame

        Raises:
            ValueError: DataFrameが無効な場合
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        # 必要なカラムの存在確認（拡張版）
        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "OpenInterest",
            "FundingRate",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # データ型の確認
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")

        # NaN値のチェック（OI/FRは0で埋められているはずなので、NaNがあれば問題）
        if df.isnull().any().any():
            logger.warning("DataFrame contains NaN values, filling with default values")
            # OI/FRのNaN値を0で埋める
            df["OpenInterest"] = df["OpenInterest"].fillna(0.0)
            df["FundingRate"] = df["FundingRate"].fillna(0.0)

            # OHLCV部分にNaNがある場合はエラー
            ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
            if df[ohlcv_cols].isnull().any().any():
                raise ValueError("OHLCV data contains NaN values")

    def validate_data_for_strategy(
        self, df: pd.DataFrame, strategy_config: dict
    ) -> None:
        """
        戦略固有のデータ検証

        Args:
            df: 検証対象のDataFrame
            strategy_config: 戦略設定

        Raises:
            ValueError: データが戦略要件を満たさない場合
        """
        # 最小データ数の確認
        min_periods = self._calculate_min_periods(strategy_config)
        if len(df) < min_periods:
            raise ValueError(
                f"Insufficient data: {len(df)} rows, but strategy requires at least {min_periods} rows"
            )

    def _calculate_min_periods(self, strategy_config: dict) -> int:
        """
        戦略に必要な最小期間数を計算

        Args:
            strategy_config: 戦略設定

        Returns:
            必要な最小期間数
        """
        min_periods = 1

        # SMA戦略の場合
        if strategy_config.get("strategy_type") == "SMA_CROSS":
            params = strategy_config.get("parameters", {})
            n1 = params.get("n1", 20)
            n2 = params.get("n2", 50)
            min_periods = max(n1, n2)

        # RSI戦略の場合
        elif strategy_config.get("strategy_type") == "RSI":
            params = strategy_config.get("parameters", {})
            period = params.get("period", 14)
            min_periods = period

        # MACD戦略の場合
        elif strategy_config.get("strategy_type") == "MACD":
            params = strategy_config.get("parameters", {})
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            min_periods = slow_period + signal_period

        return min_periods

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        データの概要情報を取得（OI/FR含む）

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        if df.empty:
            return {"error": "No data available"}

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
        if "OpenInterest" in df.columns:
            summary["open_interest_stats"] = {
                "average": float(df["OpenInterest"].mean()),
                "min": float(df["OpenInterest"].min()),
                "max": float(df["OpenInterest"].max()),
                "first": float(df["OpenInterest"].iloc[0]),
                "last": float(df["OpenInterest"].iloc[-1]),
            }

        if "FundingRate" in df.columns:
            summary["funding_rate_stats"] = {
                "average": float(df["FundingRate"].mean()),
                "min": float(df["FundingRate"].min()),
                "max": float(df["FundingRate"].max()),
                "first": float(df["FundingRate"].iloc[0]),
                "last": float(df["FundingRate"].iloc[-1]),
            }

        return summary
