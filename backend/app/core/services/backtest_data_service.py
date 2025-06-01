"""
バックテスト用データ変換サービス

backtesting.pyライブラリで使用するためのデータ変換機能を提供します。
"""

import pandas as pd
from datetime import datetime
from typing import List
from database.repositories.ohlcv_repository import OHLCVRepository
from database.models import OHLCVData


class BacktestDataService:
    """
    backtesting.py用のデータ変換サービス

    既存のOHLCVデータをbacktesting.pyライブラリで使用可能な
    pandas.DataFrame形式に変換します。
    """

    def __init__(self, ohlcv_repo: OHLCVRepository = None):
        """
        初期化

        Args:
            ohlcv_repo: OHLCVリポジトリ（テスト時にモックを注入可能）
        """
        self.ohlcv_repo = ohlcv_repo

    def get_ohlcv_for_backtest(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        OHLCVデータをbacktesting.py形式に変換

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
        df.index = pd.to_datetime([r.timestamp for r in ohlcv_data])

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
        データの概要情報を取得

        Args:
            df: 対象のDataFrame

        Returns:
            データ概要の辞書
        """
        if df.empty:
            return {"error": "No data available"}

        return {
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
