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
from database.models import OHLCVData, OpenInterestData, FundingRateData


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
            raise ValueError("OHLCVRepositoryが初期化されていません。")
        # 1. OHLCVデータを取得
        ohlcv_data = self.ohlcv_repo.get_ohlcv_data(
            symbol=symbol, timeframe=timeframe, start_time=start_date, end_time=end_date
        )

        if not ohlcv_data:
            raise ValueError(
                f"{symbol} {timeframe}のOHLCVデータが見つかりませんでした。"
            )

        # 2. OHLCVデータをDataFrameに変換
        df = self._convert_to_dataframe(ohlcv_data)

        # 3. OI/FRデータを統合
        df = self._merge_additional_data(df, symbol, start_date, end_date)

        # 4. データの整合性チェックとソート
        self._validate_extended_dataframe(df)
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
                logger.info(f"取得したOIデータ件数: {len(oi_data) if oi_data else 0}")
                if oi_data:
                    oi_df = self._convert_oi_to_dataframe(oi_data)
                    logger.info(f"OI DataFrame作成完了: {len(oi_df)}行")
                    df = pd.merge_asof(
                        df.sort_index(),
                        oi_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                    logger.info(f"OIデータマージ完了: {df['OpenInterest'].notna().sum()}/{len(df)}行に値あり")
                else:
                    logger.warning(
                        f"シンボル {symbol} のOpen Interestデータが見つかりませんでした。"
                    )
                    df["OpenInterest"] = pd.NA  # プレースホルダーとしてNAを設定
            except Exception as e:
                logger.warning(
                    f"Open Interestデータのマージ中にエラーが発生しました: {e}"
                )
                df["OpenInterest"] = pd.NA
        else:
            logger.info("OIリポジトリが設定されていません")
            df["OpenInterest"] = pd.NA

        # Funding Rateデータをマージ
        if self.fr_repo:
            try:
                fr_data = self.fr_repo.get_funding_rate_data(
                    symbol=symbol, start_time=start_date, end_time=end_date
                )
                logger.info(f"取得したFRデータ件数: {len(fr_data) if fr_data else 0}")
                if fr_data:
                    fr_df = self._convert_fr_to_dataframe(fr_data)
                    logger.info(f"FR DataFrame作成完了: {len(fr_df)}行")
                    df = pd.merge_asof(
                        df.sort_index(),
                        fr_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                    logger.info(f"FRデータマージ完了: {df['FundingRate'].notna().sum()}/{len(df)}行に値あり")
                else:
                    logger.warning(
                        f"シンボル {symbol} のFunding Rateデータが見つかりませんでした。"
                    )
                    df["FundingRate"] = pd.NA
            except Exception as e:
                logger.warning(
                    f"Funding Rateデータのマージ中にエラーが発生しました: {e}"
                )
                df["FundingRate"] = pd.NA
        else:
            logger.info("FRリポジトリが設定されていません")
            df["FundingRate"] = pd.NA

        # 欠損値を前方データで埋め、それでも残る場合は0で埋める
        if "OpenInterest" in df.columns:
            df["OpenInterest"] = df["OpenInterest"].ffill().fillna(0.0)
        if "FundingRate" in df.columns:
            df["FundingRate"] = df["FundingRate"].ffill().fillna(0.0)

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

    def _perform_common_validation(
        self, df: pd.DataFrame, required_columns: List[str]
    ) -> None:
        """
        DataFrameの共通検証を実行

        Args:
            df: 検証対象のDataFrame
            required_columns: 必須カラムのリスト

        Raises
            ValueError: DataFrameが無効な場合
        """
        if df.empty:
            raise ValueError("DataFrameが空です。")

        # 必要なカラムの存在確認
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"必須カラムが見つかりません: {missing_columns}")

        # データ型の確認
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"カラム {col} は数値型である必要があります。")

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
            "OpenInterest",
            "FundingRate",
        ]
        self._perform_common_validation(df, required_columns)

        # NaN値のチェック
        # OHLCV部分にNaNがある場合はエラー
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df[ohlcv_cols].isnull().any().any():
            raise ValueError("OHLCVデータにNaN値が含まれています。")

        # OI/FRのNaNは既にffill/fillna(0.0)で処理されているはずだが、念のためログ出力
        if "OpenInterest" in df.columns and "FundingRate" in df.columns:
            if df[["OpenInterest", "FundingRate"]].isnull().any().any():
                logger.warning("OI/FRデータに予期せぬNaN値が残っています。")

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
