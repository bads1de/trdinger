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
        logger.info(
            f"📊 データマージ開始 - OHLCV: {len(df)}行, 期間: {start_date} - {end_date}"
        )

        # Open Interestデータをマージ
        if self.oi_repo:
            try:
                oi_data = self.oi_repo.get_open_interest_data(
                    symbol=symbol, start_time=start_date, end_time=end_date
                )
                logger.info(
                    f"📈 取得したOIデータ件数: {len(oi_data) if oi_data else 0}"
                )

                if oi_data:
                    oi_df = self._convert_oi_to_dataframe(oi_data)
                    logger.info(
                        f"📈 OI DataFrame: {len(oi_df)}行, 期間: {oi_df.index.min()} - {oi_df.index.max()}"
                    )

                    # toleranceを設定（1日以内のデータのみ使用）
                    df = pd.merge_asof(
                        df.sort_index(),
                        oi_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                        tolerance=pd.Timedelta(days=1),
                    )

                    valid_oi_count = df["open_interest"].notna().sum()
                    logger.info(
                        f"📈 OIデータマージ完了: {valid_oi_count}/{len(df)}行に値あり ({valid_oi_count/len(df)*100:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"⚠️ シンボル {symbol} のOpen Interestデータが見つかりませんでした。"
                    )
                    df["open_interest"] = pd.NA
            except Exception as e:
                logger.warning(
                    f"❌ Open Interestデータのマージ中にエラーが発生しました: {e}"
                )
                df["open_interest"] = pd.NA
        else:
            logger.info("ℹ️ OIリポジトリが設定されていません")
            df["open_interest"] = pd.NA

        # Funding Rateデータをマージ
        if self.fr_repo:
            try:
                fr_data = self.fr_repo.get_funding_rate_data(
                    symbol=symbol, start_time=start_date, end_time=end_date
                )
                logger.info(
                    f"💰 取得したFRデータ件数: {len(fr_data) if fr_data else 0}"
                )

                if fr_data:
                    fr_df = self._convert_fr_to_dataframe(fr_data)
                    logger.info(
                        f"💰 FR DataFrame: {len(fr_df)}行, 期間: {fr_df.index.min()} - {fr_df.index.max()}"
                    )

                    # toleranceを設定（12時間以内のデータのみ使用、Funding Rateは8時間間隔）
                    df = pd.merge_asof(
                        df.sort_index(),
                        fr_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                        tolerance=pd.Timedelta(hours=12),
                    )

                    valid_fr_count = df["funding_rate"].notna().sum()
                    logger.info(
                        f"💰 FRデータマージ完了: {valid_fr_count}/{len(df)}行に値あり ({valid_fr_count/len(df)*100:.1f}%)"
                    )
                else:
                    logger.warning(
                        f"⚠️ シンボル {symbol} のFunding Rateデータが見つかりませんでした。"
                    )
                    df["funding_rate"] = pd.NA
            except Exception as e:
                logger.warning(
                    f"❌ Funding Rateデータのマージ中にエラーが発生しました: {e}"
                )
                df["funding_rate"] = pd.NA
        else:
            logger.info("ℹ️ FRリポジトリが設定されていません")
            df["funding_rate"] = pd.NA

        # 欠損値を前方データで埋め、それでも残る場合は0で埋める
        if "open_interest" in df.columns:
            # FutureWarningを回避するため、明示的に型を指定
            oi_series = df["open_interest"].astype("float64")
            df["open_interest"] = oi_series.ffill().fillna(0.0)
        if "funding_rate" in df.columns:
            # FutureWarningを回避するため、明示的に型を指定
            fr_series = df["funding_rate"].astype("float64")
            df["funding_rate"] = fr_series.ffill().fillna(0.0)

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
        # Fear & Greedデータをマージ
        if self.fear_greed_repo:
            try:
                fear_greed_data = self.fear_greed_repo.get_fear_greed_data(
                    start_time=start_date, end_time=end_date
                )
                logger.info(
                    f"😨 取得したFear & Greedデータ件数: {len(fear_greed_data) if fear_greed_data else 0}"
                )

                if fear_greed_data:
                    fear_greed_df = self._convert_fear_greed_to_dataframe(
                        fear_greed_data
                    )
                    logger.info(
                        f"😨 Fear & Greed DataFrame: {len(fear_greed_df)}行, 期間: {fear_greed_df.index.min()} - {fear_greed_df.index.max()}"
                    )

                    # toleranceを設定（3日以内のデータのみ使用、Fear & Greedは1日間隔）
                    df = pd.merge_asof(
                        df.sort_index(),
                        fear_greed_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                        tolerance=pd.Timedelta(days=3),
                    )

                    valid_fg_count = df["fear_greed_value"].notna().sum()
                    logger.info(
                        f"😨 Fear & Greedデータマージ完了: {valid_fg_count}/{len(df)}行に値あり ({valid_fg_count/len(df)*100:.1f}%)"
                    )
                else:
                    logger.warning("⚠️ Fear & Greedデータが見つかりませんでした。")
                    df["fear_greed_value"] = pd.NA
                    df["fear_greed_classification"] = pd.NA
            except Exception as e:
                logger.warning(
                    f"❌ Fear & Greedデータのマージ中にエラーが発生しました: {e}"
                )
                df["fear_greed_value"] = pd.NA
                df["fear_greed_classification"] = pd.NA
        else:
            logger.info("ℹ️ Fear & Greedリポジトリが設定されていません")
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA

        return df

    def _improve_data_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データ補間の改善

        Args:
            df: 対象のDataFrame

        Returns:
            補間処理されたDataFrame
        """
        logger.info("🔧 データ補間処理を開始")

        # Open Interest: forward fillで補間、残りは0で埋める
        if "open_interest" in df.columns:
            before_count = df["open_interest"].notna().sum()
            oi_series = df["open_interest"].astype("float64")
            df["open_interest"] = oi_series.ffill().fillna(0.0)
            after_count = df["open_interest"].notna().sum()
            logger.info(f"📈 OI補間: {before_count} → {after_count} 行")

        # Funding Rate: forward fillで補間、残りは0で埋める
        if "funding_rate" in df.columns:
            before_count = df["funding_rate"].notna().sum()
            fr_series = df["funding_rate"].astype("float64")
            df["funding_rate"] = fr_series.ffill().fillna(0.0)
            after_count = df["funding_rate"].notna().sum()
            logger.info(f"💰 FR補間: {before_count} → {after_count} 行")

        # Fear & Greed: forward fillで補間、残りは中立値50で埋める
        if "fear_greed_value" in df.columns:
            before_count = df["fear_greed_value"].notna().sum()
            fg_series = df["fear_greed_value"].astype("float64")
            df["fear_greed_value"] = fg_series.ffill().fillna(50.0)  # 中立値50
            after_count = df["fear_greed_value"].notna().sum()
            logger.info(f"😨 Fear & Greed値補間: {before_count} → {after_count} 行")

        if "fear_greed_classification" in df.columns:
            before_count = df["fear_greed_classification"].notna().sum()
            fg_class_series = df["fear_greed_classification"].astype("string")
            df["fear_greed_classification"] = fg_class_series.ffill().fillna("Neutral")
            after_count = df["fear_greed_classification"].notna().sum()
            logger.info(f"😨 Fear & Greed分類補間: {before_count} → {after_count} 行")

        # データ品質レポート
        self._log_data_quality_report(df)

        return df

    def _log_data_quality_report(self, df: pd.DataFrame) -> None:
        """
        データ品質レポートをログ出力

        Args:
            df: 対象のDataFrame
        """
        logger.info("📊 データ品質レポート:")
        logger.info(f"   総行数: {len(df)}")
        logger.info(f"   期間: {df.index.min()} - {df.index.max()}")

        # 各カラムのデータ品質
        for col in df.columns:
            if col in ["Open", "High", "Low", "Close", "Volume"]:
                continue  # OHLCVは必須なのでスキップ

            valid_count = df[col].notna().sum()
            coverage = valid_count / len(df) * 100
            logger.info(f"   {col}: {valid_count}/{len(df)} 行 ({coverage:.1f}%)")

        return df

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

        # 4. Fear & Greedデータを統合
        df = self._merge_fear_greed_data(df, start_date, end_date)

        # 5. データ補間の改善
        df = self._improve_data_interpolation(df)

        # 6. データの整合性チェックとソート
        self._validate_ml_training_dataframe(df)
        df = df.sort_index()  # 時系列順にソート

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
        # Fear & Greedデータをマージ
        if self.fear_greed_repo:
            try:
                fear_greed_data = self.fear_greed_repo.get_fear_greed_data(
                    start_time=start_date, end_time=end_date
                )
                if fear_greed_data:
                    fear_greed_df = self._convert_fear_greed_to_dataframe(
                        fear_greed_data
                    )
                    df = pd.merge_asof(
                        df.sort_index(),
                        fear_greed_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="backward",
                    )
                else:
                    logger.warning("Fear & Greedデータが見つかりませんでした。")
                    df["fear_greed_value"] = pd.NA
                    df["fear_greed_classification"] = pd.NA
            except Exception as e:
                logger.warning(
                    f"Fear & Greedデータのマージ中にエラーが発生しました: {e}"
                )
                df["fear_greed_value"] = pd.NA
                df["fear_greed_classification"] = pd.NA
        else:
            logger.info("Fear & Greedリポジトリが設定されていません")
            df["fear_greed_value"] = pd.NA
            df["fear_greed_classification"] = pd.NA

        # 欠損値を前方データで埋め、それでも残る場合は中立値で埋める
        if "fear_greed_value" in df.columns:
            fg_series = df["fear_greed_value"].astype("float64")
            df["fear_greed_value"] = fg_series.ffill().fillna(50.0)  # 中立値50で埋める
        if "fear_greed_classification" in df.columns:
            fg_class_series = df["fear_greed_classification"].astype("string")
            df["fear_greed_classification"] = fg_class_series.ffill().fillna("Neutral")

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
        self._perform_common_validation(df, required_columns[:5])  # OHLCV部分のみ必須

        # NaN値のチェック
        # OHLCV部分にNaNがある場合はエラー
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df[ohlcv_cols].isnull().any().any():
            raise ValueError("OHLCVデータにNaN値が含まれています。")

        # 追加データのNaNは既に処理されているはずだが、念のためログ出力
        additional_cols = ["open_interest", "funding_rate", "fear_greed_value"]
        if any(col in df.columns for col in additional_cols):
            if df[additional_cols].isnull().any().any():
                logger.warning("追加データに予期せぬNaN値が残っています。")

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
            "open_interest",
            "funding_rate",
        ]
        self._perform_common_validation(df, required_columns)

        # NaN値のチェック
        # OHLCV部分にNaNがある場合はエラー
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        if df[ohlcv_cols].isnull().any().any():
            raise ValueError("OHLCVデータにNaN値が含まれています。")

        # OI/FRのNaNは既にffill/fillna(0.0)で処理されているはずだが、念のためログ出力
        if "open_interest" in df.columns and "funding_rate" in df.columns:
            if df[["open_interest", "funding_rate"]].isnull().any().any():
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
