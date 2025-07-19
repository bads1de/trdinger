"""
外部市場データ特徴量計算クラス

外部市場データ（S&P 500、NASDAQ、DXY、VIX）から
マクロ経済環境や市場センチメントを捉える特徴量を計算します。
"""

import logging
import pandas as pd
from typing import Dict
from ....utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class ExternalMarketFeatureCalculator:
    """
    外部市場データ特徴量計算クラス

    外部市場データから特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        pass

    def calculate_external_market_features(
        self,
        df: pd.DataFrame,
        external_market_data: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """
        外部市場データ特徴量を計算

        Args:
            df: OHLCV価格データ
            external_market_data: 外部市場データ
            lookback_periods: 計算期間設定

        Returns:
            外部市場特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            if external_market_data is None or external_market_data.empty:
                logger.warning("外部市場データが空です。自動取得を試行します。")
                external_market_data = self._fetch_external_market_data_fallback(df)

                if external_market_data is None or external_market_data.empty:
                    logger.warning(
                        "外部市場データの自動取得に失敗しました。デフォルト特徴量を生成します。"
                    )
                    return self._generate_default_external_features(result_df)

            # 外部市場データをピボットして各シンボルを列にする
            if "symbol" in external_market_data.columns:
                # インデックスがtimestampの場合とdata_timestampカラムがある場合の両方に対応
                if (
                    external_market_data.index.name == "timestamp"
                    or "timestamp" in str(type(external_market_data.index))
                ):
                    # インデックスがtimestampの場合
                    # インデックスをリセットして列に変換し、その列名をpivot_tableに渡すことで、
                    # 型チェッカーが解決できる形にする
                    temp_df = external_market_data.reset_index()
                    index_name = temp_df.columns[0]
                    pivot_data = temp_df.pivot_table(
                        index=index_name,
                        columns="symbol",
                        values="close",
                        aggfunc="last",
                    )
                elif "data_timestamp" in external_market_data.columns:
                    # data_timestampカラムがある場合
                    pivot_data = external_market_data.pivot_table(
                        index="data_timestamp",
                        columns="symbol",
                        values="close",
                        aggfunc="last",
                    )
                else:
                    logger.warning(
                        "外部市場データに適切なタイムスタンプカラムがありません。デフォルト特徴量を生成します。"
                    )
                    return self._generate_default_external_features(result_df)

                # ピボット結果の検証
                if pivot_data.empty:
                    logger.warning(
                        "外部市場データのピボット結果が空です。デフォルト特徴量を生成します。"
                    )
                    return self._generate_default_external_features(result_df)
            else:
                logger.warning(
                    "外部市場データにsymbolカラムがありません。デフォルト特徴量を生成します。"
                )
                return self._generate_default_external_features(result_df)

            # インデックスを合わせてマージ
            merged_df = result_df.join(pivot_data, how="left", rsuffix="_ext")

            # 各外部市場指標の特徴量を計算
            result_df = self._calculate_sp500_features(
                result_df, merged_df, lookback_periods
            )
            result_df = self._calculate_nasdaq_features(
                result_df, merged_df, lookback_periods
            )
            result_df = self._calculate_dxy_features(
                result_df, merged_df, lookback_periods
            )
            result_df = self._calculate_vix_features(
                result_df, merged_df, lookback_periods
            )
            result_df = self._calculate_composite_features(
                result_df, merged_df, lookback_periods
            )

            logger.debug("外部市場特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"外部市場特徴量計算エラー: {e}")
            return df

    def _calculate_sp500_features(
        self,
        result_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """S&P 500関連特徴量を計算"""
        sp500_col = "^GSPC"
        if sp500_col not in merged_df.columns:
            return result_df

        # 前方補完
        merged_df[sp500_col] = merged_df[sp500_col].ffill()

        # S&P 500リターン
        result_df["SP500_Return"] = DataValidator.safe_pct_change(merged_df[sp500_col])

        # S&P 500移動平均
        result_df["SP500_MA_20"] = DataValidator.safe_rolling_mean(
            merged_df[sp500_col], window=20
        )
        result_df["SP500_MA_50"] = DataValidator.safe_rolling_mean(
            merged_df[sp500_col], window=50
        )

        # S&P 500トレンド
        result_df["SP500_Trend"] = (
            DataValidator.safe_divide(
                result_df["SP500_MA_20"], result_df["SP500_MA_50"], default_value=1.0
            )
            - 1
        )

        # S&P 500ボラティリティ
        result_df["SP500_Volatility"] = DataValidator.safe_rolling_std(
            result_df["SP500_Return"], window=20
        )

        return result_df

    def _calculate_nasdaq_features(
        self,
        result_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """NASDAQ関連特徴量を計算"""
        nasdaq_col = "^IXIC"
        if nasdaq_col not in merged_df.columns:
            return result_df

        # 前方補完
        merged_df[nasdaq_col] = merged_df[nasdaq_col].ffill()

        # NASDAQリターン
        result_df["NASDAQ_Return"] = DataValidator.safe_pct_change(
            merged_df[nasdaq_col]
        )

        # NASDAQ移動平均
        result_df["NASDAQ_MA_20"] = DataValidator.safe_rolling_mean(
            merged_df[nasdaq_col], window=20
        )

        # NASDAQトレンド
        result_df["NASDAQ_Trend"] = DataValidator.safe_pct_change(
            result_df["NASDAQ_MA_20"], periods=5
        )

        return result_df

    def _calculate_dxy_features(
        self,
        result_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """DXY（ドル指数）関連特徴量を計算"""
        dxy_col = "DX-Y.NYB"
        if dxy_col not in merged_df.columns:
            return result_df

        # 前方補完
        merged_df[dxy_col] = merged_df[dxy_col].ffill()

        # DXYリターン
        result_df["DXY_Return"] = DataValidator.safe_pct_change(merged_df[dxy_col])

        # DXY移動平均
        result_df["DXY_MA_20"] = DataValidator.safe_rolling_mean(
            merged_df[dxy_col], window=20
        )

        # DXY強度（正規化）
        result_df["DXY_Strength"] = DataValidator.safe_normalize(
            merged_df[dxy_col], window=50, default_value=0.0
        )

        return result_df

    def _calculate_vix_features(
        self,
        result_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """VIX（恐怖指数）関連特徴量を計算"""
        vix_col = "^VIX"
        if vix_col not in merged_df.columns:
            return result_df

        # 前方補完
        merged_df[vix_col] = merged_df[vix_col].ffill()

        # VIXレベル
        result_df["VIX_Level"] = merged_df[vix_col]

        # VIX変化率
        result_df["VIX_Change"] = DataValidator.safe_pct_change(merged_df[vix_col])

        # VIX移動平均
        result_df["VIX_MA_10"] = DataValidator.safe_rolling_mean(
            merged_df[vix_col], window=10
        )

        # VIXスパイク（高ボラティリティ期間）
        vix_threshold = merged_df[vix_col].quantile(0.8)  # 上位20%
        result_df["VIX_Spike"] = (merged_df[vix_col] > vix_threshold).astype(int)

        # VIX正規化
        result_df["VIX_Normalized"] = DataValidator.safe_normalize(
            merged_df[vix_col], window=50, default_value=0.0
        )

        return result_df

    def _calculate_composite_features(
        self,
        result_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        lookback_periods: Dict[str, int],
    ) -> pd.DataFrame:
        """複合特徴量を計算"""
        # 必要な列が存在するかチェック
        required_cols = ["^GSPC", "^IXIC", "DX-Y.NYB", "^VIX"]
        available_cols = [col for col in required_cols if col in merged_df.columns]

        if len(available_cols) < 2:
            return result_df

        # 株式市場強度（S&P 500 + NASDAQ）
        if "^GSPC" in merged_df.columns and "^IXIC" in merged_df.columns:
            sp500_norm = DataValidator.safe_normalize(
                merged_df["^GSPC"], window=50, default_value=0.0
            )
            nasdaq_norm = DataValidator.safe_normalize(
                merged_df["^IXIC"], window=50, default_value=0.0
            )
            result_df["Equity_Strength"] = (sp500_norm + nasdaq_norm) / 2

        # リスクオン・リスクオフ指標（VIX vs 株式）
        if "^VIX" in merged_df.columns and "^GSPC" in merged_df.columns:
            vix_inv = DataValidator.safe_divide(1, merged_df["^VIX"], default_value=0.0)
            sp500_norm = DataValidator.safe_normalize(
                merged_df["^GSPC"], window=50, default_value=0.0
            )
            result_df["Risk_On_Off"] = sp500_norm * vix_inv

        # ドル強度 vs 株式
        if "DX-Y.NYB" in merged_df.columns and "^GSPC" in merged_df.columns:
            dxy_norm = DataValidator.safe_normalize(
                merged_df["DX-Y.NYB"], window=50, default_value=0.0
            )
            sp500_norm = DataValidator.safe_normalize(
                merged_df["^GSPC"], window=50, default_value=0.0
            )
            result_df["USD_Equity_Divergence"] = dxy_norm - sp500_norm

        return result_df

    def _fetch_external_market_data_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        外部市場データの自動取得（フォールバック）

        Args:
            df: 基本データフレーム（時間範囲の参考用）

        Returns:
            取得した外部市場データ、または空のDataFrame
        """
        try:
            from ...data_collection.external_market.external_market_service import (
                ExternalMarketService,
            )

            # データフレームの時間範囲を取得
            if df.empty or not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("基本データフレームが空か、DatetimeIndexではありません")
                return pd.DataFrame()

            start_date = df.index.min().strftime("%Y-%m-%d")
            end_date = df.index.max().strftime("%Y-%m-%d")

            logger.info(f"外部市場データを自動取得中: {start_date} ～ {end_date}")

            # ExternalMarketServiceを使用してデータを取得
            service = ExternalMarketService()
            external_data = service.fetch_external_market_data(
                period=None,  # start_date/end_dateを優先
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )

            if external_data:
                # 辞書リストをDataFrameに変換
                external_df = pd.DataFrame(external_data)
                if "data_timestamp" in external_df.columns:
                    external_df["data_timestamp"] = pd.to_datetime(
                        external_df["data_timestamp"]
                    )
                    external_df = external_df.set_index("data_timestamp")

                logger.info(f"外部市場データを {len(external_df)} 件取得しました")
                return external_df
            else:
                logger.warning("外部市場データの取得結果が空でした")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"外部市場データの自動取得エラー: {e}")
            return pd.DataFrame()

    def _generate_default_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        デフォルトの外部市場特徴量を生成（すべて0で初期化）

        Args:
            df: 基本データフレーム

        Returns:
            デフォルト特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # 外部市場特徴量名のリストを取得
            feature_names = self.get_feature_names()

            # すべての特徴量を0で初期化
            for feature_name in feature_names:
                result_df[feature_name] = 0.0

            logger.info(
                f"デフォルト外部市場特徴量を {len(feature_names)} 個生成しました"
            )
            return result_df

        except Exception as e:
            logger.error(f"デフォルト外部市場特徴量生成エラー: {e}")
            return df

    def get_feature_names(self) -> list:
        """
        生成される外部市場特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # S&P 500特徴量
            "SP500_Return",
            "SP500_MA_20",
            "SP500_MA_50",
            "SP500_Trend",
            "SP500_Volatility",
            # NASDAQ特徴量
            "NASDAQ_Return",
            "NASDAQ_MA_20",
            "NASDAQ_Trend",
            # DXY特徴量
            "DXY_Return",
            "DXY_MA_20",
            "DXY_Strength",
            # VIX特徴量
            "VIX_Level",
            "VIX_Change",
            "VIX_MA_10",
            "VIX_Spike",
            "VIX_Normalized",
            # 複合特徴量
            "Equity_Strength",
            "Risk_On_Off",
            "USD_Equity_Divergence",
        ]
