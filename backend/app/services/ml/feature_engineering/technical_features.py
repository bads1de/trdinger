"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
テクニカル特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator(BaseFeatureCalculator):
    """
    テクニカル指標特徴量計算クラス

    従来のテクニカル指標特徴量を計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__()

    def calculate_features(
        self, df: pd.DataFrame, config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        テクニカル特徴量を計算（BaseFeatureCalculatorの抽象メソッド実装）

        Args:
            df: OHLCV価格データ
            config: 計算設定（lookback_periodsを含む）

        Returns:
            テクニカル特徴量が追加されたDataFrame
        """
        lookback_periods = config.get("lookback_periods", {})

        # 複数のテクニカル特徴量を順次計算
        result_df = self.calculate_market_regime_features(df, lookback_periods)
        result_df = self.calculate_momentum_features(result_df, lookback_periods)

        return result_df

    def calculate_market_regime_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        市場レジーム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            市場レジーム特徴量が追加されたDataFrame
        """
        try:
            if not self.validate_input_data(df, ["Close", "High", "Low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # トレンド強度（pandas-ta SMA使用）
            short_ma = lookback_periods.get("short_ma", 10)
            long_ma = lookback_periods.get("long_ma", 50)

            import pandas_ta as ta

            ma_short = ta.sma(result_df["Close"], length=short_ma)
            ma_long = ta.sma(result_df["Close"], length=long_ma)

            # Trend_Strength = (MA_short - MA_long) / MA_long
            if (
                ma_short is not None
                and ma_long is not None
                and isinstance(ma_short, pd.Series)
                and isinstance(ma_long, pd.Series)
            ):
                result_df["Trend_Strength"] = self.safe_ratio_calculation(
                    ma_short - ma_long, ma_long
                )
            else:
                result_df["Trend_Strength"] = 0.0

            # レンジ相場判定（pandas-ta MAX/MIN使用）
            volatility_period = lookback_periods.get("volatility", 20)
            high_vals = result_df["High"]
            low_vals = result_df["Low"]
            high_20 = high_vals.rolling(window=volatility_period).max()
            low_20 = low_vals.rolling(window=volatility_period).min()

            result_df["Range_Bound_Ratio"] = self.safe_ratio_calculation(
                result_df["Close"] - low_20, high_20 - low_20, fill_value=0.5
            )

            # ブレイクアウト強度（直前の高値・安値を使用）
            prev_high_20 = high_20.shift(1)
            prev_low_20 = low_20.shift(1)
            result_df["Breakout_Strength"] = np.where(
                result_df["Close"] > prev_high_20,
                (result_df["Close"] - prev_high_20) / prev_high_20,
                np.where(
                    result_df["Close"] < prev_low_20,
                    (prev_low_20 - result_df["Close"]) / prev_low_20,
                    0.0,
                ),
            )

            # 市場効率性（価格のランダムウォーク度）
            returns = result_df["Close"].pct_change().fillna(0)

            def safe_correlation(x):
                try:
                    if len(x) < 3:  # 最低3つのデータポイントが必要
                        return 0.0
                    x_clean = x.dropna()
                    if len(x_clean) < 3:
                        return 0.0
                    x1, x2 = x_clean[:-1], x_clean[1:]
                    if (
                        len(x1) == 0
                        or len(x2) == 0
                        or np.std(x1) == 0
                        or np.std(x2) == 0
                    ):
                        return 0.0
                    corr_matrix = np.corrcoef(x1, x2)
                    if np.isnan(corr_matrix[0, 1]):
                        return 0.0
                    return corr_matrix[0, 1]
                except Exception:
                    return 0.0

            result_df["Market_Efficiency"] = (
                returns.rolling(window=volatility_period, min_periods=3)
                .apply(safe_correlation)
                .fillna(0.0)
            )

            logger.debug("市場レジーム特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"市場レジーム特徴量計算エラー: {e}")
            return df

    def calculate_momentum_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        モメンタム特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            モメンタム特徴量が追加されたDataFrame
        """
        try:
            if not self.validate_input_data(df, ["Close", "High", "Low"]):
                return df

            result_df = self.create_result_dataframe(df)

            # RSI（pandas-ta使用）
            rsi_values = ta.rsi(result_df["Close"], length=14)
            if rsi_values is not None:
                result_df["RSI"] = rsi_values.fillna(50.0)
            else:
                result_df["RSI"] = 50.0

            # MACD（pandas-ta使用）
            macd_result = ta.macd(result_df["Close"], fast=12, slow=26, signal=9)
            if macd_result is not None:
                result_df["MACD"] = macd_result["MACD_12_26_9"].fillna(0.0)
                result_df["MACD_Signal"] = macd_result["MACDs_12_26_9"].fillna(0.0)
                result_df["MACD_Histogram"] = macd_result["MACDh_12_26_9"].fillna(0.0)
            else:
                result_df["MACD"] = 0.0
                result_df["MACD_Signal"] = 0.0
                result_df["MACD_Histogram"] = 0.0

            # ストキャスティクス（pandas-ta使用）
            stoch_result = ta.stoch(
                high=result_df["High"],
                low=result_df["Low"],
                close=result_df["Close"],
                k=14,
                d=3,
                smooth_k=3,
            )
            if stoch_result is not None:
                result_df["Stochastic_K"] = stoch_result["STOCHk_14_3_3"].fillna(50.0)
                result_df["Stochastic_D"] = stoch_result["STOCHd_14_3_3"].fillna(50.0)
            else:
                result_df["Stochastic_K"] = 50.0
                result_df["Stochastic_D"] = 50.0

            # ウィリアムズ%R（pandas-ta使用）
            willr_values = ta.willr(
                high=result_df["High"],
                low=result_df["Low"],
                close=result_df["Close"],
                length=14,
            )
            if willr_values is not None:
                result_df["Williams_R"] = willr_values.fillna(-50.0)
            else:
                result_df["Williams_R"] = -50.0

            # CCI（Commodity Channel Index）（pandas-ta使用）
            cci_values = ta.cci(
                high=result_df["High"],
                low=result_df["Low"],
                close=result_df["Close"],
                length=20,
            )
            if cci_values is not None:
                result_df["CCI"] = cci_values.fillna(0.0)
            else:
                result_df["CCI"] = 0.0

            # ROC（Rate of Change）（pandas-ta使用）
            roc_values = ta.roc(result_df["Close"], length=12)
            if roc_values is not None:
                result_df["ROC"] = roc_values.fillna(0.0)
            else:
                result_df["ROC"] = 0.0

            # モメンタム（pandas-ta使用）
            momentum_values = ta.mom(result_df["Close"], length=10)
            if momentum_values is not None:
                result_df["Momentum"] = momentum_values.fillna(0.0)
            else:
                result_df["Momentum"] = 0.0

            logger.debug("モメンタム特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"モメンタム特徴量計算エラー: {e}")
            return df


    def safe_ratio_calculation(self, numerator: pd.Series | Any, denominator: pd.Series | Any, fill_value: float = 0.0) -> pd.Series:
        """
        ゼロ除算を防ぐための安全な比率計算

        Args:
            numerator: 分子
            denominator: 分母
            fill_value: ゼロ除算時の埋め値

        Returns:
            計算結果のSeries
        """
        if (
            denominator is None
            or numerator is None
            or not isinstance(numerator, pd.Series)
            or not isinstance(denominator, pd.Series)
        ):
            length = len(numerator) if isinstance(numerator, pd.Series) else 0
            return pd.Series([fill_value] * length)

        # ゼロ除算を防ぐ
        ratio = numerator / denominator.replace(0, np.nan)
        return ratio.replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series | Any:
        """RSIを計算（内部使用）- pandas-ta使用"""
        import pandas_ta as ta

        rsi_values = ta.rsi(df["Close"], length=period)
        if rsi_values is None or not isinstance(rsi_values, pd.Series):
            return pd.Series([50.0] * len(df))
        return rsi_values.fillna(50.0)

    def get_feature_names(self) -> list:
        """
        生成されるテクニカル特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        return [
            # 市場レジーム特徴量
            "Trend_Strength",
            "Range_Bound_Ratio",
            "Breakout_Strength",
            "Market_Efficiency",
            # モメンタム特徴量
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "Stochastic_K",
            "Stochastic_D",
            "Williams_R",
            "CCI",
            "ROC",
            "Momentum",
        ]


# 互換性のための別名（旧名: TechnicalFeatureEngineer）
class TechnicalFeatureEngineer(TechnicalFeatureCalculator):
    # 旧API互換
    def create_features(self, df: pd.DataFrame, config=None) -> pd.DataFrame:
        if config is None:
            config = {"lookback_periods": {}}
        return self.calculate_features(df, config)
