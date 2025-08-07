"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
高度なパターン認識特徴量を計算します。
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import talib

from .base_feature_calculator import BaseFeatureCalculator

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator(BaseFeatureCalculator):
    """
    テクニカル指標特徴量計算クラス

    従来のテクニカル指標と高度なパターン認識特徴量を計算します。
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
        result_df = self.calculate_pattern_features(result_df, lookback_periods)

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

            # トレンド強度
            short_ma = lookback_periods.get("short_ma", 10)
            long_ma = lookback_periods.get("long_ma", 50)

            ma_short = self.safe_rolling_mean_calculation(
                result_df["Close"], window=short_ma
            )
            ma_long = self.safe_rolling_mean_calculation(
                result_df["Close"], window=long_ma
            )

            result_df["Trend_Strength"] = self.safe_ratio_calculation(
                ma_short - ma_long, ma_long, default_value=0.0
            )

            # レンジ相場判定
            volatility_period = lookback_periods.get("volatility", 20)
            high_20 = (
                result_df["High"].rolling(window=volatility_period, min_periods=1).max()
            )
            low_20 = (
                result_df["Low"].rolling(window=volatility_period, min_periods=1).min()
            )

            result_df["Range_Bound_Ratio"] = self.safe_ratio_calculation(
                result_df["Close"] - low_20, high_20 - low_20, default_value=0.5
            )

            # ブレイクアウト強度
            result_df["Breakout_Strength"] = np.where(
                result_df["Close"] > high_20.shift(1),
                (result_df["Close"] - high_20.shift(1)) / high_20.shift(1),
                np.where(
                    result_df["Close"] < low_20.shift(1),
                    (low_20.shift(1) - result_df["Close"]) / low_20.shift(1),
                    0,
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

            # RSI（TA-lib使用）
            try:
                close_values = result_df["Close"].values.astype(np.float64)
                rsi_values = talib.RSI(close_values, timeperiod=14)
                result_df["RSI"] = pd.Series(rsi_values, index=result_df.index).fillna(
                    50.0
                )
            except Exception as e:
                logger.warning(f"TA-lib RSI計算エラー、フォールバック実装を使用: {e}")
                # フォールバック実装
                delta = result_df["Close"].diff().fillna(0.0).astype(float)
                gain = self.safe_rolling_mean_calculation(
                    delta.clip(lower=0), window=14
                )
                loss = self.safe_rolling_mean_calculation(
                    -delta.clip(upper=0), window=14
                )
                rs = self.safe_ratio_calculation(gain, loss, default_value=1.0)
                result_df["RSI"] = 100 - (100 / (1 + rs))

            # MACD（TA-lib使用）
            try:
                close_values = result_df["Close"].values.astype(np.float64)
                macd, macd_signal, macd_histogram = talib.MACD(
                    close_values, fastperiod=12, slowperiod=26, signalperiod=9
                )
                result_df["MACD"] = pd.Series(macd, index=result_df.index).fillna(0.0)
                result_df["MACD_Signal"] = pd.Series(
                    macd_signal, index=result_df.index
                ).fillna(0.0)
                result_df["MACD_Histogram"] = pd.Series(
                    macd_histogram, index=result_df.index
                ).fillna(0.0)
            except Exception as e:
                logger.warning(f"TA-lib MACD計算エラー、フォールバック実装を使用: {e}")
                # フォールバック実装
                ema_12 = result_df["Close"].ewm(span=12).mean()
                ema_26 = result_df["Close"].ewm(span=26).mean()
                result_df["MACD"] = ema_12 - ema_26
                result_df["MACD_Signal"] = result_df["MACD"].ewm(span=9).mean()
                result_df["MACD_Histogram"] = (
                    result_df["MACD"] - result_df["MACD_Signal"]
                )

            # ストキャスティクス（TA-lib使用）
            try:
                high_values = result_df["High"].values.astype(np.float64)
                low_values = result_df["Low"].values.astype(np.float64)
                close_values = result_df["Close"].values.astype(np.float64)
                slowk, slowd = talib.STOCH(
                    high_values,
                    low_values,
                    close_values,
                    fastk_period=14,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0,
                )
                result_df["Stochastic_K"] = pd.Series(
                    slowk, index=result_df.index
                ).fillna(50.0)
                result_df["Stochastic_D"] = pd.Series(
                    slowd, index=result_df.index
                ).fillna(50.0)
            except Exception as e:
                logger.warning(
                    f"TA-lib Stochastic計算エラー、フォールバック実装を使用: {e}"
                )
                # フォールバック実装
                period = 14
                low_14 = result_df["Low"].rolling(window=period, min_periods=1).min()
                high_14 = result_df["High"].rolling(window=period, min_periods=1).max()
                result_df["Stochastic_K"] = 100 * self.safe_ratio_calculation(
                    result_df["Close"] - low_14, high_14 - low_14, default_value=0.5
                )
                result_df["Stochastic_D"] = self.safe_rolling_mean_calculation(
                    result_df["Stochastic_K"], window=3
                )

            # ウィリアムズ%R（TA-lib使用）
            try:
                high_values = result_df["High"].values.astype(np.float64)
                low_values = result_df["Low"].values.astype(np.float64)
                close_values = result_df["Close"].values.astype(np.float64)
                willr_values = talib.WILLR(
                    high_values, low_values, close_values, timeperiod=14
                )
                result_df["Williams_R"] = pd.Series(
                    willr_values, index=result_df.index
                ).fillna(-50.0)
            except Exception as e:
                logger.warning(
                    f"TA-lib Williams %R計算エラー、フォールバック実装を使用: {e}"
                )
                # フォールバック実装（high_14, low_14は上のStochasticで計算済み）
                try:
                    high_14 = result_df["High"].rolling(window=14, min_periods=1).max()
                    low_14 = result_df["Low"].rolling(window=14, min_periods=1).min()
                    result_df["Williams_R"] = -100 * self.safe_ratio_calculation(
                        high_14 - result_df["Close"],
                        high_14 - low_14,
                        default_value=0.5,
                    )
                except Exception:
                    result_df["Williams_R"] = -50.0

            # CCI（Commodity Channel Index）（TA-lib使用）
            try:
                high_values = result_df["High"].values.astype(np.float64)
                low_values = result_df["Low"].values.astype(np.float64)
                close_values = result_df["Close"].values.astype(np.float64)
                cci_values = talib.CCI(
                    high_values, low_values, close_values, timeperiod=20
                )
                result_df["CCI"] = pd.Series(cci_values, index=result_df.index).fillna(
                    0.0
                )
            except Exception as e:
                logger.warning(f"TA-lib CCI計算エラー、フォールバック実装を使用: {e}")
                # フォールバック実装
                typical_price = (
                    result_df["High"] + result_df["Low"] + result_df["Close"]
                ) / 3
                sma_tp = self.safe_rolling_mean_calculation(typical_price, window=20)
                mad = typical_price.rolling(window=20, min_periods=1).apply(
                    lambda x: np.mean(np.abs(x - x.mean()))
                )
                result_df["CCI"] = self.safe_ratio_calculation(
                    typical_price - sma_tp, 0.015 * mad, default_value=0.0
                )

            # ROC（Rate of Change）（TA-lib使用）
            try:
                close_values = result_df["Close"].values.astype(np.float64)
                roc_values = talib.ROC(close_values, timeperiod=12)
                result_df["ROC"] = pd.Series(roc_values, index=result_df.index).fillna(
                    0.0
                )
            except Exception as e:
                logger.warning(f"TA-lib ROC計算エラー、フォールバック実装を使用: {e}")
                # フォールバック実装
                roc_change = self.safe_ratio_calculation(
                    result_df["Close"] - result_df["Close"].shift(12),
                    result_df["Close"].shift(12),
                    default_value=0.0,
                )
                result_df["ROC"] = roc_change * 100

            # モメンタム（TA-lib使用）
            try:
                close_values = result_df["Close"].values.astype(np.float64)
                momentum_values = talib.MOM(close_values, timeperiod=10)
                result_df["Momentum"] = pd.Series(
                    momentum_values, index=result_df.index
                ).fillna(0.0)
            except Exception as e:
                logger.warning(
                    f"TA-lib Momentum計算エラー、フォールバック実装を使用: {e}"
                )
                # フォールバック実装
                result_df["Momentum"] = self.safe_ratio_calculation(
                    result_df["Close"], result_df["Close"].shift(10), default_value=1.0
                )

            logger.debug("モメンタム特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"モメンタム特徴量計算エラー: {e}")
            return df

    def calculate_pattern_features(
        self, df: pd.DataFrame, lookback_periods: Dict[str, int]
    ) -> pd.DataFrame:
        """
        パターン認識特徴量を計算

        Args:
            df: OHLCV価格データ
            lookback_periods: 計算期間設定

        Returns:
            パターン認識特徴量が追加されたDataFrame
        """
        try:
            result_df = df.copy()

            # ダイバージェンス検出
            rsi = (
                result_df["RSI"]
                if "RSI" in result_df.columns
                else self._calculate_rsi(result_df)
            )

            # 価格とRSIのダイバージェンス
            price_trend = (
                result_df["Close"]
                .rolling(window=10)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            )
            rsi_trend = rsi.rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )

            # ベアダイバージェンス（価格上昇、RSI下降）
            result_df["Bear_Divergence"] = ((price_trend > 0) & (rsi_trend < 0)).astype(
                int
            )

            # ブルダイバージェンス（価格下降、RSI上昇）
            result_df["Bull_Divergence"] = ((price_trend < 0) & (rsi_trend > 0)).astype(
                int
            )

            # サポート・レジスタンス距離
            period = lookback_periods.get("volatility", 20)

            # 直近の高値・安値
            recent_high = result_df["High"].rolling(window=period).max()
            recent_low = result_df["Low"].rolling(window=period).min()

            result_df["Support_Distance"] = self.safe_ratio_calculation(
                result_df["Close"] - recent_low, result_df["Close"]
            )
            result_df["Resistance_Distance"] = self.safe_ratio_calculation(
                recent_high - result_df["Close"], result_df["Close"]
            )

            # ピボットポイント
            prev_high = result_df["High"].shift(1)
            prev_low = result_df["Low"].shift(1)
            prev_close = result_df["Close"].shift(1)

            pivot = (prev_high + prev_low + prev_close) / 3
            result_df["Pivot_Distance"] = self.safe_ratio_calculation(
                result_df["Close"] - pivot, pivot
            )

            # フィボナッチレベル
            swing_high = result_df["High"].rolling(window=period).max()
            swing_low = result_df["Low"].rolling(window=period).min()

            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                fib_price = swing_low + (swing_high - swing_low) * level
                result_df[f"Fib_{int(level*1000)}_Distance"] = (
                    self.safe_ratio_calculation(
                        abs(result_df["Close"] - fib_price), result_df["Close"]
                    )
                )

            # ギャップ分析
            gap = result_df["Open"] - result_df["Close"].shift(1)
            gap_pct = self.safe_ratio_calculation(gap, result_df["Close"].shift(1))

            result_df["Gap_Up"] = (pd.Series(gap_pct) > 0.01).astype(int)
            result_df["Gap_Down"] = (pd.Series(gap_pct) < -0.01).astype(int)
            result_df["Gap_Size"] = abs(gap_pct)

            logger.debug("パターン認識特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"パターン認識特徴量計算エラー: {e}")
            return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSIを計算（内部使用）- TA-lib使用"""
        try:
            close_values = df["Close"].values.astype(np.float64)
            rsi_values = talib.RSI(close_values, timeperiod=period)
            return pd.Series(rsi_values, index=df.index).fillna(50.0)
        except Exception as e:
            logger.warning(f"TA-lib RSI計算エラー、フォールバック実装を使用: {e}")
            # フォールバック実装
            delta = df["Close"].diff().fillna(0.0).astype(float)
            gain = (delta.clip(lower=0)).rolling(window=period).mean()
            loss = (-delta.clip(upper=0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))

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
            # パターン特徴量
            "Bear_Divergence",
            "Bull_Divergence",
            "Support_Distance",
            "Resistance_Distance",
            "Pivot_Distance",
            "Fib_236_Distance",
            "Fib_382_Distance",
            "Fib_500_Distance",
            "Fib_618_Distance",
            "Fib_786_Distance",
            "Gap_Up",
            "Gap_Down",
            "Gap_Size",
        ]
