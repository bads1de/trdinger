"""
テクニカル指標特徴量計算クラス

従来のテクニカル指標（RSI、MACD、ストキャスティクスなど）と
高度なパターン認識特徴量を計算します。
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from ....utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator:
    """
    テクニカル指標特徴量計算クラス

    従来のテクニカル指標と高度なパターン認識特徴量を計算します。
    """

    def __init__(self):
        """初期化"""

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
            result_df = df.copy()

            # トレンド強度
            short_ma = lookback_periods.get("short_ma", 10)
            long_ma = lookback_periods.get("long_ma", 50)

            ma_short = DataValidator.safe_rolling_mean(
                result_df["Close"], window=short_ma
            )
            ma_long = DataValidator.safe_rolling_mean(
                result_df["Close"], window=long_ma
            )

            result_df["Trend_Strength"] = DataValidator.safe_divide(
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

            result_df["Range_Bound_Ratio"] = DataValidator.safe_divide(
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
            result_df = df.copy()

            # RSI（安全な計算）
            delta = result_df["Close"].diff().fillna(0.0).astype(float)
            gain = DataValidator.safe_rolling_mean(delta.clip(lower=0), window=14)
            loss = DataValidator.safe_rolling_mean(-delta.clip(upper=0), window=14)
            rs = DataValidator.safe_divide(gain, loss, default_value=1.0)
            result_df["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = result_df["Close"].ewm(span=12).mean()
            ema_26 = result_df["Close"].ewm(span=26).mean()
            result_df["MACD"] = ema_12 - ema_26
            result_df["MACD_Signal"] = result_df["MACD"].ewm(span=9).mean()
            result_df["MACD_Histogram"] = result_df["MACD"] - result_df["MACD_Signal"]

            # ストキャスティクス（安全な計算）
            period = 14
            low_14 = result_df["Low"].rolling(window=period, min_periods=1).min()
            high_14 = result_df["High"].rolling(window=period, min_periods=1).max()

            result_df["Stochastic_K"] = 100 * DataValidator.safe_divide(
                result_df["Close"] - low_14, high_14 - low_14, default_value=0.5
            )
            result_df["Stochastic_D"] = DataValidator.safe_rolling_mean(
                result_df["Stochastic_K"], window=3
            )

            # ウィリアムズ%R（安全な計算）
            result_df["Williams_R"] = -100 * DataValidator.safe_divide(
                high_14 - result_df["Close"], high_14 - low_14, default_value=0.5
            )

            # CCI（Commodity Channel Index）（安全な計算）
            typical_price = (
                result_df["High"] + result_df["Low"] + result_df["Close"]
            ) / 3
            sma_tp = DataValidator.safe_rolling_mean(typical_price, window=20)
            mad = typical_price.rolling(window=20, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            result_df["CCI"] = DataValidator.safe_divide(
                typical_price - sma_tp, 0.015 * mad, default_value=0.0
            )

            # ROC（Rate of Change）（安全な計算）
            result_df["ROC"] = (
                DataValidator.safe_pct_change(result_df["Close"], periods=12) * 100
            )

            # モメンタム（安全な計算）
            result_df["Momentum"] = DataValidator.safe_divide(
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

            result_df["Support_Distance"] = DataValidator.safe_divide(
                result_df["Close"] - recent_low, result_df["Close"]
            )
            result_df["Resistance_Distance"] = DataValidator.safe_divide(
                recent_high - result_df["Close"], result_df["Close"]
            )

            # ピボットポイント
            prev_high = result_df["High"].shift(1)
            prev_low = result_df["Low"].shift(1)
            prev_close = result_df["Close"].shift(1)

            pivot = (prev_high + prev_low + prev_close) / 3
            result_df["Pivot_Distance"] = DataValidator.safe_divide(
                result_df["Close"] - pivot, pivot
            )

            # フィボナッチレベル
            swing_high = result_df["High"].rolling(window=period).max()
            swing_low = result_df["Low"].rolling(window=period).min()

            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            for level in fib_levels:
                fib_price = swing_low + (swing_high - swing_low) * level
                result_df[f"Fib_{int(level*1000)}_Distance"] = (
                    DataValidator.safe_divide(
                        abs(result_df["Close"] - fib_price), result_df["Close"]
                    )
                )

            # ギャップ分析
            gap = result_df["Open"] - result_df["Close"].shift(1)
            gap_pct = DataValidator.safe_divide(gap, result_df["Close"].shift(1))

            result_df["Gap_Up"] = (pd.Series(gap_pct) > 0.01).astype(int)
            result_df["Gap_Down"] = (pd.Series(gap_pct) < -0.01).astype(int)
            result_df["Gap_Size"] = abs(gap_pct)

            logger.debug("パターン認識特徴量計算完了")
            return result_df

        except Exception as e:
            logger.error(f"パターン認識特徴量計算エラー: {e}")
            return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSIを計算（内部使用）"""
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
