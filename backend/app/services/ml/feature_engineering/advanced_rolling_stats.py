"""
Advanced Rolling Statistics Features

価格とボリュームの高次統計量を計算し、市場の異常状態を検出します。
Kaggle上位入賞で頻繁に使用される手法。
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional


class AdvancedRollingStatsCalculator:
    """
    ローリングウィンドウでの高次統計量計算クラス

    正規分布からの逸脱（歪度・尖度）やテールリスクを捉えます。
    """

    def __init__(self, windows: Optional[List[int]] = None):
        """
        Args:
            windows: 統計計算に使用するウィンドウサイズのリスト
        """
        self.windows = windows or [5, 10, 20, 50]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced Rolling Statistics特徴量を計算

        Args:
            df: OHLCV DataFrame

        Returns:
            統計特徴量を含むDataFrame
        """
        result = pd.DataFrame(index=df.index)

        # リターン計算
        returns = df["close"].pct_change()
        log_returns = np.log(df["close"] / df["close"].shift(1))

        # === 高度なボラティリティ推定量（学術的に実証済み）===
        # Parkinson Volatility (High-Low Range based)
        # 終値ベースより5倍効率的
        const_parkinson = 1.0 / (4.0 * np.log(2.0))
        log_hl = np.log(df["high"] / df["low"])
        parkinson_vol = np.sqrt(const_parkinson * log_hl**2)

        # Garman-Klass Volatility (OHLC based)
        # 終値ベースより8倍効率的
        log_co = np.log(df["close"] / df["open"])
        garman_klass_vol = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)

        for window in self.windows:
            # === 価格統計 ===
            # 歪度（Skewness）- 分布の非対称性
            result[f"Returns_Skewness_{window}"] = returns.rolling(window).skew()

            # 尖度（Kurtosis）- テールの厚さ
            result[f"Returns_Kurtosis_{window}"] = returns.rolling(window).kurt()

            # 対数リターンの歪度
            result[f"LogReturns_Skewness_{window}"] = log_returns.rolling(window).skew()

            # === ボリューム統計 ===
            # ボリュームの歪度
            result[f"Volume_Skewness_{window}"] = df["volume"].rolling(window).skew()

            # ボリュームの尖度
            result[f"Volume_Kurtosis_{window}"] = df["volume"].rolling(window).kurt()

            # === ボラティリティ統計 ===
            # High-Low比率の平均（ボラティリティ代理）
            hl_ratio = (df["high"] - df["low"]) / df["close"]
            result[f"HL_Ratio_Mean_{window}"] = hl_ratio.rolling(window).mean()
            result[f"HL_Ratio_Std_{window}"] = hl_ratio.rolling(window).std()

            # Parkinson Volatility (Rolling)
            result[f"Parkinson_Vol_{window}"] = parkinson_vol.rolling(window).mean()

            # Garman-Klass Volatility (Rolling)
            result[f"Garman_Klass_Vol_{window}"] = garman_klass_vol.rolling(
                window
            ).mean()

            # === 価格位置統計 ===
            # Close位置（High-Low範囲内での位置）
            close_position = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
            result[f"Close_Position_Mean_{window}"] = close_position.rolling(
                window
            ).mean()
            result[f"Close_Position_Std_{window}"] = close_position.rolling(
                window
            ).std()

            # === 高度な統計 ===
            # リターンの絶対値の平均（実現ボラティリティ代理）
            result[f"Abs_Returns_Mean_{window}"] = returns.abs().rolling(window).mean()

            # 正・負リターンの非対称性
            positive_returns = returns.clip(lower=0).rolling(window).mean()
            negative_returns = returns.clip(upper=0).abs().rolling(window).mean()
            result[f"Return_Asymmetry_{window}"] = positive_returns - negative_returns

            # === テールリスク指標 ===
            # 極端なリターンの頻度（±2σ超え）
            returns_std = returns.rolling(window).std()
            extreme_returns = (returns.abs() > 2 * returns_std).astype(float)
            result[f"Extreme_Returns_Freq_{window}"] = extreme_returns.rolling(
                window
            ).mean()

        # === クロス統計（価格とボリュームの関係）===
        for window in [10, 20]:
            # 価格-ボリューム相関
            result[f"Price_Volume_Corr_{window}"] = returns.rolling(window).corr(
                df["volume"].pct_change()
            )

            # ボリューム加重リターン歪度
            result[f"Volume_Weighted_Returns_Skew_{window}"] = (
                self._volume_weighted_skew(returns, df["volume"], window)
            )

        # 欠損値とinf値を処理
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result

    def _volume_weighted_skew(
        self, returns: pd.Series, volume: pd.Series, window: int
    ) -> pd.Series:
        """
        ボリューム加重歪度を計算

        大きな出来高を伴うリターンに重みを付けた歪度
        """
        result = pd.Series(index=returns.index, dtype=float)

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i - window : i].values
            window_volume = volume.iloc[i - window : i].values

            if window_volume.sum() > 0:
                # ボリューム正規化
                weights = window_volume / window_volume.sum()

                # 加重平均
                weighted_mean = np.average(window_returns, weights=weights)

                # 加重標準偏差
                weighted_var = np.average(
                    (window_returns - weighted_mean) ** 2, weights=weights
                )
                weighted_std = np.sqrt(weighted_var)

                if weighted_std > 0:
                    # 加重歪度
                    weighted_skew = np.average(
                        ((window_returns - weighted_mean) / weighted_std) ** 3,
                        weights=weights,
                    )
                    result.iloc[i] = weighted_skew

        return result.fillna(0)
