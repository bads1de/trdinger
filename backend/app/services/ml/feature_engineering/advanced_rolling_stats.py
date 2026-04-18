"""
Advanced Rolling Statistics Features

価格とボリュームの高次統計量を計算し、市場の異常状態を検出します。
Kaggle上位入賞で頻繁に使用される手法。
"""

from typing import List, Optional, cast

import numpy as np
import pandas as pd

from ...indicators.technical_indicators.advanced_features import AdvancedFeatures
from .volatility_estimators import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)


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
        """Advanced Rolling Statistics特徴量を計算"""
        res = pd.DataFrame(index=df.index)
        rets = cast(pd.Series, df["close"]).pct_change()
        log_rets = pd.Series(np.log(df["close"] / df["close"].shift(1)), index=df.index)
        vol = cast(pd.Series, df["volume"])

        for w in self.windows:
            # 価格統計
            res[f"Returns_Skewness_{w}"] = rets.rolling(w).skew()
            res[f"Returns_Kurtosis_{w}"] = rets.rolling(w).kurt()
            res[f"LogReturns_Skewness_{w}"] = log_rets.rolling(w).skew()

            # ボリューム統計
            res[f"Volume_Skewness_{w}"] = vol.rolling(w).skew()
            res[f"Volume_Kurtosis_{w}"] = vol.rolling(w).kurt()

            # ボラティリティ統計
            hl_r = (df["high"] - df["low"]) / df["close"]
            res[f"HL_Ratio_Mean_{w}"], res[f"HL_Ratio_Std_{w}"] = (
                hl_r.rolling(w).mean(),
                hl_r.rolling(w).std(),
            )
            res[f"Parkinson_Vol_{w}"] = parkinson_volatility(
                cast(pd.Series, df["high"]), cast(pd.Series, df["low"]), window=w
            )
            res[f"Garman_Klass_Vol_{w}"] = garman_klass_volatility(
                cast(pd.Series, df["open"]),
                cast(pd.Series, df["high"]),
                cast(pd.Series, df["low"]),
                cast(pd.Series, df["close"]),
                window=w,
            )
            res[f"Yang_Zhang_Vol_{w}"] = yang_zhang_volatility(
                cast(pd.Series, df["open"]),
                cast(pd.Series, df["high"]),
                cast(pd.Series, df["low"]),
                cast(pd.Series, df["close"]),
                window=w,
            )

            # 価格位置 & テールリスク
            c_pos = (
                cast(pd.Series, df["close"]) - cast(pd.Series, df["low"])
            ) / (cast(pd.Series, df["high"]) - cast(pd.Series, df["low"]) + 1e-9)
            res[f"Close_Position_Mean_{w}"], res[f"Close_Position_Std_{w}"] = (
                c_pos.rolling(w).mean(),
                c_pos.rolling(w).std(),
            )
            res[f"Abs_Returns_Mean_{w}"] = rets.abs().rolling(w).mean()
            res[f"Return_Asymmetry_{w}"] = (
                rets.clip(lower=0).rolling(w).mean()
                - rets.clip(upper=0).abs().rolling(w).mean()
            )

            r_std = rets.rolling(w).std()
            res[f"Extreme_Returns_Freq_{w}"] = (
                (rets.abs() > 2 * r_std).astype(float).rolling(w).mean()
            )

        for w in [10, 20]:
            res[f"Price_Volume_Corr_{w}"] = rets.rolling(w).corr(vol.pct_change())
            res[f"Volume_Weighted_Returns_Skew_{w}"] = self._volume_weighted_skew(
                cast(pd.Series, rets), cast(pd.Series, vol), w
            )

        # ハースト指数 (長期記憶性) - 計算コストが高いため期間固定
        res["Hurst_Exponent_100"] = AdvancedFeatures.hurst_exponent(
            cast(pd.Series, df["close"]), window=100
        )

        return res.fillna(0)

    def _volume_weighted_skew(
        self, returns: pd.Series, volume: pd.Series, window: int
    ) -> pd.Series:
        """
        ボリューム加重歪度を計算（完全ベクトル化版）
        """
        # 加重平均
        v_sum = volume.rolling(window).sum()
        m = (returns * volume).rolling(window).sum() / v_sum

        # 加重分散
        # sum(w_i * (x_i - m)^2) = sum(w_i * x_i^2) - m^2
        v_var = (returns**2 * volume).rolling(window).sum() / v_sum - m**2
        v_std = np.sqrt(v_var.clip(lower=0))

        # 加重歪度
        # sum(w_i * (x_i - m)^3) = sum(w_i * x_i^3) - 3m*sum(w_i * x_i^2) + 2m^3
        v_skew = (
            (returns**3 * volume).rolling(window).sum() / v_sum
            - 3 * m * (returns**2 * volume).rolling(window).sum() / v_sum
            + 2 * m**3
        ) / (v_std**3 + 1e-9)

        return v_skew.fillna(0)
