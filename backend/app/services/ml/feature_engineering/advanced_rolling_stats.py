"""
Advanced Rolling Statistics Features

価格とボリュームの高次統計量を計算し、市場の異常状態を検出します。
Kaggle上位入賞で頻繁に使用される手法。
"""

from typing import List, Optional

import numpy as np
import pandas as pd


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
        rets = df["close"].pct_change()
        log_rets = np.log(df["close"] / df["close"].shift(1))
        vol = df["volume"]

        # 高度なボラティリティ推定量
        log_hl = np.log(df["high"] / df["low"])
        park_vol = np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl**2)
        log_co = np.log(df["close"] / df["open"])
        gk_vol = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)

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
            res[f"HL_Ratio_Mean_{w}"], res[f"HL_Ratio_Std_{w}"] = hl_r.rolling(w).mean(), hl_r.rolling(w).std()
            res[f"Parkinson_Vol_{w}"] = park_vol.rolling(w).mean()
            res[f"Garman_Klass_Vol_{w}"] = gk_vol.rolling(w).mean()
            res[f"Yang_Zhang_Vol_{w}"] = self._yang_zhang_volatility(df, w)

            # 価格位置 & テールリスク
            c_pos = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
            res[f"Close_Position_Mean_{w}"], res[f"Close_Position_Std_{w}"] = c_pos.rolling(w).mean(), c_pos.rolling(w).std()
            res[f"Abs_Returns_Mean_{w}"] = rets.abs().rolling(w).mean()
            res[f"Return_Asymmetry_{w}"] = rets.clip(lower=0).rolling(w).mean() - rets.clip(upper=0).abs().rolling(w).mean()
            
            r_std = rets.rolling(w).std()
            res[f"Extreme_Returns_Freq_{w}"] = (rets.abs() > 2 * r_std).astype(float).rolling(w).mean()

        for w in [10, 20]:
            res[f"Price_Volume_Corr_{w}"] = rets.rolling(w).corr(vol.pct_change())
            res[f"Volume_Weighted_Returns_Skew_{w}"] = self._volume_weighted_skew(rets, vol, w)

        return res.fillna(0)

    def _yang_zhang_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Yang-Zhang Volatility Estimator"""
        # 対数価格
        l_o, l_h, l_l, l_c = np.log(df["open"]), np.log(df["high"]), np.log(df["low"]), np.log(df["close"])

        # 1. Overnight Jump & 2. Open-to-Close
        sigma_oj_sq = (l_o - l_c.shift(1)).rolling(window).var()
        sigma_oc_sq = (l_c - l_o).rolling(window).var()

        # 3. Rogers-Satchell
        rs_term = (l_h - l_c) * (l_h - l_o) + (l_l - l_c) * (l_l - l_o)
        sigma_rs_sq = rs_term.rolling(window).mean()

        # Weight k
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        return np.sqrt(sigma_oj_sq + k * sigma_oc_sq + (1 - k) * sigma_rs_sq).fillna(0)

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



