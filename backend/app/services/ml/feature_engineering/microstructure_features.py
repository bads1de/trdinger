import numpy as np
import pandas as pd
from typing import Optional


class MicrostructureFeatureCalculator:
    """
    市場の微細構造（Microstructure）に関する特徴量を計算するクラス。
    LSレシオの弾力性、FRの歪み、流動性枯渇に特化。
    """

    def calculate_features(
        self, 
        ohlcv: pd.DataFrame, 
        fr_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,
        ls_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(index=ohlcv.index)

        # 1. 流動性・マーケットインパクト
        df["Amihud_Illiquidity_20h"] = self.calculate_amihud_illiquidity(ohlcv, window=20)
        df["Kyles_Lambda_20h"] = self.calculate_kyles_lambda(ohlcv, window=20)

        # 2. 統計的異常度
        returns = ohlcv["close"].pct_change()
        df["Returns_Kurtosis_50"] = returns.rolling(window=50).kurt()
        df["Returns_Skewness_50"] = returns.rolling(window=50).skew()

        # 3. Funding Rate (NaN対策を強化)
        if fr_df is not None and not fr_df.empty:
            fr_aligned = fr_df.reindex(ohlcv.index).ffill().bfill()
            col = "funding_rate" if "funding_rate" in fr_aligned.columns else fr_aligned.columns[0]
            fr = fr_aligned[col]
            mean_fr = fr.rolling(168).mean()
            std_fr = fr.rolling(168).std()
            # stdが0の場合でもNaNにならないように保護
            df["FR_Extremity_Zscore"] = (fr - mean_fr) / (std_fr + 0.00001)
            df["FR_Change_4h"] = fr.diff(4)

        # 4. Long/Short Ratio (NaN対策を強化)
        if ls_df is not None and not ls_df.empty:
            ls_aligned = ls_df.reindex(ohlcv.index).ffill().bfill()
            ls_col = "long_short_ratio" if "long_short_ratio" in ls_aligned.columns else ls_aligned.columns[0]
            ls = ls_aligned[ls_col]

            # A. Sentiment Elasticity
            price_change = ohlcv["close"].pct_change(3).abs()
            ls_change = ls.pct_change(3).abs()
            df["LS_Sentiment_Elasticity"] = ls_change / (price_change + 1e-6)

            # B. LS Acceleration
            df["LS_Acceleration"] = ls.diff().diff()

            # C. LS Price Incongruence
            df["LS_Price_Incongruence"] = ohlcv["close"].pct_change(4) * ls.diff(4) * -1
            
            # D. LS FR Stress Index
            if "FR_Extremity_Zscore" in df.columns:
                ls_mean = ls.rolling(168).mean()
                ls_std = ls.rolling(168).std()
                ls_z = (ls - ls_mean) / (ls_std + 0.00001)
                df["LS_FR_Stress_Index"] = df["FR_Extremity_Zscore"] * ls_z

        # 5. 全てのNaNを0で埋めて、特徴量が消えるのを防ぐ
        df = df.ffill().fillna(0)

        return df

    def calculate_amihud_illiquidity(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = df["close"].pct_change().abs()
        dollar_volume = df["volume"] * df["close"]
        return (returns / (dollar_volume + 1e-9)).rolling(window=window).mean().fillna(0)

    def calculate_kyles_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        returns = df["close"].pct_change()
        sign = np.sign(df["close"] - df["open"])
        signed_volume = df["volume"] * sign
        cov = returns.rolling(window=window).cov(signed_volume)
        var = signed_volume.rolling(window=window).var()
        return (cov / (var + 1e-9)).fillna(0)