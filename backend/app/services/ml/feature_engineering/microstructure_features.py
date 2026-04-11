"""
市場微細構造（Microstructure）特徴量モジュール

市場の流動性、マーケットインパクト、Funding Rateの歪み、
Long/Short比率の弾性など、高頻度取引特有の微細構造に起因する
特徴量を計算します。
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base_feature_calculator import sanitize_numeric_dataframe


class MicrostructureFeatureCalculator:
    """市場の微細構造（Microstructure）に関する特徴量を計算するクラス。

    OHLCVデータに加えて、Funding Rate、Open Interest、Long/Short比率
    などの追加データソースを使用して、以下のような特徴量を生成します:

    - Amihud非流動性指標、Kyle's Lambda（流動性測定）
    - リターンの歪度・尖度（統計的異常度）
    - Funding Rateの異常度（Zスコア、変化率）
    - Long/Short比率の弾性、加速度、価格との不一致
    - LS/FRストレスインデックス
    """

    def calculate_features(
        self,
        ohlcv: pd.DataFrame,
        fr_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,
        ls_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """市場微細構造特徴量を計算する。

        OHLCVデータをベースに、オプションでFunding Rate、Open Interest、
        Long/Short比率のデータを結合して特徴量を生成します。

        Args:
            ohlcv: OHLCVデータ。インデックスはDatetimeIndex。
            fr_df: Funding Rateデータ（オプション）。
            oi_df: Open Interestデータ（オプション。現在は未使用）。
            ls_df: Long/Short Ratioデータ（オプション）。

        Returns:
            pd.DataFrame: 計算された微細構造特徴量。
                インデックスはohlcvと同じ。NaNは0で埋められる。
        """
        df = pd.DataFrame(index=ohlcv.index)

        # 1. 流動性・マーケットインパクト
        df["Amihud_Illiquidity_20h"] = self.calculate_amihud_illiquidity(
            ohlcv, window=20
        )
        df["Kyles_Lambda_20h"] = self.calculate_kyles_lambda(ohlcv, window=20)

        # 2. 統計的異常度
        returns = ohlcv["close"].pct_change()
        df["Returns_Kurtosis_50"] = returns.rolling(window=50).kurt()
        df["Returns_Skewness_50"] = returns.rolling(window=50).skew()

        # 3. Funding Rate (NaN対策を強化)
        if fr_df is not None and not fr_df.empty:
            # 未来値で埋める backfill は使わず、開始直後は 0 埋めにする
            fr_aligned = sanitize_numeric_dataframe(
                fr_df.reindex(ohlcv.index), fill_value=0.0, forward_fill=True
            )
            col = (
                "funding_rate"
                if "funding_rate" in fr_aligned.columns
                else fr_aligned.columns[0]
            )
            fr = fr_aligned[col]
            mean_fr = fr.rolling(168).mean()
            std_fr = fr.rolling(168).std()
            # stdが0の場合でもNaNにならないように保護
            df["FR_Extremity_Zscore"] = (fr - mean_fr) / (std_fr + 0.00001)
            df["FR_Change_4h"] = fr.diff(4)

        # 4. Long/Short Ratio (NaN対策を強化)
        if ls_df is not None and not ls_df.empty:
            # 未来値で埋める backfill は使わず、開始直後は 0 埋めにする
            ls_aligned = sanitize_numeric_dataframe(
                ls_df.reindex(ohlcv.index), fill_value=0.0, forward_fill=True
            )
            ls_col = (
                "long_short_ratio"
                if "long_short_ratio" in ls_aligned.columns
                else ls_aligned.columns[0]
            )
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
        df = sanitize_numeric_dataframe(df, fill_value=0.0, forward_fill=True)

        return df

    def calculate_amihud_illiquidity(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.Series:
        """Amihud非流動性指標を計算する。

        絶対リターンをドル出来高で割った値の移動平均を計算し、
        市場の流動性の低さを測定します。値が大きいほど流動性が低いことを示します。

        Args:
            df: OHLCVデータ。"close"と"volume"カラムが必要。
            window: 移動平均の窓期間（デフォルト: 20）。

        Returns:
            pd.Series: Amihud非流動性指標の時系列。
        """
        returns = df["close"].pct_change().abs()
        dollar_volume = df["volume"] * df["close"]
        return (
            (returns / (dollar_volume + 1e-9)).rolling(window=window).mean().fillna(0)
        )

    def calculate_kyles_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Kyle's Lambda（価格インパクト指標）を計算する。

        リターンと符号付き出来高の共分散を、符号付き出来高の分散で割ることで、
        注文フローが価格に与える影響度（マーケットインパクト）を測定します。

        Args:
            df: OHLCVデータ。"close"、"open"、"volume"カラムが必要。
            window: 共分散・分散計算の窓期間（デフォルト: 20）。

        Returns:
            pd.Series: Kyle's Lambdaの時系列。値が大きいほど価格インパクトが大きい。
        """
        returns = df["close"].pct_change()
        sign = np.sign(df["close"] - df["open"])
        signed_volume = df["volume"] * sign
        cov = returns.rolling(window=window).cov(signed_volume)
        var = signed_volume.rolling(window=window).var()
        return (cov / (var + 1e-9)).fillna(0)
