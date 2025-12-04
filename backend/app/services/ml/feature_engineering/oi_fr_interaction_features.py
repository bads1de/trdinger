"""
OI/FR Advanced Interaction Features

Open InterestとFunding Rateの高度な相互作用パターンを捉える特徴量。
DRW Kaggle Competition と学術論文で有効性が実証された手法。
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...indicators.technical_indicators.advanced_features import AdvancedFeatures


class OIFRInteractionFeatureCalculator:
    """
    OI（建玉）とFR（資金調達率）の非線形相互作用を捉える特徴量計算クラス

    個別指標では捉えられない市場参加者の行動パターンを検出します。
    """

    def calculate_features(
        self,
        df: pd.DataFrame,
        oi_data: Optional[pd.DataFrame] = None,
        fr_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        OI/FR相互作用特徴量を計算

        Args:
            df: OHLCV DataFrame
            oi_data: Open Interest data
            fr_data: Funding Rate data

        Returns:
            相互作用特徴量を含むDataFrame
        """
        result = pd.DataFrame(index=df.index)

        if oi_data is None or fr_data is None or len(oi_data) == 0 or len(fr_data) == 0:
            # データがない場合は空のDataFrameを返す
            return result

        # OI/FRデータが既にOHLCVと同じインデックスの場合はそのまま使用
        if len(oi_data) == len(df) and (oi_data.index == df.index).all():
            oi_aligned = oi_data
        else:
            # 異なる場合は空を返す（事前にデータ整合性が取れているはず）
            return result

        if len(fr_data) == len(df) and (fr_data.index == df.index).all():
            fr_aligned = fr_data
        else:
            return result

        # 価格変化率
        price_change = df["close"].pct_change()

        # OI変化率
        oi_change = (
            oi_aligned.iloc[:, 0].pct_change()
            if len(oi_aligned.columns) > 0
            else pd.Series(0, index=df.index)
        )

        # FR値（最初のカラムを使用）
        fr_value = (
            fr_aligned.iloc[:, 0]
            if len(fr_aligned.columns) > 0
            else pd.Series(0, index=df.index)
        )
        
        # OI Series (for advanced features)
        oi_series = (
            oi_aligned.iloc[:, 0]
            if len(oi_aligned.columns) > 0
            else pd.Series(0, index=df.index)
        )

        # === 1. OI-Price Regime（市場レジーム識別）===
        result["OI_Price_Regime"] = (
            np.sign(price_change) * np.sign(oi_change) * np.abs(oi_change)
        )

        # === 2. FR Acceleration（FR変化の加速度）===
        fr_change = fr_value.diff()
        result["FR_Acceleration"] = fr_change.diff()

        # === 3. Liquidation Pressure（清算圧力）===
        result["Liquidation_Pressure"] = (
            np.abs(oi_change) * np.abs(fr_value) * np.sign(fr_value)
        )

        # === 4. Smart Money Flow（スマートマネーの流れ）===
        oi_increasing = oi_change > 0
        result["Smart_Money_Flow"] = np.where(
            oi_increasing,
            price_change * np.abs(oi_change),
            -price_change * np.abs(oi_change),
        )

        # === 5. FR-OI Divergence（FR-OI乖離）===
        result["FR_OI_Divergence"] = (np.sign(fr_change) != np.sign(oi_change)).astype(
            float
        )

        # === 6. Market Stress Indicator（市場ストレス指標・改良版）===
        result["Market_Stress_V2"] = (
            np.abs(fr_value) * np.abs(oi_change) * np.abs(price_change)
        )

        # === 7. OI Momentum with FR Confirmation（FR確認付きOIモメンタム）===
        oi_momentum = oi_change.rolling(5).mean()
        fr_direction = np.sign(fr_value)
        result["OI_Momentum_FR_Confirmed"] = oi_momentum * fr_direction

        # === 8. Funding Rate Regime（FR状態分類）===
        fr_abs = np.abs(fr_value)
        fr_percentile = fr_abs.rolling(50).apply(
            lambda x: (
                (x.iloc[-1] > np.percentile(x, 75)).astype(float) if len(x) > 0 else 0
            )
        )
        result["FR_Extreme_Regime"] = fr_percentile

        # === 9. OI-Volume Interaction（OI-出来高相互作用）===
        volume_change = df["volume"].pct_change()
        result["OI_Volume_Interaction"] = oi_change * volume_change

        # === 10. Cumulative OI-Price Divergence（累積OI-価格乖離）===
        oi_price_alignment = (np.sign(oi_change) == np.sign(price_change)).astype(float)
        result["Cumulative_OI_Price_Divergence"] = (
            (1 - oi_price_alignment).rolling(20).sum()
        )
        
        # === 11. Liquidation Cascade Score ===
        result["Liquidation_Cascade_Score"] = AdvancedFeatures.liquidation_cascade_score(
           close=df["close"],
           open_interest=oi_series,
           volume=df["volume"]
        ).fillna(0.0)

        # === 12. Squeeze Probability ===
        result["Squeeze_Probability"] = AdvancedFeatures.squeeze_probability(
           close=df["close"],
           funding_rate=fr_value,
           open_interest=oi_series,
           low=df["low"]
        ).fillna(0.0)

        # === 13. Trend Quality ===
        result["Trend_Quality_20"] = AdvancedFeatures.trend_quality(
           close=df["close"],
           open_interest=oi_series,
           window=20
        ).fillna(0.0)

        # === 14. OI Weighted Funding Rate ===
        result["OI_Weighted_FR"] = AdvancedFeatures.oi_weighted_funding_rate(
           funding_rate=fr_value,
           open_interest=oi_series
        ).fillna(0.0)
   
        # === 15. Liquidity Efficiency ===
        result["Liquidity_Efficiency"] = AdvancedFeatures.liquidity_efficiency(
           open_interest=oi_series,
           volume=df["volume"]
        ).fillna(0.0)

        # 欠損値処理
        result = result.fillna(0)

        # inf値を0に置換
        result = result.replace([np.inf, -np.inf], 0)

        return result
