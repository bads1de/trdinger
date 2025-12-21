"""
OI/FR Advanced Interaction Features

Open InterestとFunding Rateの高度な相互作用パターンを捉える特徴量。
DRW Kaggle Competition と学術論文で有効性が実証された手法。
"""

from typing import Optional

import numpy as np
import pandas as pd

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

        # === 1. OI-Price Regime（市場レジーム識別 - 4象限）===
        # 0: Bull, 1: Short Cover, 2: Bear, 3: Long Liq
        result["OI_Price_Regime"] = AdvancedFeatures.regime_quadrant(
            close=df["close"], open_interest=oi_series
        ).fillna(-1)

        # === 1-B. OI-Price Confirmation (ダイバージェンス検知) ===
        result["OI_Price_Confirmation"] = AdvancedFeatures.oi_price_confirmation(
            close=df["close"], open_interest=oi_series
        ).fillna(0.0)

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

        # === 8. Funding Rate Regime ===
        fr_abs = np.abs(fr_value)
        # rolling.quantile を使用してベクトル化
        fr_q75 = fr_abs.rolling(50).quantile(0.75)
        result["FR_Extreme_Regime"] = (fr_abs > fr_q75).astype(float).fillna(0)

        # === 9. OI-Volume Interaction（OI-出来高相互作用）===
        volume_change = df["volume"].pct_change()
        result["OI_Volume_Interaction"] = oi_change * volume_change
        
        # OI / Volume Ratio (Liquidity Efficiency)
        result["OI_Volume_Ratio"] = AdvancedFeatures.liquidity_efficiency(
            open_interest=oi_series, volume=df["volume"]
        ).fillna(0.0)

        # === New 1. Void Oscillator (流動性真空検知器) ===
        result["Void_Oscillator"] = AdvancedFeatures.void_oscillator(
            close=df["close"], volume=df["volume"], window=20
        ).fillna(0.0)

        # === New 2. Crypto Leverage Index (CLI) ===
        # L/S Ratioデータがこのクラスには渡されていないため、その項は0（中立）として計算
        # 将来的にMarketDataFeatureCalculatorなどで統合することを推奨
        dummy_ls_div = pd.Series(0, index=df.index)
        
        result["Crypto_Leverage_Index"] = AdvancedFeatures.crypto_leverage_index(
            open_interest=oi_series,
            funding_rate=fr_value,
            ls_ratio_divergence=dummy_ls_div, # L/S情報なし
            window=50 # 長めの期間で過熱感を見る
        ).fillna(0.0)

        # === New 3. Triplet Imbalance (Upper/Lower Shadow Balance) ===
        # (High - Close) / (Close - Low) の対数変換
        result["Triplet_Imbalance"] = AdvancedFeatures.triplet_imbalance(
            high=df["high"], low=df["low"], close=df["close"]
        ).fillna(0.0)

        # === New 4. Fakeout Detection (Volume Divergence) ===
        # 高値更新時の出来高減衰シグナル
        result["Fakeout_Volume_Divergence"] = AdvancedFeatures.volume_divergence_fakeout(
            close=df["close"], volume=df["volume"], window=20
        ).fillna(0.0)

        # === 10. Cumulative OI-Price Divergence（累積OI-価格乖離）===
        oi_price_alignment = (np.sign(oi_change) == np.sign(price_change)).astype(float)
        result["Cumulative_OI_Price_Divergence"] = (
            (1 - oi_price_alignment).rolling(20).sum()
        )

        # === 11. Liquidation Cascade Score ===
        result["Liquidation_Cascade_Score"] = (
            AdvancedFeatures.liquidation_cascade_score(
                close=df["close"], open_interest=oi_series, volume=df["volume"]
            ).fillna(0.0)
        )

        # === 12. Squeeze Probability ===
        result["Squeeze_Probability"] = AdvancedFeatures.squeeze_probability(
            close=df["close"],
            funding_rate=fr_value,
            open_interest=oi_series,
            low=df["low"],
        ).fillna(0.0)

        # === 13. Trend Quality ===
        result["Trend_Quality_20"] = AdvancedFeatures.trend_quality(
            close=df["close"], open_interest=oi_series, window=20
        ).fillna(0.0)

        # === 14. OI Weighted Funding Rate ===
        result["OI_Weighted_FR"] = AdvancedFeatures.oi_weighted_funding_rate(
            funding_rate=fr_value, open_interest=oi_series
        ).fillna(0.0)

        # === 15. Liquidity Efficiency ===
        result["Liquidity_Efficiency"] = AdvancedFeatures.liquidity_efficiency(
            open_interest=oi_series, volume=df["volume"]
        ).fillna(0.0)

        # === 16. Leverage Ratio (Estimated) ===
        # Market Cap = Price * Circulating Supply
        # Leverage Ratio = Total OI (USD) / Market Cap (USD)
        #                = Total OI (USD) / (Price * Circulating Supply)
        #
        # Note: Since we don't have historical circulating supply, we use a constant approximation.
        # For BTC, approx 19.7M (as of late 2024).
        # If OI data is in USD (Contract Value), this formula holds.
        # If OI is in Coins, then Leverage Ratio = OI (Coins) / Circulating Supply.
        # Most exchanges provide OI in USD or contracts converted to USD.
        #
        # Assuming OI is USD-denominated (standard in this system):
        estimated_supply = 19_700_000  # BTC approx supply
        estimated_market_cap = df["close"] * estimated_supply

        result["Leverage_Ratio"] = AdvancedFeatures.leverage_ratio(
            open_interest=oi_series, market_cap=estimated_market_cap
        ).fillna(0.0)

        # 欠損値処理
        result = result.fillna(0)

        # inf値を0に置換
        result = result.replace([np.inf, -np.inf], 0)

        return result



