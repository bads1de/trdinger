"""特徴量数カウント"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.ml.feature_engineering.feature_engineering_service import (
    DEFAULT_FEATURE_ALLOWLIST,
    FAKEOUT_DETECTION_ALLOWLIST,
)

print(f"\n=== 特徴量総数 ===")
print(f"DEFAULT_FEATURE_ALLOWLIST: {len(DEFAULT_FEATURE_ALLOWLIST)}個")
print(f"FAKEOUT_DETECTION_ALLOWLIST: {len(FAKEOUT_DETECTION_ALLOWLIST)}個")

print(f"\n=== DEFAULT_FEATURE_ALLOWLIST 特徴量一覧 ===")
for i, feature in enumerate(DEFAULT_FEATURE_ALLOWLIST, 1):
    print(f"{i:3d}. {feature}")

print(f"\n=== カテゴリ別内訳 ===")
categories = {
    "Volume/Flow": ["AD", "OBV", "Volume_MA_20", "ADOSC"],
    "Momentum": ["RSI", "MACD_Histogram"],
    "Trend": ["ADX", "MA_Long", "SMA_Cross_50_200", "Trend_strength_20"],
    "Volatility": [
        "NATR",
        "Parkinson_Vol_20",
        "Close_range_20",
        "Historical_Volatility_20",
        "BB_Width",
        "Volume_CV",
    ],
    "Price Structure": ["price_vs_low_24h", "VWAP_Deviation", "Price_Skewness_20"],
    "Volume Profile": [
        "POC_Distance_50",
        "VAH_Distance_50",
        "VAL_Distance_50",
        "In_Value_Area_50",
        "HVN_Distance",
        "VP_Skewness",
    ],
    "Advanced Rolling Stats": [
        "Returns_Skewness_20",
        "Returns_Kurtosis_20",
        "Volume_Skewness_20",
        "HL_Ratio_Mean_20",
        "Return_Asymmetry_20",
    ],
    "OI/FR Interaction": [
        "OI_Price_Regime",
        "FR_Acceleration",
        "Smart_Money_Flow",
        "Market_Stress_V2",
        "OI_Volume_Interaction",
    ],
    "Multi-Timeframe": [
        "HTF_4h_Trend_Direction",
        "HTF_4h_Trend_Strength",
        "HTF_1d_Trend_Direction",
        "Timeframe_Alignment_Score",
        "Timeframe_Alignment_Direction",
        "Price_Distance_From_4h_SMA50",
    ],
    "Market Data (OI/FR)": [
        "Price_OI_Divergence",
        "OI_Volume_Correlation",
        "OI_Momentum_Ratio",
        "OI_Liquidation_Risk",
        "FR_Extremity_Zscore",
        "FR_Cumulative_Trend",
        "FR_OI_Sentiment",
        "Liquidation_Risk",
        "FR_Volatility",
        "Volume_OI_Ratio",
    ],
    "OI/FR Technicals": [
        "OI_MACD",
        "OI_MACD_Hist",
        "OI_BB_Position",
        "OI_BB_Width",
        "FR_MACD",
    ],
    "Market Structure": ["Amihud_Illiquidity", "Efficiency_Ratio", "Market_Impact"],
}

for category, features in categories.items():
    print(f"  {category}: {len(features)}個")

total = sum(len(f) for f in categories.values())
print(f"\n合計（カテゴリ別集計）: {total}個")
