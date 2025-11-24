"""
列挙型定義モジュール

ラベル生成に関連する列挙型を定義します。
"""

from enum import Enum


class ThresholdMethod(Enum):
    """閾値計算方法"""

    FIXED = "fixed"  # 固定閾値
    QUANTILE = "quantile"  # 分位数ベース（KBinsDiscretizerのquantile戦略）
    PERCENTILE = "quantile"
    STD_DEVIATION = "std_deviation"  # 標準偏差ベース
    ADAPTIVE = "adaptive"  # 適応的閾値（GridSearchCVを使用）
    DYNAMIC_VOLATILITY = "dynamic_volatility"  # 動的ボラティリティベース
    KBINS_DISCRETIZER = "kbins_discretizer"  # KBinsDiscretizerベース（推奨）
    TRIPLE_BARRIER = "triple_barrier" # Triple Barrier Method (利確/損切り/時間切れ)