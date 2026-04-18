"""
遺伝子生成用定数

create_random_* 関数で使用する確率、重み、デフォルト値などの定数を定義します。
"""

from typing import Dict

from ..config.constants import EntryType, ExitType

# ==================== EntryGene ====================

# エントリータイプの出現確率（初期段階では成行注文を多めに設定）
ENTRY_TYPE_WEIGHTS: Dict[EntryType, float] = {
    EntryType.MARKET: 0.6,
    EntryType.LIMIT: 0.2,
    EntryType.STOP: 0.15,
    EntryType.STOP_LIMIT: 0.05,
}

# ==================== ExitGene ====================

# イグジットタイプの出現確率（全決済を多めに設定）
EXIT_TYPE_WEIGHTS: Dict[ExitType, float] = {
    ExitType.FULL: 0.5,
    ExitType.PARTIAL: 0.3,
    ExitType.TRAILING: 0.2,
}

# フラグ設定の確率
EXIT_PARTIAL_ENABLED_PROBABILITY = 0.3
EXIT_TRAILING_ACTIVATION_PROBABILITY = 0.2

# ==================== TPSLGene ====================

# method_weights の生成範囲
TPSL_METHOD_WEIGHT_RANGES = {
    "fixed": (0.1, 0.4),
    "risk_reward": (0.2, 0.5),
    "volatility": (0.1, 0.4),
    "statistical": (0.1, 0.3),
}

# ==================== IndicatorGene ====================

# トレンド系指標の優先リスト
PREFERRED_TREND_INDICATORS = [
    "SMA",
    "EMA",
    "ADX",
    "SUPERTREND",
    "VORTEX",
    "AROON",
    "CHOP",
    "HMA",
    "KAMA",
    "ZLEMA",
]

# トレンド系指標を選択する確率
TREND_INDICATOR_SELECTION_PROBABILITY = 0.7

# MAクロス補完の確率
MA_CROSS_ENHANCEMENT_PROBABILITY = 0.25

# MA期間の候補
MA_PERIOD_CANDIDATES = [10, 14, 20, 30, 50]

# 指標生成の最大試行回数倍率
INDICATOR_GENERATION_MAX_ATTEMPTS_MULTIPLIER = 5

# ==================== 共通 ====================

# 優先度（priority）の生成範囲
PRIORITY_GENERATION_RANGE = (0.5, 1.5)
