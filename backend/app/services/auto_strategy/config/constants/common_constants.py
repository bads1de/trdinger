"""
Auto Strategy 共通定数

システム全体で使用される共通定数を定義します。
"""

from app.config.constants import DEFAULT_MARKET_SYMBOL

# === データソース定数 ===
DATA_SOURCES = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "OpenInterest",
    "FundingRate",
]

# === 取引ペア定数 ===
DEFAULT_SYMBOL = DEFAULT_MARKET_SYMBOL

# 複合指標
COMPOSITE_INDICATORS = [
    "ICHIMOKU",
    "UI",
]

# 移動平均系指標の定数
MOVING_AVERAGE_INDICATORS = {
    "SMA",
    "EMA",
    "WMA",
    "KAMA",
    "TEMA",
    "DEMA",
    "T3",
}

# 優先的な移動平均指標（MA系を2本以上生成する場合の候補）
PREFERRED_MA_INDICATORS = {"SMA", "EMA"}

# === 自動戦略デフォルト値 (フォールバック用) ===
AUTO_STRATEGY_DEFAULTS = {
    # ポジションサイジング
    "fallback_atr_multiplier": 0.04,
    "assumed_win_rate": 0.55,
    "assumed_avg_win": 0.02,
    "assumed_avg_loss": 0.015,
}

# === 戦略一覧APIとGAフォールバックの既定値 ===
DEFAULT_STRATEGIES_LIMIT = 20
MAX_STRATEGIES_LIMIT = 100
GA_FALLBACK_START_DATE = "2024-01-01"
GA_FALLBACK_END_DATE = "2024-04-09"
