"""
指標特性データベース
定数定義から分離された動的指標特性管理モジュール
"""


def _get_merged_characteristics(original):
    # Late import to avoid circular imports
    from app.services.auto_strategy.utils.yaml_utils import YamlIndicatorUtils

    return YamlIndicatorUtils.initialize_yaml_based_characteristics(original)


# === 技術指標特性データベース ===
# condition_generator.pyから移行した指標特性定義
_INDICATOR_CHARACTERISTICS_BASE = {
    "RSI": {
        "type": "momentum",
        "range": (0, 100),
        "long_zones": [(0, 30), (40, 60)],
        "short_zones": [(40, 60), (70, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 30,
        "overbought_threshold": 70,
    },
    "STOCH": {
        "type": "momentum",
        "range": (0, 100),
        "long_zones": [(0, 20), (40, 60)],
        "short_zones": [(40, 60), (80, 100)],
        "neutral_zone": (40, 60),
        "oversold_threshold": 20,
        "overbought_threshold": 80,
    },
    "CCI": {
        "type": "momentum",
        "range": (-200, 200),
        "long_zones": [(-200, -100), (-50, 50)],
        "short_zones": [(-50, 50), (100, 200)],
        "neutral_zone": (-50, 50),
        "oversold_threshold": -100,
        "overbought_threshold": 100,
    },
    "MACD": {
        "type": "momentum",
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
        "SMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
        },
        "EMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
        },
        "ADX": {
            "type": "trend",
            "range": (0, 100),
            "trend_strength": True,
            "no_direction": True,  # 方向性を示さない
            "strong_trend_threshold": 25,
        },
        "BB": {
            "type": "volatility",
            "components": ["upper", "middle", "lower"],
            "mean_reversion": True,
            "breakout_strategy": True,
        },
        "ATR": {
            "type": "volatility",
            "range": None,
            "volatility_measure": True,
        },
        "KELTNER": {
            "type": "volatility",
            "components": ["upper", "middle", "lower"],
            "price_comparison": True,
            "volatility_based": True,
        },
        "SUPERTREND": {
            "type": "trend",
            "range": None,
            "trend_following": True,
            "price_comparison": True,
            "reversal_indicator": True,
        },
        "DONCHIAN": {
            "type": "volatility",
            "components": ["upper", "middle", "lower"],
            "price_comparison": True,
            "breakout_strategy": True,
        },
        "ACCBANDS": {
            "type": "volatility",
            "components": ["upper", "middle", "lower"],
            "price_comparison": True,
            "acceleration_based": True,
        },
        "UI": {
            "type": "volatility",
            "range": (0, 100),  # パーセント
            "volatility_measure": True,
            "relative_volatility": True,
            "zero_threshold": 10,  # 安定を示す閾値
            "high_risk_level": 50,
        },
        "OBV": {
            "type": "volume",
            "range": None,
            "volume_accumulation": True,
            "price_confirmation": True,
        },
        "AD": {
            "type": "volume",
            "range": None,
            "volume_pressure": True,
            "price_confirmation": True,
        },
        "ADOSC": {
            "type": "volume",
            "range": None,
            "volume_oscillation": True,
            "zero_cross": True,
        },
        "EFI": {
            "type": "volume",
            "range": None,
            "volume_force": True,
            "price_volume_combination": True,
        },
        "VWAP": {
            "type": "trend",
            "price_comparison": True,
            "volume_weighted": True,
            "intraday_trend": True,
        },
        "WMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
            "linear_weighted": True,
        },
        "DEMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
            "double_smoothing": True,
        },
        "TEMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
            "triple_smoothing": True,
        },
        "T3": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
            "multiple_smoothing": True,
        },
        "KAMA": {
            "type": "trend",
            "price_comparison": True,
            "trend_following": True,
            "adaptive_smoothing": True,
            "efficiency_ratio": True,
        },
        "SAR": {
            "type": "trend",
            "trend_following": True,
            "price_comparison": True,
            "reversal_indicator": True,
            "price_level": True,
        },
        "WILLR": {
            "type": "momentum",
            "range": (-100, 0),
            "long_zones": [(-80, 0)],
            "short_zones": [(-100, -20)],
            "neutral_zone": (-80, -20),
            "oversold_threshold": -80,
            "overbought_threshold": -20,
        },
        "MOM": {
            "type": "momentum",
            "range": None,
            "zero_cross": True,
            "momentum_indicator": True,
        },
        "ROC": {
            "type": "momentum",
            "range": None,
            "zero_cross": True,
            "rate_of_change": True,
        },
        "QQE": {
            "type": "momentum",
            "range": (0, 100),
            "long_zones": [(30, 50), (50, 70)],
            "short_zones": [(50, 70), (70, 100)],
            "neutral_zone": (40, 60),
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
        "SQUEEZE": {
            "type": "momentum",
            "range": (0, 1),  # バイナリ値（収縮時1）
            "squeeze_indicator": True,
            "volatility_squeeze": True,
        },
    },
}


# YAML設定に基づいて特性を生成してマージ
INDICATOR_CHARACTERISTICS = _get_merged_characteristics(_INDICATOR_CHARACTERISTICS_BASE)
