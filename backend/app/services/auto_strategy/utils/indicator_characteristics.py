"""
指標特性データベース
定数定義から分離された動的指標特性管理モジュール
"""

# YAMLベースの特性動的生成処理


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
    },
    "MACDEXT": {
        "type": "momentum",
        "range": None,  # 価格依存
        "zero_cross": True,
        "signal_line": True,
    },
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
    "BBANDS": {
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
    "ML_UP_PROB": {
        "type": "ml_prediction",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0.6, 1.0)],
        "short_zones": [(0, 0.4)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_DOWN_PROB": {
        "type": "ml_prediction",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.4)],
        "short_zones": [(0.6, 1.0)],
        "neutral_zone": (0.4, 0.6),
        "high_confidence_threshold": 0.7,
    },
    "ML_RANGE_PROB": {
        "type": "ml_prediction",
        "range": (0, 1),  # 確率値
        "ml_prediction": True,
        "long_zones": [(0, 0.3)],
        "short_zones": [(0, 0.3)],
        "neutral_zone": (0.7, 1.0),
        "high_confidence_threshold": 0.8,
    },
    "FWMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "weighted_average": True,  # フィボナッチ重み付き移動平均
    },
    "SWMA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "symmetric_weighting": True,  # 対称重み付き移動平均
    },
    "VIDYA": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "adaptive_smoothing": True,  # 適応的スムージング
        "volatility_adaptive": True,  # ボラティリティ適応型
    },
    "LINREG": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
        "regression_based": True,  # 回帰ベース
        "linear_regression": True,  # 線形回帰
    },
    "LINREG_SLOPE": {
        "type": "trend",
        "range": None,  # 価格トレンドに応じた値
        "regression_based": True,
        "slope_tracking": True,  # 勾配追跡
        "momentum_like": True,  # モメンタム様
    },
    "LINREG_INTERCEPT": {
        "type": "trend",
        "price_comparison": True,
        "regression_based": True,
        "trend_following": True,
    },
    "LINREG_ANGLE": {
        "type": "trend",
        "range": (-90, 90),  # 角度範囲
        "angle_tracking": True,  # 角度追跡
        "trend_strength": True,  # トレンド強度を示唆
        "oscillator_like": True,  # オシレーター様
    },
    "PPO": {
        "type": "trend",
        "range": (-100, 100),
        "long_zones": [(-60, -20), (-10, -1)],
        "short_zones": [(1, 10), (20, 60)],
        "neutral_zone": (-1, 1),
        "zero_cross": True,
        "signal_line": True,
    },
    "STC": {
        "type": "trend",
        "range": (0, 100),
        "long_zones": [(25, 50), (50, 75)],
        "short_zones": [(50, 75), (75, 100)],
        "neutral_zone": (40, 60),
    },
    "MAVP": {
        "type": "trend",
        "price_comparison": True,
        "trend_following": True,
    },
    "SAREXT": {
        "type": "trend",
        "trend_following": True,
        "price_comparison": True,
        "reversal_indicator": True,
    },
}


# YAML設定に基づいて特性を生成してマージ
INDICATOR_CHARACTERISTICS = _get_merged_characteristics(_INDICATOR_CHARACTERISTICS_BASE)