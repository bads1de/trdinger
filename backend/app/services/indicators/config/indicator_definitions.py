"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

import logging


from .momentum_indicators_config import setup_momentum_indicators
from .trend_indicators_config import setup_trend_indicators
from .volatility_indicators_config import setup_volatility_indicators
from .volume_indicators_config import setup_volume_indicators
from .indicator_config import indicator_registry

logger = logging.getLogger(__name__)


# 各指標の最小必要データ長定義に使用するヘルパー関数
def get_param_value(params, keys, default):
    """パラメータ名がlengthまたはwindowの場合の値取得をサポート"""
    for key in keys:
        if key in params:
            return params[key]
    return default


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()


def generate_positional_functions() -> set:
    """レジストリからPOSITIONAL_DATA_FUNCTIONSを動的生成

    Returns:
        位置パラメータを使用する関数名のセット
    """
    functions = set()
    all_indicators = indicator_registry.get_all_indicators()

    for name, indicator_config in all_indicators.items():
        # エイリアスではなく本名のみを使用
        if not indicator_config.aliases or name not in indicator_config.aliases:
            functions.add(indicator_config.indicator_name.lower())

    return functions


# モジュール読み込み時に初期化
# python-ta動的処理設定のグローバル設定
PANDAS_TA_CONFIG = {
    "RSI": {
        "function": "rsi",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 14),
    },
    "SMA": {
        "function": "sma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 20),
    },
    "EMA": {
        "function": "ema",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
        "min_length": lambda params: max(2, get_param_value(params, ["length", "window"], 20) // 3),
    },
    "WMA": {
        "function": "wma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 20),
    },
    "DEMA": {
        "function": "dema",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 5},
    },
    "TEMA": {
        "function": "tema",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
        "min_length": lambda params: max(3, get_param_value(params, ["length", "window"], 14) // 2),
    },
    "T3": {
        "function": "t3",
        "params": {"length": ["length", "period"], "a": ["a", "vfactor"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 5, "a": 0.7},
    },
    "KAMA": {
        "function": "kama",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 10},
    },
    "MACD": {
        "function": "macd",
        "params": {"fast": ["fast"], "slow": ["slow"], "signal": ["signal"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["MACD", "Signal", "Histogram"],
        "default_values": {"fast": 12, "slow": 26, "signal": 9},
        "min_length": lambda params: params.get("slow", 26) + params.get("signal", 9) + 5,
    },
    "STOCH": {
        "function": "stoch",
        "params": {
            "k_length": ["k", "fastk"],
            "smooth_k": ["smooth_k", "slowk"],
            "d_length": ["d", "slowd"],
        },
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "multiple",
        "return_cols": ["STOCH_K", "STOCH_D"],
        "default_values": {"k_length": 14, "smooth_k": 1, "d_length": 3},
    },
    "CCI": {
        "function": "cci",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "single",
        "default_values": {"length": 20},
    },
    "WILLR": {
        "function": "willr",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "single",
        "default_values": {"length": 14},
    },
    "ROC": {
        "function": "roc",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 10},
    },
    "MOM": {
        "function": "mom",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 10},
    },
    "ADX": {
        "function": "adx",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "multiple",
        "return_cols": ["ADX", "DMP", "DMN"],
        "default_values": {"length": 14},
    },
    "QQE": {
        "function": "qqe",
        "params": {"length": ["length"], "smooth": ["smooth"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["QQE", "QQE_SIGNAL"],
        "default_values": {"length": 14, "smooth": 5},
    },
    "SAR": {
        "function": "psar",
        "params": {
            "af": ["acceleration", "acc_factor"],
            "max_af": ["maximum_acceleration"],
        },
        "multi_column": True,
        "data_columns": ["High", "Low"],
        "returns": "single",
        "default_values": {"af": 0.02, "max_af": 0.2},
    },
    "ATR": {
        "function": "atr",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "single",
        "default_values": {"length": 14},
    },
    "BBANDS": {
        "function": "bbands",
        "params": {"length": ["length", "period"], "std": ["std", "multiplier"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["BBL", "BBM", "BBU"],
        "default_values": {"length": 20, "std": 2.0},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 20),
    },
    "KELTNER": {
        "function": "kc",
        "params": {"length": ["length"], "multiplier": ["multiplier"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "multiple",
        "return_cols": ["KC_LB", "KC_MID", "KC_UB"],
        "default_values": {"length": 20, "multiplier": 2.0},
    },
    "SUPERTREND": {
        "function": "supertrend",
        "params": {"length": ["length"], "multiplier": ["multiplier", "factor"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "complex",
        "return_cols": ["ST", "D"],
        "default_values": {"length": 10, "multiplier": 3.0},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 10) + 10,
    },
    "DONCHIAN": {
        "function": "donchian",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "multiple",
        "return_cols": ["DC_LB", "DC_MB", "DC_UB"],
        "default_values": {"length": 20},
    },
    "ACCBANDS": {
        "function": "accbands",
        "params": {"length": ["length"], "std": ["scale"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "multiple",
        "return_cols": ["ACC_LB", "ACC_MB", "ACC_UB"],
        "default_values": {"length": 30, "std": 4.0},
    },
    "UI": {
        "function": "ui",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
        "min_length": lambda params: get_param_value(params, ["length", "window"], 14),
    },
    "OBV": {
        "function": "obv",
        "params": {},
        "multi_column": True,
        "data_columns": ["Close", "Volume"],
        "returns": "single",
        "default_values": {},
    },
    "AD": {
        "function": "ad",
        "params": {},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {},
    },
    "ADOSC": {
        "function": "adosc",
        "params": {"fast": ["fast"], "slow": ["slow"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {"fast": 3, "slow": 10},
    },
    "CMF": {
        "function": "cmf",
        "params": {"length": ["length"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {"length": 20},
    },
    "EFI": {
        "function": "efi",
        "params": {
            "length": ["length", "period"],
            "mamode": ["mamode"],
            "drift": ["drift"],
        },
        "multi_column": True,
        "data_columns": ["Close", "Volume"],
        "returns": "single",
        "default_values": {"length": 13, "mamode": "ema", "drift": 1},
    },
    "VWAP": {
        "function": "vwap",
        "params": {},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {},
    },
    "SQUEEZE": {
        "function": "squeeze",
        "params": {
            "bb_length": ["bb_length"],
            "bb_std": ["bb_std"],
            "kc_length": ["kc_length"],
            "kc_scalar": ["kc_scalar"],
            "mom_length": ["mom_length"],
            "mom_smooth": ["mom_smooth"],
            "use_tr": ["use_tr"],
        },
        "multi_column": True,
        "data_columns": ["High", "Low", "Close"],
        "returns": "single",
        "default_values": {
            "bb_length": 20,
            "bb_std": 2.0,
            "kc_length": 20,
            "kc_scalar": 1.5,
            "mom_length": 12,
            "mom_smooth": 6,
            "use_tr": True,
        },
    },
    "MFI": {
        "function": "mfi",
        "params": {"length": ["length"], "drift": ["drift"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {"length": 14, "drift": 1},
    },
    "MONEY_FLOW_INDEX": {
        "function": "mfi",
        "params": {"length": ["length"], "drift": ["drift"]},
        "multi_column": True,
        "data_columns": ["High", "Low", "Close", "Volume"],
        "returns": "single",
        "default_values": {"length": 14, "drift": 1},
    },
}

# 動的設定初期化（レジストリから生成）
POSITIONAL_DATA_FUNCTIONS = generate_positional_functions()

initialize_all_indicators()
