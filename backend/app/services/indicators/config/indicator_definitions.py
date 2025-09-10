"""
インジケーター定義

各インジケーターの設定を定義し、レジストリに登録します。
"""

import logging


from .momentum_indicators_config import setup_momentum_indicators
from .trend_indicators_config import setup_trend_indicators
from .volatility_indicators_config import setup_volatility_indicators
from .volume_indicators_config import setup_volume_indicators

logger = logging.getLogger(__name__)


def initialize_all_indicators():
    """全インジケーターの設定を初期化"""
    setup_momentum_indicators()
    setup_trend_indicators()
    setup_volatility_indicators()
    setup_volume_indicators()


# モジュール読み込み時に初期化
# python-ta動的処理設定のグローバル設定
PANDAS_TA_CONFIG = {
    "RSI": {
        "function": "rsi",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "SMA": {
        "function": "sma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "EMA": {
        "function": "ema",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "WMA": {
        "function": "wma",
        "params": {"length": ["length", "period"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 20},
    },
    "MACD": {
        "function": "macd",
        "params": {"fast": ["fast"], "slow": ["slow"], "signal": ["signal"]},
        "data_column": "Close",
        "returns": "multiple",
        "return_cols": ["MACD", "Signal", "Histogram"],
        "default_values": {"fast": 12, "slow": 26, "signal": 9},
    },
    "SUPERTREND": {
        "function": "supertrend",
        "params": {"length": ["length"], "multiplier": ["multiplier", "factor"]},
        "data_column": "open_high_low_close",
        "returns": "complex",
        "return_cols": ["ST", "D"],
        "default_values": {"length": 10, "multiplier": 3.0},
    },
    "UI": {
        "function": "ui",
        "params": {"length": ["length"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 14},
    },
    "TEMA": {
        "function": "tema",
        "params": {"length": ["length"]},
        "data_column": "Close",
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
    },
    "T3": {
        "function": "t3",
        "params": {"length": ["length", "period"], "a": ["a", "vfactor"]},
        "data_column": "Close",
        "returns": "single",
        "default_values": {"length": 5, "a": 0.7},
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
}

POSITIONAL_DATA_FUNCTIONS = {
    "rsi",
    "macd",
    "stoch",
    "cci",
    "mom",
    "adx",
    "willr",
    "roc",
    "qqe",
    "sma",
    "ema",
    "wma",
    "dema",
    "tema",
    "t3",
    "kama",
    "sar",
    "atr",
    "bbands",
    "accbands",
    "ui",
    "obv",
    "ad",
    "adosc",
    "efi",
    "vwap",
    "cmf",
    "keltner",
    "donchian",
    "supertrend",
}

# ---- Append new pandas-ta indicators and custom ones ----
initialize_all_indicators()


def setup_pandas_ta_indicators():
    """
    pandas-ta設定からインジケーターを登録
    """
    # PANDAS_TA_CONFIGを使用してインジケーターを登録（外から参照可能）
    pass


# 初期化時にpandas-taインジケーターを設定
def initialize_pandas_ta_indicators():
    """pandas-taインジケーターの初期化"""
    setup_pandas_ta_indicators()


# モジュール読み込み時に初期化
initialize_pandas_ta_indicators()
