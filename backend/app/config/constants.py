"""
共通定数

アプリケーション全体で共有される定数を定義します。
設定クラスの循環依存を避けるため、軽量なモジュールとして分離しています。
"""

SUPPORTED_TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
]

DEFAULT_ENSEMBLE_ALGORITHMS = ("lightgbm", "xgboost", "catboost")
DEFAULT_MARKET_EXCHANGE = "bybit"
DEFAULT_MARKET_SYMBOL = "BTC/USDT:USDT"
DEFAULT_MARKET_TIMEFRAME = "1h"
DEFAULT_DATA_LIMIT = 100
MAX_DATA_LIMIT = 1000
MIN_DATA_LIMIT = 1
