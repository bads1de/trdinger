"""
共通定数モジュール

アプリケーション全体で共有される定数を定義します。

このモジュールは設定クラスの循環依存を避けるため、
軽量な独立モジュールとして分離されています。

主な定数カテゴリ:
- 時間軸（SUPPORTED_TIMEFRAMES）: サポートされるOHLCVデータの間隔
- 取引所・シンボル（DEFAULT_MARKET_*）: デフォルトの市場データ設定
- アンサンブル（DEFAULT_ENSEMBLE_*）: デフォルトのMLアルゴリズム

設計方針:
- 環境変数で上書き可能な値は `unified_config.py` の設定クラスで定義
- ここで定義する値は「変更されない固定的な定数」に限定
- 新しい定数を追加する際は、設定変更が必要か検討してから追加してください
"""

# サポートされている時間軸のリスト
# 取引所から取得可能なOHLCVデータの時間間隔
SUPPORTED_TIMEFRAMES = [
    "1m",  # 1分足
    "5m",  # 5分足
    "15m",  # 15分足
    "30m",  # 30分足
    "1h",  # 1時間足
    "4h",  # 4時間足
    "1d",  # 日足
]

# デフォルトのアンサンブル学習アルゴリズム
# アンサンブル学習で使用するベースモデルの種類
DEFAULT_ENSEMBLE_ALGORITHMS = ("lightgbm", "xgboost", "catboost")
# デフォルトの取引所
# 市場データの取得に使用するデフォルトの取引所名
DEFAULT_MARKET_EXCHANGE = "bybit"
# デフォルトの取引ペアシンボル
# Bybit形式のシンボル表記（BTC/USDTの無期限契約）
DEFAULT_MARKET_SYMBOL = "BTC/USDT:USDT"
# デフォルトの時間軸
# データ取得・バックテストで使用するデフォルトの時間間隔
DEFAULT_MARKET_TIMEFRAME = "1h"
# デフォルトのデータ取得件数
# APIから一度に取得するデータのデフォルト件数
DEFAULT_DATA_LIMIT = 100
# 最大データ取得件数
# APIから一度に取得できるデータの最大件数
MAX_DATA_LIMIT = 1000
# 最小データ取得件数
# APIから一度に取得できるデータの最小件数
MIN_DATA_LIMIT = 1
