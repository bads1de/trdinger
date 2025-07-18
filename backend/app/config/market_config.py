"""
市場データサービスの設定管理（レガシー）

このモジュールは後方互換性のために残されています。
新しいコードでは unified_config.MarketConfig を使用してください。
"""

from dataclasses import dataclass
from .validators import MarketDataValidator


@dataclass
class MarketDataConfig:
    """市場データサービスの設定クラス"""

    # サポートされている取引所
    SUPPORTED_EXCHANGES = ["bybit"]

    # サポートされているシンボル（Bybit形式）- BTCUSDT無期限先物のみ
    SUPPORTED_SYMBOLS = [
        "BTC/USDT:USDT",
    ]

    # サポートされている時間軸
    SUPPORTED_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]

    # デフォルト設定
    DEFAULT_EXCHANGE = "bybit"
    DEFAULT_SYMBOL = "BTC/USDT:USDT"
    DEFAULT_TIMEFRAME = "1h"
    DEFAULT_LIMIT = 100

    # 制限値
    MIN_LIMIT = 1
    MAX_LIMIT = 1000

    # Bybit固有の設定
    BYBIT_CONFIG = {
        "sandbox": False,  # 本番環境を使用
        "enableRateLimit": True,  # レート制限を有効化
        "timeout": 30000,  # タイムアウト（ミリ秒）
    }

    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        シンボルが有効かどうかを検証します（レガシー）

        新しいコードでは MarketDataValidator.validate_symbol を使用してください。
        """
        return MarketDataValidator.validate_symbol(symbol, cls.SUPPORTED_SYMBOLS)

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """
        時間軸が有効かどうかを検証します（レガシー）

        新しいコードでは MarketDataValidator.validate_timeframe を使用してください。
        """
        return MarketDataValidator.validate_timeframe(
            timeframe, cls.SUPPORTED_TIMEFRAMES
        )

    @classmethod
    def validate_limit(cls, limit: int) -> bool:
        """
        制限値が有効かどうかを検証します（レガシー）

        新しいコードでは MarketDataValidator.validate_limit を使用してください。
        """
        return MarketDataValidator.validate_limit(limit, cls.MIN_LIMIT, cls.MAX_LIMIT)

    # シンボル正規化マッピング（BTCのみ）
    SYMBOL_MAPPING = {
        # Bitcoin マッピング - 全てBTC/USDT:USDTに正規化
        "BTCUSDT": "BTC/USDT:USDT",  # USDT永続契約
        "BTC-USDT": "BTC/USDT:USDT",  # 様々な表記からUSDT永続契約へ
        "BTC/USDT": "BTC/USDT:USDT",  # スポット表記から永続契約へ
        "BTC/USDT:USDT": "BTC/USDT:USDT",  # 永続契約正規化
        "BTCUSDT_PERP": "BTC/USDT:USDT",  # 永続契約表記
    }

    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """
        シンボルを正規化します（レガシー）

        新しいコードでは MarketDataValidator.normalize_symbol を使用してください。
        """
        return MarketDataValidator.normalize_symbol(
            symbol, cls.SYMBOL_MAPPING, cls.SUPPORTED_SYMBOLS
        )


# 設定のインスタンス
config = MarketDataConfig()
