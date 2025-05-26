"""
市場データサービスの設定管理

このモジュールは、CCXT ライブラリを使用した市場データ取得に関する
設定を管理します。

@author Trdinger Development Team
@version 1.0.0
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class MarketDataConfig:
    """市場データサービスの設定クラス"""

    # サポートされている取引所
    SUPPORTED_EXCHANGES = ["bybit"]

    # サポートされているシンボル（Bybit形式）- 主要銘柄 + 先物 + イーサリアム
    SUPPORTED_SYMBOLS = [
        # Bitcoin 関連
        "BTC/USDT",  # Bitcoin スポット
        "BTC/USD",   # Bitcoin 先物（永続契約）
        "BTCUSD",    # Bitcoin 先物（代替表記）

        # Ethereum 関連
        "ETH/USDT",  # Ethereum スポット
        "ETH/USD",   # Ethereum 先物（永続契約）
        "ETH/BTC",   # Ethereum/Bitcoin ペア
        "ETHUSD",    # Ethereum 先物（代替表記）

        # その他主要アルトコイン
        "BNB/USDT",  # Binance Coin スポット
        "ADA/USDT",  # Cardano スポット
        "SOL/USDT",  # Solana スポット
        "XRP/USDT",  # Ripple スポット
        "DOT/USDT",  # Polkadot スポット
        "AVAX/USDT",  # Avalanche スポット
        "LTC/USDT",  # Litecoin スポット
        "UNI/USDT",  # Uniswap スポット
    ]

    # サポートされている時間軸
    SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

    # デフォルト設定
    DEFAULT_EXCHANGE = "bybit"
    DEFAULT_SYMBOL = "BTC/USDT"
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
        シンボルが有効かどうかを検証します

        Args:
            symbol: 検証するシンボル

        Returns:
            有効な場合True、無効な場合False
        """
        return symbol in cls.SUPPORTED_SYMBOLS

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """
        時間軸が有効かどうかを検証します

        Args:
            timeframe: 検証する時間軸

        Returns:
            有効な場合True、無効な場合False
        """
        return timeframe in cls.SUPPORTED_TIMEFRAMES

    @classmethod
    def validate_limit(cls, limit: int) -> bool:
        """
        制限値が有効かどうかを検証します

        Args:
            limit: 検証する制限値

        Returns:
            有効な場合True、無効な場合False
        """
        return cls.MIN_LIMIT <= limit <= cls.MAX_LIMIT

    # シンボル正規化マッピング
    SYMBOL_MAPPING = {
        # Bitcoin マッピング
        "BTCUSD": "BTCUSD",      # 先物はそのまま
        "BTC-USD": "BTC/USD",    # 先物（永続契約）
        "BTCUSDT": "BTC/USDT",   # スポット
        "BTC-USDT": "BTC/USDT",  # スポット（代替表記）

        # Ethereum マッピング
        "ETHUSD": "ETHUSD",      # 先物はそのまま
        "ETH-USD": "ETH/USD",    # 先物（永続契約）
        "ETHUSDT": "ETH/USDT",   # スポット
        "ETH-USDT": "ETH/USDT",  # スポット（代替表記）
        "ETHBTC": "ETH/BTC",     # ETH/BTC ペア
        "ETH-BTC": "ETH/BTC",    # ETH/BTC ペア（代替表記）

        # その他のマッピング
        "BNBUSD": "BNB/USDT",
        "ADAUSD": "ADA/USDT",
        "SOLUSD": "SOL/USDT",
        "XRPUSD": "XRP/USDT",
        "DOTUSD": "DOT/USDT",
        "AVAXUSD": "AVAX/USDT",
        "LTCUSD": "LTC/USDT",
        "UNIUSD": "UNI/USDT",
    }

    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """
        シンボルを正規化します

        Args:
            symbol: 正規化するシンボル

        Returns:
            正規化されたシンボル

        Raises:
            ValueError: サポートされていないシンボルの場合
        """
        # 大文字に変換し、空白を除去
        symbol = symbol.strip().upper()

        # マッピングテーブルから検索
        if symbol in cls.SYMBOL_MAPPING:
            normalized = cls.SYMBOL_MAPPING[symbol]
        else:
            normalized = symbol

        # サポートされているシンボルかチェック
        if normalized not in cls.SUPPORTED_SYMBOLS:
            raise ValueError(
                f"サポートされていないシンボルです: {symbol}. "
                f"サポート対象: {', '.join(cls.SUPPORTED_SYMBOLS)}"
            )

        return normalized

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> bool:
        """
        時間軸が有効かチェックします

        Args:
            timeframe: チェックする時間軸

        Returns:
            有効な場合True
        """
        return timeframe in cls.SUPPORTED_TIMEFRAMES

    @classmethod
    def validate_limit(cls, limit: int) -> bool:
        """
        制限値が有効かチェックします

        Args:
            limit: チェックする制限値

        Returns:
            有効な場合True
        """
        return cls.MIN_LIMIT <= limit <= cls.MAX_LIMIT


# 設定のインスタンス
config = MarketDataConfig()
