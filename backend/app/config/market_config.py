"""
市場データサービスの設定管理

このモジュールは、CCXT ライブラリを使用した市場データ取得に関する
設定を管理します。
"""

from dataclasses import dataclass


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
                f"サポートされていないシンボルです: '{symbol}'。サポートされているシンボルは {', '.join(cls.SUPPORTED_SYMBOLS)} です。"
            )

        return normalized


# 設定のインスタンス
config = MarketDataConfig()
