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
    SUPPORTED_EXCHANGES = ['bybit']
    
    # サポートされているシンボル（Bybit形式）
    SUPPORTED_SYMBOLS = [
        'BTC/USD:BTC',  # BTC無期限先物
        'BTC/USDT',     # BTCスポット
        'ETH/USD:ETH',  # ETH無期限先物
        'ETH/USDT',     # ETHスポット
    ]
    
    # サポートされている時間軸
    SUPPORTED_TIMEFRAMES = [
        '1m', '5m', '15m', '30m', 
        '1h', '4h', '1d'
    ]
    
    # デフォルト設定
    DEFAULT_EXCHANGE = 'bybit'
    DEFAULT_SYMBOL = 'BTC/USD:BTC'
    DEFAULT_TIMEFRAME = '1h'
    DEFAULT_LIMIT = 100
    
    # 制限値
    MIN_LIMIT = 1
    MAX_LIMIT = 1000
    
    # Bybit固有の設定
    BYBIT_CONFIG = {
        'sandbox': False,  # 本番環境を使用
        'enableRateLimit': True,  # レート制限を有効化
        'timeout': 30000,  # タイムアウト（ミリ秒）
    }
    
    # シンボル正規化マッピング
    SYMBOL_MAPPING = {
        'BTCUSD': 'BTC/USD:BTC',
        'BTC/USD': 'BTC/USD:BTC',
        'BTCUSDT': 'BTC/USDT',
        'BTC-USD': 'BTC/USD:BTC',
        'BTC-USDT': 'BTC/USDT',
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
        # 大文字に変換
        symbol = symbol.upper()
        
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
