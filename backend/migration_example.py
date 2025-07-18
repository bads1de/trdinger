#!/usr/bin/env python3
"""
統一設定システムへの移行例

既存コードを統一設定システムに移行する方法を示すサンプルコードです。
"""

# === 移行前（レガシー） ===

# 古い方法：個別の設定ファイルからインポート
from app.config.settings import settings
from app.config.market_config import MarketDataConfig

def legacy_api_endpoint_example():
    """レガシーAPIエンドポイントの例"""
    
    # 古い方法：直接設定クラスにアクセス
    app_name = settings.app_name
    debug_mode = settings.debug
    default_symbol = MarketDataConfig.DEFAULT_SYMBOL
    default_timeframe = MarketDataConfig.DEFAULT_TIMEFRAME
    
    # 古い方法：バリデーション
    is_valid_symbol = MarketDataConfig.validate_symbol("BTC/USDT:USDT")
    is_valid_timeframe = MarketDataConfig.validate_timeframe("1h")
    
    print("=== レガシー方式 ===")
    print(f"アプリ名: {app_name}")
    print(f"デバッグモード: {debug_mode}")
    print(f"デフォルトシンボル: {default_symbol}")
    print(f"デフォルト時間軸: {default_timeframe}")
    print(f"シンボル検証: {is_valid_symbol}")
    print(f"時間軸検証: {is_valid_timeframe}")


# === 移行後（統一設定システム） ===

# 新しい方法：統一設定システムからインポート
from app.config import unified_config, MarketDataValidator

def modern_api_endpoint_example():
    """モダンAPIエンドポイントの例"""
    
    # 新しい方法：階層的な設定アクセス
    app_name = unified_config.app.app_name
    debug_mode = unified_config.app.debug
    default_symbol = unified_config.market.default_symbol
    default_timeframe = unified_config.market.default_timeframe
    
    # 新しい方法：専用バリデーター使用
    is_valid_symbol = MarketDataValidator.validate_symbol(
        "BTC/USDT:USDT", 
        unified_config.market.supported_symbols
    )
    is_valid_timeframe = MarketDataValidator.validate_timeframe(
        "1h", 
        unified_config.market.supported_timeframes
    )
    
    print("\n=== 統一設定システム方式 ===")
    print(f"アプリ名: {app_name}")
    print(f"デバッグモード: {debug_mode}")
    print(f"デフォルトシンボル: {default_symbol}")
    print(f"デフォルト時間軸: {default_timeframe}")
    print(f"シンボル検証: {is_valid_symbol}")
    print(f"時間軸検証: {is_valid_timeframe}")


# === FastAPI エンドポイントの移行例 ===

from fastapi import Query

def legacy_fastapi_endpoint():
    """レガシーFastAPIエンドポイントの例"""
    
    # 古い方法：直接設定クラス参照
    def get_ohlcv_data(
        symbol: str = Query(..., description="取引ペアシンボル"),
        timeframe: str = Query(
            MarketDataConfig.DEFAULT_TIMEFRAME,
            description="時間軸",
        ),
        limit: int = Query(
            MarketDataConfig.DEFAULT_LIMIT,
            ge=MarketDataConfig.MIN_LIMIT,
            le=MarketDataConfig.MAX_LIMIT,
            description="取得するデータ数",
        ),
    ):
        # バリデーション
        if not MarketDataConfig.validate_symbol(symbol):
            raise ValueError("無効なシンボル")
        
        if not MarketDataConfig.validate_timeframe(timeframe):
            raise ValueError("無効な時間軸")
        
        if not MarketDataConfig.validate_limit(limit):
            raise ValueError("無効な制限値")
        
        return {"symbol": symbol, "timeframe": timeframe, "limit": limit}
    
    return get_ohlcv_data


def modern_fastapi_endpoint():
    """モダンFastAPIエンドポイントの例"""
    
    # 新しい方法：統一設定システム使用
    def get_ohlcv_data(
        symbol: str = Query(..., description="取引ペアシンボル"),
        timeframe: str = Query(
            unified_config.market.default_timeframe,
            description="時間軸",
        ),
        limit: int = Query(
            unified_config.market.default_limit,
            ge=unified_config.market.min_limit,
            le=unified_config.market.max_limit,
            description="取得するデータ数",
        ),
    ):
        # バリデーション（専用バリデーター使用）
        if not MarketDataValidator.validate_symbol(
            symbol, unified_config.market.supported_symbols
        ):
            raise ValueError("無効なシンボル")
        
        if not MarketDataValidator.validate_timeframe(
            timeframe, unified_config.market.supported_timeframes
        ):
            raise ValueError("無効な時間軸")
        
        if not MarketDataValidator.validate_limit(
            limit, unified_config.market.min_limit, unified_config.market.max_limit
        ):
            raise ValueError("無効な制限値")
        
        return {"symbol": symbol, "timeframe": timeframe, "limit": limit}
    
    return get_ohlcv_data


# === サービスクラスの移行例 ===

class LegacyMarketDataService:
    """レガシー市場データサービスの例"""
    
    def __init__(self):
        # 古い方法：直接設定クラス参照
        self.default_symbol = MarketDataConfig.DEFAULT_SYMBOL
        self.default_exchange = MarketDataConfig.DEFAULT_EXCHANGE
        self.bybit_config = MarketDataConfig.BYBIT_CONFIG
        
    def normalize_symbol(self, symbol: str) -> str:
        # 古い方法：設定クラスのメソッド使用
        return MarketDataConfig.normalize_symbol(symbol)
    
    def get_config_info(self):
        return {
            "default_symbol": self.default_symbol,
            "default_exchange": self.default_exchange,
            "bybit_config": self.bybit_config,
        }


class ModernMarketDataService:
    """モダン市場データサービスの例"""
    
    def __init__(self):
        # 新しい方法：統一設定システム使用
        self.market_config = unified_config.market
        
    def normalize_symbol(self, symbol: str) -> str:
        # 新しい方法：専用バリデーター使用
        return MarketDataValidator.normalize_symbol(
            symbol,
            self.market_config.symbol_mapping,
            self.market_config.supported_symbols
        )
    
    def get_config_info(self):
        return {
            "default_symbol": self.market_config.default_symbol,
            "default_exchange": self.market_config.default_exchange,
            "bybit_config": self.market_config.bybit_config,
        }


# === 環境変数の活用例 ===

def environment_variable_example():
    """環境変数の活用例"""
    
    print("\n=== 環境変数の活用 ===")
    
    # 統一設定システムは環境変数を自動的に読み込む
    # 例：APP__DEBUG=true, MARKET__DEFAULT_SYMBOL=ETH/USDT:USDT
    
    print(f"アプリデバッグ: {unified_config.app.debug}")
    print(f"データベースホスト: {unified_config.database.host}")
    print(f"市場デフォルトシンボル: {unified_config.market.default_symbol}")
    print(f"ログレベル: {unified_config.logging.level}")
    
    # 設定の妥当性検証
    is_valid = unified_config.validate_all()
    print(f"設定妥当性: {is_valid}")


# === 移行のメリット ===

def migration_benefits():
    """移行のメリットを示す例"""
    
    print("\n=== 移行のメリット ===")
    
    # 1. 階層的な設定アクセス
    print("1. 階層的な設定アクセス:")
    print(f"   アプリ設定: {unified_config.app.app_name}")
    print(f"   DB設定: {unified_config.database.host}:{unified_config.database.port}")
    print(f"   市場設定: {unified_config.market.default_symbol}")
    
    # 2. 環境変数の自動読み込み
    print("\n2. 環境変数の自動読み込み:")
    print("   APP__DEBUG=true → unified_config.app.debug")
    print("   DB__HOST=localhost → unified_config.database.host")
    
    # 3. 型安全性
    print("\n3. 型安全性:")
    print(f"   ポート番号（int）: {unified_config.app.port}")
    print(f"   デバッグフラグ（bool）: {unified_config.app.debug}")
    
    # 4. バリデーション分離
    print("\n4. バリデーション分離:")
    print("   設定クラス：データ保持のみ")
    print("   バリデーター：検証ロジックのみ")
    
    # 5. 一元管理
    print("\n5. 一元管理:")
    print("   全設定が unified_config 経由でアクセス可能")


def main():
    """メイン実行関数"""
    print("🚀 統一設定システム移行例")
    print("=" * 50)
    
    # 移行前後の比較
    legacy_api_endpoint_example()
    modern_api_endpoint_example()
    
    # サービスクラスの比較
    print("\n=== サービスクラス比較 ===")
    
    legacy_service = LegacyMarketDataService()
    modern_service = ModernMarketDataService()
    
    print("レガシーサービス設定:")
    print(legacy_service.get_config_info())
    
    print("\nモダンサービス設定:")
    print(modern_service.get_config_info())
    
    # 環境変数の活用
    environment_variable_example()
    
    # 移行のメリット
    migration_benefits()
    
    print("\n" + "=" * 50)
    print("✅ 移行例の実行完了")


if __name__ == "__main__":
    main()
