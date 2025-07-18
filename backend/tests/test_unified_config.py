#!/usr/bin/env python3
"""
統一設定システムの動作確認テスト

リファクタリング後の統一設定システムとバリデーターが正常に動作することを確認します。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from app.config import (
        unified_config,
        MarketDataValidator,
        MLConfigValidator,
        DatabaseValidator,
        AppValidator,
    )
    print("✅ 統一設定システムのインポート成功")
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)


def test_unified_config():
    """統一設定システムのテスト"""
    print("\n🧪 統一設定システムテスト")
    
    try:
        # 基本設定の確認
        assert unified_config.app.app_name == "Trdinger Trading API"
        assert unified_config.app.port == 8000
        assert unified_config.database.port == 5432
        assert unified_config.market.default_symbol == "BTC/USDT:USDT"
        
        # 設定検証
        assert unified_config.validate_all() is True
        
        print("✅ 統一設定システムテスト成功")
        return True
    except Exception as e:
        print(f"❌ 統一設定システムテスト失敗: {e}")
        return False


def test_market_validator():
    """市場データバリデーターのテスト"""
    print("\n🧪 市場データバリデーターテスト")
    
    try:
        # シンボル検証
        assert MarketDataValidator.validate_symbol("BTC/USDT:USDT", ["BTC/USDT:USDT"]) is True
        assert MarketDataValidator.validate_symbol("INVALID", ["BTC/USDT:USDT"]) is False
        
        # 時間軸検証
        timeframes = ["15m", "30m", "1h", "4h", "1d"]
        assert MarketDataValidator.validate_timeframe("1h", timeframes) is True
        assert MarketDataValidator.validate_timeframe("5m", timeframes) is False
        
        # 制限値検証
        assert MarketDataValidator.validate_limit(100, 1, 1000) is True
        assert MarketDataValidator.validate_limit(0, 1, 1000) is False
        assert MarketDataValidator.validate_limit(1001, 1, 1000) is False
        
        # シンボル正規化
        mapping = {"BTCUSDT": "BTC/USDT:USDT"}
        supported = ["BTC/USDT:USDT"]
        normalized = MarketDataValidator.normalize_symbol("BTCUSDT", mapping, supported)
        assert normalized == "BTC/USDT:USDT"
        
        print("✅ 市場データバリデーターテスト成功")
        return True
    except Exception as e:
        print(f"❌ 市場データバリデーターテスト失敗: {e}")
        return False


def test_ml_validator():
    """MLバリデーターのテスト"""
    print("\n🧪 MLバリデーターテスト")
    
    try:
        # 予測値検証
        valid_predictions = {"up": 0.33, "down": 0.33, "range": 0.34}
        assert MLConfigValidator.validate_predictions(valid_predictions) is True
        
        invalid_predictions = {"up": 1.5, "down": -0.1, "range": 0.1}
        assert MLConfigValidator.validate_predictions(invalid_predictions) is False
        
        # 確率範囲検証
        assert MLConfigValidator.validate_probability_range(0.0, 1.0, 0.8, 1.2) is True
        assert MLConfigValidator.validate_probability_range(-0.1, 1.0, 0.8, 1.2) is False
        
        # データ処理設定検証
        errors = MLConfigValidator.validate_data_processing_config(
            max_ohlcv_rows=10000,
            max_feature_rows=50000,
            feature_timeout=30,
            training_timeout=300,
            prediction_timeout=10,
        )
        assert len(errors) == 0
        
        # 無効な設定のテスト
        errors = MLConfigValidator.validate_data_processing_config(
            max_ohlcv_rows=-1,
            max_feature_rows=100,
            feature_timeout=0,
            training_timeout=10,
            prediction_timeout=-5,
        )
        assert len(errors) > 0
        
        print("✅ MLバリデーターテスト成功")
        return True
    except Exception as e:
        print(f"❌ MLバリデーターテスト失敗: {e}")
        return False


def test_database_validator():
    """データベースバリデーターのテスト"""
    print("\n🧪 データベースバリデーターテスト")
    
    try:
        # 有効な接続パラメータ
        errors = DatabaseValidator.validate_connection_params(
            host="localhost",
            port=5432,
            name="trdinger",
            user="postgres"
        )
        assert len(errors) == 0
        
        # 無効な接続パラメータ
        errors = DatabaseValidator.validate_connection_params(
            host="",
            port=99999,
            name="",
            user=""
        )
        assert len(errors) > 0
        
        print("✅ データベースバリデーターテスト成功")
        return True
    except Exception as e:
        print(f"❌ データベースバリデーターテスト失敗: {e}")
        return False


def test_app_validator():
    """アプリケーションバリデーターのテスト"""
    print("\n🧪 アプリケーションバリデーターテスト")
    
    try:
        # サーバー設定検証
        errors = AppValidator.validate_server_config("127.0.0.1", 8000)
        assert len(errors) == 0
        
        errors = AppValidator.validate_server_config("", 0)
        assert len(errors) > 0
        
        # CORS設定検証
        errors = AppValidator.validate_cors_origins(["http://localhost:3000", "https://example.com"])
        assert len(errors) == 0
        
        errors = AppValidator.validate_cors_origins(["invalid-url", ""])
        assert len(errors) > 0
        
        print("✅ アプリケーションバリデーターテスト成功")
        return True
    except Exception as e:
        print(f"❌ アプリケーションバリデーターテスト失敗: {e}")
        return False


def test_backward_compatibility():
    """後方互換性のテスト"""
    print("\n🧪 後方互換性テスト")
    
    try:
        # レガシー設定のインポート
        from app.config import settings, MarketDataConfig
        
        # 基本的な動作確認
        assert settings.app_name == "Trdinger Trading API"
        assert MarketDataConfig.DEFAULT_SYMBOL == "BTC/USDT:USDT"
        
        # レガシーバリデーション機能の確認
        assert MarketDataConfig.validate_symbol("BTC/USDT:USDT") is True
        assert MarketDataConfig.validate_timeframe("1h") is True
        assert MarketDataConfig.validate_limit(100) is True
        
        print("✅ 後方互換性テスト成功")
        return True
    except Exception as e:
        print(f"❌ 後方互換性テスト失敗: {e}")
        return False


def main():
    """メインテスト実行"""
    print("🚀 統一設定システム動作確認テスト開始")
    print("=" * 50)
    
    tests = [
        test_unified_config,
        test_market_validator,
        test_ml_validator,
        test_database_validator,
        test_app_validator,
        test_backward_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 テスト結果: {passed} 成功, {failed} 失敗")
    
    if failed == 0:
        print("🎉 全てのテストが成功しました！")
        return 0
    else:
        print("⚠️ 一部のテストが失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
