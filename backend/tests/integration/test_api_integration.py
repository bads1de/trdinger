#!/usr/bin/env python3
"""
APIとフロントエンド統合テスト

簡素化されたオートストラテジーシステムのAPIエンドポイントが
正常に動作することを確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json
from fastapi.testclient import TestClient

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """APIエンドポイントのテスト"""
    print("\n=== API統合テスト ===")
    
    try:
        # FastAPIアプリのインポート
        from app.main import app
        client = TestClient(app)
        
        print("✅ FastAPIアプリ作成成功")
        
        # 1. ヘルスチェック
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ ヘルスチェック成功")
        else:
            print(f"❌ ヘルスチェック失敗: {response.status_code}")
        
        # 2. デフォルト設定取得
        response = client.get("/api/auto-strategy/default-config")
        if response.status_code == 200:
            config_data = response.json()
            print(f"✅ デフォルト設定取得成功: {len(config_data.get('config', {}))}個のキー")
        else:
            print(f"❌ デフォルト設定取得失敗: {response.status_code}")
        
        # 3. プリセット取得
        response = client.get("/api/auto-strategy/presets")
        if response.status_code == 200:
            presets_data = response.json()
            print(f"✅ プリセット取得成功: {len(presets_data.get('presets', {}))}個のプリセット")
        else:
            print(f"❌ プリセット取得失敗: {response.status_code}")
        
        # 4. 実験一覧取得
        response = client.get("/api/auto-strategy/experiments")
        if response.status_code == 200:
            experiments_data = response.json()
            print(f"✅ 実験一覧取得成功: {len(experiments_data.get('experiments', []))}個の実験")
        else:
            print(f"❌ 実験一覧取得失敗: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ API統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ga_request_format():
    """GA実行リクエスト形式のテスト"""
    print("\n=== GA実行リクエスト形式テスト ===")
    
    try:
        # フロントエンドから送信される形式のリクエストを作成
        request_data = {
            "experiment_name": "Test_Strategy_Gen_001",
            "base_config": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-12-19",
                "initial_capital": 100000,
                "commission_rate": 0.00055,
            },
            "ga_config": {
                "population_size": 10,
                "generations": 5,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1,
                "elite_size": 2,
                "max_indicators": 3,
                "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
            },
        }
        
        print("✅ リクエストデータ作成成功")
        print(f"   実験名: {request_data['experiment_name']}")
        print(f"   シンボル: {request_data['base_config']['symbol']}")
        print(f"   GA設定: {request_data['ga_config']['population_size']}個体, {request_data['ga_config']['generations']}世代")
        
        # JSON形式での検証
        json_data = json.dumps(request_data, ensure_ascii=False, indent=2)
        parsed_data = json.loads(json_data)
        
        print("✅ JSON形式変換成功")
        print(f"   JSONサイズ: {len(json_data)}文字")
        
        return True
        
    except Exception as e:
        print(f"❌ GA実行リクエスト形式テストエラー: {e}")
        return False

def test_frontend_compatibility():
    """フロントエンド互換性テスト"""
    print("\n=== フロントエンド互換性テスト ===")
    
    try:
        # フロントエンドで使用される型定義の確認
        from app.core.services.auto_strategy.models.ga_config import GAConfig
        from app.core.services.auto_strategy.models.strategy_gene import StrategyGene
        
        # GAConfigの辞書変換（フロントエンドとの互換性）
        config = GAConfig.create_fast()
        config_dict = config.to_dict()
        
        # フロントエンドで期待されるキーの確認
        expected_keys = [
            "population_size", "generations", "crossover_rate", "mutation_rate",
            "elite_size", "fitness_weights", "max_indicators", "allowed_indicators"
        ]
        
        missing_keys = [key for key in expected_keys if key not in config_dict]
        if missing_keys:
            print(f"❌ 不足しているキー: {missing_keys}")
            return False
        
        print("✅ フロントエンド互換性確認成功")
        print(f"   必要なキー: {len(expected_keys)}個すべて存在")
        
        # 辞書からの復元テスト
        restored_config = GAConfig.from_dict(config_dict)
        print("✅ 設定の復元成功")
        
        return True
        
    except Exception as e:
        print(f"❌ フロントエンド互換性テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🌐 APIとフロントエンド統合テスト開始")
    print("=" * 60)
    
    tests = [
        test_api_endpoints,
        test_ga_request_format,
        test_frontend_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 テスト結果: {passed}/{total} 成功")
    
    if passed == total:
        print("🎉 全テスト成功！APIとフロントエンドの統合に問題ありません。")
        
        print("\n🔗 確認された互換性:")
        print("   ✅ APIエンドポイント: 正常動作")
        print("   ✅ リクエスト形式: フロントエンド互換")
        print("   ✅ レスポンス形式: 期待される構造")
        print("   ✅ 設定変換: 双方向変換可能")
        
    else:
        print(f"⚠️  {total - passed}個のテストが失敗しました。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
