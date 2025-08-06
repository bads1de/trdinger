"""
フロントエンド・バックエンド統合テスト

フロントエンドのML設定管理機能とバックエンドAPIの統合をテストします。
"""

import requests

BACKEND_URL = "http://127.0.0.1:8001"
FRONTEND_URL = "http://localhost:3000"

def test_backend_endpoints():
    """バックエンドエンドポイントのテスト"""
    print("=== バックエンドエンドポイントテスト ===")
    
    endpoints = [
        ("GET", "/api/ml/config", "ML設定取得"),
        ("GET", "/api/ml/models", "MLモデル一覧取得"),
        ("GET", "/api/ml/feature-importance", "特徴量重要度取得"),
        ("GET", "/api/ml/status", "MLステータス取得"),
        ("GET", "/docs", "API仕様書"),
    ]
    
    results = {}
    
    for method, endpoint, description in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
            
            status = "✅ 成功" if response.status_code == 200 else f"❌ 失敗 ({response.status_code})"
            print(f"{description}: {status}")
            results[endpoint] = response.status_code == 200
            
        except Exception as e:
            print(f"{description}: ❌ エラー ({e})")
            results[endpoint] = False
    
    return results

def test_ml_config_crud():
    """ML設定のCRUD操作テスト"""
    print("\n=== ML設定CRUD操作テスト ===")
    
    # 1. 初期設定取得
    try:
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        if response.status_code != 200:
            print("❌ 初期設定取得失敗")
            return False
        
        initial_config = response.json()
        print("✅ 初期設定取得成功")
        
        # 2. 設定更新
        update_data = {
            "prediction": {
                "default_up_prob": 0.35,
                "default_down_prob": 0.35,
                "default_range_prob": 0.3
            }
        }
        
        response = requests.put(
            f"{BACKEND_URL}/api/ml/config",
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print("❌ 設定更新失敗")
            return False
        
        result = response.json()
        if not result.get("success"):
            print("❌ 設定更新失敗（APIレスポンス）")
            return False
        
        print("✅ 設定更新成功")
        
        # 3. 更新確認
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        updated_config = response.json()
        
        if updated_config["prediction"]["default_up_prob"] == 0.35:
            print("✅ 設定更新確認成功")
        else:
            print("❌ 設定更新確認失敗")
            return False
        
        # 4. 設定リセット
        response = requests.post(f"{BACKEND_URL}/api/ml/config/reset")
        if response.status_code != 200:
            print("❌ 設定リセット失敗")
            return False
        
        result = response.json()
        if not result.get("success"):
            print("❌ 設定リセット失敗（APIレスポンス）")
            return False
        
        print("✅ 設定リセット成功")
        
        # 5. リセット確認
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        reset_config = response.json()
        
        if reset_config["prediction"]["default_up_prob"] == initial_config["prediction"]["default_up_prob"]:
            print("✅ 設定リセット確認成功")
            return True
        else:
            print("❌ 設定リセット確認失敗")
            return False
        
    except Exception as e:
        print(f"❌ CRUD操作エラー: {e}")
        return False

def test_frontend_accessibility():
    """フロントエンドアクセシビリティテスト"""
    print("\n=== フロントエンドアクセシビリティテスト ===")
    
    try:
        # フロントエンドのメインページにアクセス
        response = requests.get(FRONTEND_URL, timeout=10)
        if response.status_code == 200:
            print("✅ フロントエンドメインページアクセス成功")
        else:
            print(f"❌ フロントエンドメインページアクセス失敗 ({response.status_code})")
            return False
        
        # MLページにアクセス
        response = requests.get(f"{FRONTEND_URL}/ml", timeout=10)
        if response.status_code == 200:
            print("✅ MLページアクセス成功")
            return True
        else:
            print(f"❌ MLページアクセス失敗 ({response.status_code})")
            return False
        
    except Exception as e:
        print(f"❌ フロントエンドアクセスエラー: {e}")
        return False

def test_cors_configuration():
    """CORS設定テスト"""
    print("\n=== CORS設定テスト ===")
    
    try:
        # フロントエンドからのリクエストをシミュレート
        headers = {
            "Origin": "http://localhost:3000",
            "Content-Type": "application/json"
        }
        
        response = requests.get(f"{BACKEND_URL}/api/ml/config", headers=headers)
        
        # CORSヘッダーの確認
        cors_headers = {
            "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
            "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
            "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers"),
        }
        
        if cors_headers["Access-Control-Allow-Origin"]:
            print("✅ CORS設定確認成功")
            print(f"   Allow-Origin: {cors_headers['Access-Control-Allow-Origin']}")
            return True
        else:
            print("⚠️ CORS設定が確認できませんでした")
            return False
        
    except Exception as e:
        print(f"❌ CORS設定テストエラー: {e}")
        return False

def test_api_response_format():
    """APIレスポンス形式テスト"""
    print("\n=== APIレスポンス形式テスト ===")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        config = response.json()
        
        # 必要なセクションの存在確認
        required_sections = [
            "data_processing", "model", "training", 
            "prediction", "ensemble", "retraining"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if not missing_sections:
            print("✅ APIレスポンス形式確認成功")
            print(f"   含まれるセクション: {list(config.keys())}")
            return True
        else:
            print(f"❌ 不足しているセクション: {missing_sections}")
            return False
        
    except Exception as e:
        print(f"❌ APIレスポンス形式テストエラー: {e}")
        return False

def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")
    
    try:
        # 無効なエンドポイントへのアクセス
        response = requests.get(f"{BACKEND_URL}/api/ml/invalid-endpoint")
        if response.status_code == 404:
            print("✅ 無効なエンドポイントエラーハンドリング成功")
        else:
            print(f"❌ 無効なエンドポイントエラーハンドリング失敗 ({response.status_code})")
        
        # 無効なJSONデータでの更新
        response = requests.put(
            f"{BACKEND_URL}/api/ml/config",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 422:  # Unprocessable Entity
            print("✅ 無効なJSONエラーハンドリング成功")
            return True
        else:
            print(f"❌ 無効なJSONエラーハンドリング失敗 ({response.status_code})")
            return False
        
    except Exception as e:
        print(f"❌ エラーハンドリングテストエラー: {e}")
        return False

def main():
    """メイン統合テスト実行"""
    print("🚀 フロントエンド・バックエンド統合テスト開始")
    print("=" * 60)
    
    test_results = {}
    
    # バックエンドエンドポイントテスト
    backend_results = test_backend_endpoints()
    test_results["backend_endpoints"] = all(backend_results.values())
    
    # ML設定CRUD操作テスト
    test_results["ml_config_crud"] = test_ml_config_crud()
    
    # フロントエンドアクセシビリティテスト
    test_results["frontend_accessibility"] = test_frontend_accessibility()
    
    # CORS設定テスト
    test_results["cors_configuration"] = test_cors_configuration()
    
    # APIレスポンス形式テスト
    test_results["api_response_format"] = test_api_response_format()
    
    # エラーハンドリングテスト
    test_results["error_handling"] = test_error_handling()
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 統合テスト結果サマリー")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n合計: {passed_tests}/{total_tests} テスト成功")
    
    if passed_tests == total_tests:
        print("🎉 全ての統合テストが成功しました！")
        print("フロントエンドとバックエンドが正常に連携しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
        print("失敗したテストを確認して修正してください。")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
