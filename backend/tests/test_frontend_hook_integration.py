"""
フロントエンドフック・バックエンドAPI連携テスト

useMLSettingsフックとバックエンドML設定管理APIの連携をテストします。
"""

import requests
import json
import time
from typing import Dict, Any

BACKEND_URL = "http://127.0.0.1:8001"

def test_hook_api_compatibility():
    """フロントエンドフックとAPIの互換性テスト"""
    print("=== フロントエンドフック・API互換性テスト ===")
    
    # フロントエンドのuseMLSettingsフックが期待するAPIエンドポイントをテスト
    endpoints_to_test = [
        {
            "name": "fetchConfig",
            "method": "GET",
            "url": "/api/ml/config",
            "description": "ML設定取得（useMLSettings.fetchConfig）"
        },
        {
            "name": "saveConfig", 
            "method": "PUT",
            "url": "/api/ml/config",
            "description": "ML設定更新（useMLSettings.saveConfig）",
            "test_data": {
                "prediction": {
                    "default_up_prob": 0.35,
                    "default_down_prob": 0.35,
                    "default_range_prob": 0.3
                }
            }
        },
        {
            "name": "resetToDefaults",
            "method": "POST", 
            "url": "/api/ml/config/reset",
            "description": "ML設定リセット（useMLSettings.resetToDefaults）"
        }
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        try:
            print(f"\n--- {endpoint['description']} ---")
            
            if endpoint["method"] == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint['url']}")
            elif endpoint["method"] == "PUT":
                response = requests.put(
                    f"{BACKEND_URL}{endpoint['url']}",
                    json=endpoint.get("test_data", {}),
                    headers={"Content-Type": "application/json"}
                )
            elif endpoint["method"] == "POST":
                response = requests.post(f"{BACKEND_URL}{endpoint['url']}")
            
            print(f"ステータスコード: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {endpoint['name']} API成功")
                
                # レスポンス形式の確認
                if endpoint["name"] == "fetchConfig":
                    # フロントエンドのMLConfig型と一致するか確認
                    required_sections = [
                        "data_processing", "model", "training", "prediction"
                    ]
                    missing_sections = [s for s in required_sections if s not in data]
                    if not missing_sections:
                        print("  ✅ MLConfig型と互換性あり")
                        results[endpoint["name"]] = True
                    else:
                        print(f"  ❌ 不足セクション: {missing_sections}")
                        results[endpoint["name"]] = False
                
                elif endpoint["name"] in ["saveConfig", "resetToDefaults"]:
                    # APIResponseHelper形式の確認
                    if "success" in data and "message" in data:
                        print("  ✅ APIResponseHelper形式と互換性あり")
                        if data.get("success"):
                            print(f"  ✅ 操作成功: {data.get('message')}")
                            results[endpoint["name"]] = True
                        else:
                            print(f"  ❌ 操作失敗: {data.get('message')}")
                            results[endpoint["name"]] = False
                    else:
                        print("  ❌ APIResponseHelper形式と不一致")
                        results[endpoint["name"]] = False
                
            else:
                print(f"❌ {endpoint['name']} API失敗: {response.status_code}")
                results[endpoint["name"]] = False
                
        except Exception as e:
            print(f"❌ {endpoint['name']} エラー: {e}")
            results[endpoint["name"]] = False
    
    return results

def test_ml_config_type_compatibility():
    """MLConfig型の互換性テスト"""
    print("\n=== MLConfig型互換性テスト ===")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        if response.status_code != 200:
            print("❌ ML設定取得失敗")
            return False
        
        config = response.json()
        
        # フロントエンドのMLConfig型で定義されている必須フィールドをチェック
        expected_structure = {
            "data_processing": [
                "max_ohlcv_rows", "max_feature_rows", 
                "feature_calculation_timeout", "model_training_timeout"
            ],
            "model": [
                "model_save_path", "max_model_versions", "model_retention_days"
            ],
            "training": [
                "train_test_split", "prediction_horizon", 
                "threshold_up", "threshold_down"
            ],
            "prediction": [
                "default_up_prob", "default_down_prob", "default_range_prob"
            ]
        }
        
        compatibility_issues = []
        
        for section, fields in expected_structure.items():
            if section not in config:
                compatibility_issues.append(f"セクション '{section}' が存在しません")
                continue
            
            for field in fields:
                if field not in config[section]:
                    compatibility_issues.append(f"フィールド '{section}.{field}' が存在しません")
        
        if not compatibility_issues:
            print("✅ MLConfig型完全互換")
            
            # 型の確認
            print("\n--- 型チェック ---")
            type_checks = [
                ("data_processing.max_ohlcv_rows", int),
                ("training.train_test_split", (int, float)),
                ("prediction.default_up_prob", (int, float))
            ]
            
            for field_path, expected_type in type_checks:
                sections = field_path.split('.')
                value = config
                for section in sections:
                    value = value[section]
                
                if isinstance(value, expected_type):
                    print(f"  ✅ {field_path}: {type(value).__name__}")
                else:
                    print(f"  ❌ {field_path}: 期待型 {expected_type}, 実際型 {type(value)}")
            
            return True
        else:
            print("❌ MLConfig型互換性問題:")
            for issue in compatibility_issues:
                print(f"  - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ 型互換性テストエラー: {e}")
        return False

def test_error_handling_compatibility():
    """エラーハンドリング互換性テスト"""
    print("\n=== エラーハンドリング互換性テスト ===")
    
    test_cases = [
        {
            "name": "無効なJSON",
            "method": "PUT",
            "url": "/api/ml/config",
            "data": "invalid json",
            "headers": {"Content-Type": "application/json"},
            "expected_status": 422
        },
        {
            "name": "無効な設定値",
            "method": "PUT", 
            "url": "/api/ml/config",
            "data": {
                "prediction": {
                    "default_up_prob": 1.5  # 無効値（1.0超過）
                }
            },
            "expected_status": 200  # バリデーションエラーはAPIレスポンスで返される
        },
        {
            "name": "存在しないエンドポイント",
            "method": "GET",
            "url": "/api/ml/nonexistent",
            "expected_status": 404
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        try:
            print(f"\n--- {test_case['name']} ---")
            
            if test_case["method"] == "GET":
                response = requests.get(f"{BACKEND_URL}{test_case['url']}")
            elif test_case["method"] == "PUT":
                if isinstance(test_case["data"], str):
                    response = requests.put(
                        f"{BACKEND_URL}{test_case['url']}",
                        data=test_case["data"],
                        headers=test_case.get("headers", {})
                    )
                else:
                    response = requests.put(
                        f"{BACKEND_URL}{test_case['url']}",
                        json=test_case["data"],
                        headers={"Content-Type": "application/json"}
                    )
            
            print(f"ステータスコード: {response.status_code}")
            
            if response.status_code == test_case["expected_status"]:
                print(f"✅ 期待通りのステータスコード")
                
                # バリデーションエラーの場合、レスポンス内容も確認
                if test_case["name"] == "無効な設定値" and response.status_code == 200:
                    data = response.json()
                    if not data.get("success"):
                        print(f"  ✅ バリデーションエラーが適切に処理されました")
                        print(f"  エラーメッセージ: {data.get('message')}")
                        results[test_case["name"]] = True
                    else:
                        print(f"  ❌ 無効な値が受け入れられました")
                        results[test_case["name"]] = False
                else:
                    results[test_case["name"]] = True
            else:
                print(f"❌ 期待ステータス: {test_case['expected_status']}, 実際: {response.status_code}")
                results[test_case["name"]] = False
                
        except Exception as e:
            print(f"❌ {test_case['name']} エラー: {e}")
            results[test_case["name"]] = False
    
    return results

def test_real_world_scenario():
    """実際のユースケースシナリオテスト"""
    print("\n=== 実際のユースケースシナリオテスト ===")
    
    try:
        # シナリオ: フロントエンドでML設定を変更する流れ
        print("シナリオ: フロントエンドでML設定変更")
        
        # 1. 初期設定取得（useMLSettings.fetchConfig）
        print("\n1. 初期設定取得...")
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        if response.status_code != 200:
            print("❌ 初期設定取得失敗")
            return False
        
        initial_config = response.json()
        print(f"✅ 初期設定取得成功")
        print(f"  初期値: default_up_prob = {initial_config['prediction']['default_up_prob']}")
        
        # 2. フロントエンドでの設定変更をシミュレート（updateConfig）
        print("\n2. フロントエンドでの設定変更をシミュレート...")
        new_config = initial_config.copy()
        new_config["prediction"]["default_up_prob"] = 0.4
        new_config["prediction"]["default_down_prob"] = 0.3
        new_config["prediction"]["default_range_prob"] = 0.3
        print(f"  変更後: default_up_prob = {new_config['prediction']['default_up_prob']}")
        
        # 3. 設定保存（useMLSettings.saveConfig）
        print("\n3. 設定保存...")
        response = requests.put(
            f"{BACKEND_URL}/api/ml/config",
            json=new_config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print("❌ 設定保存失敗")
            return False
        
        save_result = response.json()
        if not save_result.get("success"):
            print(f"❌ 設定保存失敗: {save_result.get('message')}")
            return False
        
        print("✅ 設定保存成功")
        
        # 4. 設定変更確認
        print("\n4. 設定変更確認...")
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        updated_config = response.json()
        
        if updated_config["prediction"]["default_up_prob"] == 0.4:
            print("✅ 設定変更が正しく反映されました")
        else:
            print("❌ 設定変更が反映されていません")
            return False
        
        # 5. 設定リセット（useMLSettings.resetToDefaults）
        print("\n5. 設定リセット...")
        response = requests.post(f"{BACKEND_URL}/api/ml/config/reset")
        
        if response.status_code != 200:
            print("❌ 設定リセット失敗")
            return False
        
        reset_result = response.json()
        if not reset_result.get("success"):
            print(f"❌ 設定リセット失敗: {reset_result.get('message')}")
            return False
        
        print("✅ 設定リセット成功")
        
        # 6. リセット確認
        print("\n6. リセット確認...")
        response = requests.get(f"{BACKEND_URL}/api/ml/config")
        final_config = response.json()
        
        if final_config["prediction"]["default_up_prob"] == initial_config["prediction"]["default_up_prob"]:
            print("✅ 設定が正しくリセットされました")
            print("🎉 実際のユースケースシナリオテスト完全成功！")
            return True
        else:
            print("❌ 設定リセットが正しく動作していません")
            return False
        
    except Exception as e:
        print(f"❌ シナリオテストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🚀 フロントエンドフック・バックエンドAPI連携テスト開始")
    print("=" * 70)
    
    test_results = {}
    
    # 1. フック・API互換性テスト
    hook_api_results = test_hook_api_compatibility()
    test_results["hook_api_compatibility"] = all(hook_api_results.values())
    
    # 2. MLConfig型互換性テスト
    test_results["mlconfig_type_compatibility"] = test_ml_config_type_compatibility()
    
    # 3. エラーハンドリング互換性テスト
    error_handling_results = test_error_handling_compatibility()
    test_results["error_handling_compatibility"] = all(error_handling_results.values())
    
    # 4. 実際のユースケースシナリオテスト
    test_results["real_world_scenario"] = test_real_world_scenario()
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 フロントエンドフック連携テスト結果サマリー")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n合計: {passed_tests}/{total_tests} テスト成功")
    
    if passed_tests == total_tests:
        print("🎉 フロントエンドフックとバックエンドAPIが完全に連携しています！")
        print("useMLSettingsフックが正常に動作し、ML設定管理が完璧に機能しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
        print("フロントエンドフックとバックエンドAPIの連携に問題があります。")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    main()
