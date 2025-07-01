#!/usr/bin/env python3
"""
Validate API直接テストスクリプト
"""

import requests
import json

# バックエンドAPIのURL
BACKEND_URL = "http://localhost:8000"

def test_validate_direct():
    """validateエンドポイントを直接テスト"""
    
    # 検証用のテストデータ
    test_data = {
        "strategy_config": {
            "indicators": [
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "id": "condition_1",
                    "type": "threshold",
                    "indicator1": "SMA",
                    "operator": ">",
                    "value": 100
                }
            ],
            "exit_conditions": [
                {
                    "id": "condition_2",
                    "type": "threshold",
                    "indicator1": "SMA",
                    "operator": "<",
                    "value": 90
                }
            ]
        }
    }
    
    print("=== Validate API 直接テスト ===")
    print(f"テストデータ:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    print()
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/strategy-builder/validate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"ステータスコード: {response.status_code}")
        print(f"レスポンス:")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("✅ 検証成功!")
        else:
            print(f"❌ エラー: {response.status_code}")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print("❌ バックエンドサーバーに接続できません。")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    test_validate_direct()
