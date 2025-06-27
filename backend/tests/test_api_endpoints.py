#!/usr/bin/env python3
"""
APIエンドポイントのテストスクリプト
"""

import requests
import json

BASE_URL = "http://localhost:8001"

def test_indicators_endpoint():
    """指標一覧取得エンドポイントのテスト"""
    print("=== 指標一覧取得エンドポイントのテスト ===")
    try:
        response = requests.get(f"{BASE_URL}/api/strategy-builder/indicators")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                categories = data["data"]["categories"]
                print(f"✓ 指標取得成功: {len(categories)}カテゴリ")
                for category, indicators in categories.items():
                    print(f"  {category}: {len(indicators)}個")
                return True
            else:
                print(f"✗ APIエラー: {data.get('message')}")
                return False
        else:
            print(f"✗ HTTPエラー: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ リクエストエラー: {e}")
        return False

def test_save_strategy_endpoint():
    """戦略保存エンドポイントのテスト"""
    print("\n=== 戦略保存エンドポイントのテスト ===")
    
    strategy_data = {
        "name": "APIテスト戦略",
        "description": "APIエンドポイント統合テスト用の戦略",
        "strategy_config": {
            "indicators": [
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "SMA",
                        "parameters": {"period": 20}
                    }
                }
            ],
            "entry_conditions": [
                {
                    "type": "threshold",
                    "indicator": "SMA",
                    "operator": ">",
                    "value": 100
                }
            ],
            "exit_conditions": [
                {
                    "type": "threshold",
                    "indicator": "SMA",
                    "operator": "<",
                    "value": 95
                }
            ]
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/strategy-builder/save",
            json=strategy_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                strategy = data["data"]
                print(f"✓ 戦略保存成功: ID={strategy['id']}")
                print(f"  名前: {strategy['name']}")
                return strategy["id"]
            else:
                print(f"✗ APIエラー: {data.get('message')}")
                return None
        else:
            print(f"✗ HTTPエラー: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ リクエストエラー: {e}")
        return None

def test_get_strategies_endpoint():
    """戦略一覧取得エンドポイントのテスト"""
    print("\n=== 戦略一覧取得エンドポイントのテスト ===")
    try:
        response = requests.get(f"{BASE_URL}/api/strategy-builder/strategies")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                strategies = data["data"]["strategies"]
                count = data["data"]["count"]
                print(f"✓ 戦略一覧取得成功: {count}件")
                for strategy in strategies:
                    print(f"  - ID={strategy['id']}, 名前={strategy['name']}")
                return strategies
            else:
                print(f"✗ APIエラー: {data.get('message')}")
                return None
        else:
            print(f"✗ HTTPエラー: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ リクエストエラー: {e}")
        return None

def test_validate_strategy_endpoint():
    """戦略検証エンドポイントのテスト"""
    print("\n=== 戦略検証エンドポイントのテスト ===")
    
    strategy_config = {
        "indicators": [
            {
                "type": "RSI",
                "parameters": {"period": 14},
                "enabled": True
            }
        ],
        "entry_conditions": [
            {
                "type": "threshold",
                "indicator": "RSI",
                "operator": "<",
                "value": 30
            }
        ],
        "exit_conditions": [
            {
                "type": "threshold",
                "indicator": "RSI",
                "operator": ">",
                "value": 70
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/strategy-builder/validate",
            json={"strategy_config": strategy_config},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                validation = data["data"]
                is_valid = validation["is_valid"]
                errors = validation["errors"]
                print(f"✓ 戦略検証成功: valid={is_valid}")
                if errors:
                    print(f"  エラー: {errors}")
                return is_valid
            else:
                print(f"✗ APIエラー: {data.get('message')}")
                return False
        else:
            print(f"✗ HTTPエラー: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ リクエストエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("ストラテジービルダーAPIエンドポイントの統合テストを開始します\n")
    
    results = []
    
    # 指標一覧取得テスト
    results.append(test_indicators_endpoint())
    
    # 戦略検証テスト
    results.append(test_validate_strategy_endpoint())
    
    # 戦略保存テスト
    strategy_id = test_save_strategy_endpoint()
    results.append(strategy_id is not None)
    
    # 戦略一覧取得テスト
    strategies = test_get_strategies_endpoint()
    results.append(strategies is not None)
    
    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)
    
    test_names = [
        "指標一覧取得",
        "戦略検証",
        "戦略保存",
        "戦略一覧取得"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{i+1}. {name}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 すべてのテストが成功しました！")
        return True
    else:
        print("⚠️ 一部のテストが失敗しました")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
