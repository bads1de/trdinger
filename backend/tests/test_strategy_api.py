#!/usr/bin/env python3
"""
Strategy Builder API テストスクリプト

フロントエンドから送信される形式のデータでバックエンドAPIをテストします。
"""

import requests
import json

# バックエンドAPIのURL
BACKEND_URL = "http://localhost:8000"

def test_strategy_save():
    """戦略保存APIのテスト"""
    
    # フロントエンドから送信される形式のテストデータ
    test_data = {
        "name": "テスト戦略_修正版",
        "description": "フロントエンドCondition構造対応テスト",
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
                },
                {
                    "type": "RSI",
                    "parameters": {"period": 14},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "RSI",
                        "parameters": {"period": 14}
                    }
                }
            ],
            "entry_conditions": [
                {
                    "id": "condition_1",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": "<",
                    "value": 30,
                    "logicalOperator": "AND"
                }
            ],
            "exit_conditions": [
                {
                    "id": "condition_3",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": ">",
                    "value": 70,
                    "logicalOperator": "OR"
                }
            ],
            "risk_management": {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
                "position_sizing": "fixed"
            },
            "metadata": {
                "created_by": "strategy_builder",
                "version": "1.0",
                "created_at": "2025-06-28T11:57:00.000Z"
            }
        }
    }
    
    print("=== Strategy Builder Save API テスト ===")
    print(f"テストデータ:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    print()
    
    try:
        # APIリクエスト送信
        response = requests.post(
            f"{BACKEND_URL}/api/strategy-builder/save",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"ステータスコード: {response.status_code}")
        print(f"レスポンス:")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("✅ 戦略保存成功!")
        else:
            print(f"❌ エラー: {response.status_code}")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print("❌ バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

def test_strategy_validate():
    """戦略検証APIのテスト"""
    
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
    
    print("\n=== Strategy Builder Validate API テスト ===")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/strategy-builder/validate",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("✅ 戦略検証成功!")
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
    # 検証APIテスト
    test_strategy_validate()
    
    # 保存APIテスト
    test_strategy_save()
