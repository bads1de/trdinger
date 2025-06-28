#!/usr/bin/env python3
"""
動作するケースのテスト

修正したコードが正しく動作するかを確認します。
"""

import requests
import json

# バックエンドAPIのURL
BACKEND_URL = "http://localhost:8000"


def test_working_case():
    """動作するケースのテスト"""

    # 動作することが確認されているテストデータ
    test_data = {
        "name": "動作テスト戦略",
        "description": "修正後の動作確認",
        "strategy_config": {
            "indicators": [
                {
                    "type": "RSI",
                    "parameters": {"period": 14},
                    "enabled": True,
                    "json_config": {
                        "indicator_name": "RSI",
                        "parameters": {"period": 14},
                    },
                }
            ],
            "entry_conditions": [
                {
                    "id": "condition_1",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": "<",
                    "value": 30,
                    "logicalOperator": "AND",
                }
            ],
            "exit_conditions": [
                {
                    "id": "condition_2",
                    "type": "threshold",
                    "indicator1": "RSI",
                    "operator": ">",
                    "value": 70,
                    "logicalOperator": "OR",
                }
            ],
            "risk_management": {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
                "position_sizing": "fixed",
            },
            "metadata": {
                "created_by": "strategy_builder",
                "version": "1.0",
                "created_at": "2025-06-28T11:57:00.000Z",
            },
        },
    }

    print("=== 動作テスト ===")

    # 1. まず検証APIをテスト
    print("1. 検証APIテスト")
    validate_data = {"strategy_config": test_data["strategy_config"]}

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/strategy-builder/validate",
            json=validate_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"検証ステータスコード: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"検証結果: {result['data']['is_valid']}")
            if result["data"]["errors"]:
                print(f"エラー: {result['data']['errors']}")
        else:
            print(f"検証エラー: {response.status_code}")
            return

    except Exception as e:
        print(f"検証API呼び出しエラー: {e}")
        return

    print()

    # 2. 保存APIをテスト
    print("2. 保存APIテスト")

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/strategy-builder/save",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"保存ステータスコード: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ 戦略保存成功!")
            print(f"保存された戦略ID: {result['data']['id']}")
            print(f"戦略名: {result['data']['name']}")
        else:
            print(f"❌ 保存エラー: {response.status_code}")
            try:
                error_data = response.json()
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print(response.text)

    except Exception as e:
        print(f"保存API呼び出しエラー: {e}")


if __name__ == "__main__":
    test_working_case()
