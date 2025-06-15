import requests
import json

print("=== 統合API動作確認 ===")

# バックエンドAPIの直接テスト
backend_url = "http://127.0.0.1:8000/api/strategies/unified"

try:
    print("1. バックエンドAPI直接呼び出し:")
    response = requests.get(backend_url, timeout=10)
    print(f"   ステータス: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f'   成功: {data.get("success", False)}')
        print(f'   戦略数: {len(data.get("strategies", []))}')
        print(f'   総数: {data.get("total_count", 0)}')

        # 戦略の詳細を表示
        strategies = data.get("strategies", [])
        if strategies:
            print("   戦略詳細:")
            for i, strategy in enumerate(strategies[:2]):
                print(f'     [{i+1}] ID: {strategy.get("id", "不明")}')
                print(f'         名前: {strategy.get("name", "不明")}')
                print(f'         ソース: {strategy.get("source", "不明")}')
                print(
                    f'         期待リターン: {strategy.get("expected_return", 0):.3f}'
                )
                if strategy.get("experiment_id"):
                    print(f'         実験ID: {strategy.get("experiment_id")}')
                    print(f'         フィットネス: {strategy.get("fitness_score")}')
    else:
        print(f"   エラー: {response.text}")

except requests.exceptions.ConnectionError:
    print("   エラー: バックエンドサーバーに接続できません")
except Exception as e:
    print(f"   エラー: {e}")

print()

# フロントエンドAPI経由のテスト
frontend_url = "http://localhost:3000/api/strategies/unified"

try:
    print("2. フロントエンドAPI経由呼び出し:")
    response = requests.get(frontend_url, timeout=10)
    print(f"   ステータス: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f'   成功: {data.get("success", False)}')
        print(f'   戦略数: {len(data.get("strategies", []))}')
        print(f'   総数: {data.get("total_count", 0)}')
        print(f'   メッセージ: {data.get("message", "なし")}')
    else:
        print(f"   エラー: {response.text}")

except requests.exceptions.ConnectionError:
    print("   エラー: フロントエンドサーバーに接続できません")
except Exception as e:
    print(f"   エラー: {e}")

print()

# 改善されたフィルタリング機能のテスト
try:
    print("3. 改善されたフィルタリング機能テスト:")

    # フィットネススコア順でソート
    params = {"sort_by": "fitness_score", "sort_order": "desc", "limit": 5}
    response = requests.get(backend_url, params=params, timeout=10)
    print(f"   フィットネススコア順ソート: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        strategies = data.get("strategies", [])
        print(f"   取得戦略数: {len(strategies)}")
        for i, strategy in enumerate(strategies):
            print(
                f'     [{i+1}] {strategy.get("name", "不明")} - フィットネス: {strategy.get("fitness_score", "N/A")}'
            )

    print()

    # 実験IDフィルター
    params = {"experiment_id": 2, "limit": 10}
    response = requests.get(backend_url, params=params, timeout=10)
    print(f"   実験ID=2フィルター: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        strategies = data.get("strategies", [])
        print(f"   実験ID=2の戦略数: {len(strategies)}")
        for strategy in strategies:
            print(
                f'     {strategy.get("name", "不明")} (実験ID: {strategy.get("experiment_id")})'
            )

    print()

    # 最小フィットネススコアフィルター
    params = {"min_fitness": 0.8, "limit": 10}
    response = requests.get(backend_url, params=params, timeout=10)
    print(f"   最小フィットネス=0.8フィルター: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        strategies = data.get("strategies", [])
        print(f"   フィットネス≥0.8の戦略数: {len(strategies)}")
        for strategy in strategies:
            print(
                f'     {strategy.get("name", "不明")} - フィットネス: {strategy.get("fitness_score", "N/A")}'
            )

except Exception as e:
    print(f"   エラー: {e}")

print("=== API動作確認完了 ===")
