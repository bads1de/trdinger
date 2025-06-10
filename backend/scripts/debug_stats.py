import requests
import json

# 統計情報を取得
stats_url = "http://127.0.0.1:8000/api/strategies/showcase/stats"
stats_response = requests.get(stats_url)

print(f"Status Code: {stats_response.status_code}")
print(f"Response: {json.dumps(stats_response.json(), indent=2, ensure_ascii=False)}")

# 戦略一覧も確認
list_url = "http://127.0.0.1:8000/api/strategies/showcase?limit=5"
list_response = requests.get(list_url)

print(f"\n戦略一覧 Status Code: {list_response.status_code}")
print(f"戦略数: {list_response.json()['total_count']}")
