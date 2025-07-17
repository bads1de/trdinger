import requests

print("=== バックエンドAPIレスポンス構造確認 ===")

# 最新データAPI
try:
    response = requests.get("http://localhost:8000/api/external-market/latest?limit=3")
    print(f"最新データAPI Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("レスポンス構造:")
        print(f"  success: {data.get('success')}")
        print(f"  message: {data.get('message')}")
        print(f"  data type: {type(data.get('data'))}")
        print(f"  data length: {len(data.get('data', []))}")
        if data.get("data"):
            print(f"  sample data: {data['data'][0]}")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")

# 状態API
try:
    response = requests.get("http://localhost:8000/api/external-market/status")
    print(f"\n状態API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("レスポンス構造:")
        print(f"  success: {data.get('success')}")
        statistics = data.get("data", {}).get("statistics", {})
        print(f"  data.statistics.count: {statistics.get('count')}")
        print(f"  data.statistics.symbols: {statistics.get('symbols')}")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")
