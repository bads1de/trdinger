import requests

print("=== フロントエンドAPIルートテスト ===")

# バックエンドAPIを直接テスト
try:
    response = requests.get("http://localhost:8000/api/external-market/latest?limit=5")
    print(f"バックエンドAPI Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"データ件数: {len(data['data'])}")
        if data["data"]:
            sample = data["data"][0]
            print(f"サンプル: {sample['symbol']} - {sample['close']}")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")
    print("バックエンドサーバーが起動していない可能性があります")
