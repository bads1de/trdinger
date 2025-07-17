import requests

print("=== 履歴データ収集テスト ===")

try:
    response = requests.post("http://localhost:8000/api/external-market/collect-historical")
    print(f"履歴データ収集API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data.get('success')}")
        print(f"メッセージ: {data.get('message')}")
        result = data.get("data", {})
        print(f"取得件数: {result.get('fetched_count', 0)}")
        print(f"挿入件数: {result.get('inserted_count', 0)}")
        print(f"収集タイプ: {result.get('collection_type')}")
        print(f"開始日: {result.get('start_date')}")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")
