import requests
import json

print("=== バックエンドAPIレスポンス詳細確認（外部市場データ含む） ===")

try:
    response = requests.post("http://localhost:8000/api/data-collection/bulk-incremental-update?symbol=BTC/USDT:USDT&timeframe=1h")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nレスポンス構造:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # 外部市場データの確認
        em_data = data.get("data", {}).get("data", {}).get("external_market")
        if em_data:
            print("\n外部市場データ結果:")
            print(f"  取得件数: {em_data.get('fetched_count', 0)}")
            print(f"  挿入件数: {em_data.get('inserted_count', 0)}")
            print(f"  成功: {em_data.get('success', False)}")
        else:
            print("\n外部市場データ: 結果なし")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")
