import requests

print("=== 一括差分更新テスト（外部市場データ含む） ===")

try:
    response = requests.post("http://localhost:8000/api/data-collection/bulk-incremental-update?symbol=BTC/USDT:USDT&timeframe=1h")
    print(f"一括差分更新API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"成功: {data.get('success')}")
        print(f"メッセージ: {data.get('message')}")
        result = data.get("data", {})
        print(f"総保存件数: {result.get('total_saved_count', 0)}")
        
        # 各データタイプの結果
        data_results = result.get("data", {})
        print("\nデータ別結果:")
        print(f"  OHLCV: {data_results.get('ohlcv', {}).get('saved_count', 0)}件")
        print(f"  ファンディングレート: {data_results.get('funding_rate', {}).get('saved_count', 0)}件")
        print(f"  オープンインタレスト: {data_results.get('open_interest', {}).get('saved_count', 0)}件")
        
        # 外部市場データの結果
        em_result = data_results.get("external_market", {})
        if em_result:
            print(f"  外部市場データ: {em_result.get('inserted_count', 0)}件 (取得: {em_result.get('fetched_count', 0)}件)")
            print(f"    成功: {em_result.get('success', False)}")
            print(f"    収集タイプ: {em_result.get('collection_type', '不明')}")
        else:
            print("  外部市場データ: 結果なし")
    else:
        print(f"エラー: {response.text}")
except Exception as e:
    print(f"接続エラー: {e}")
