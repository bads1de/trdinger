#!/usr/bin/env python3
"""
OI履歴データ収集用 HTTP リクエストスクリプト

APIサーバー経由でOIデータを収集します。
"""

import requests

BASE_URL = "http://localhost:8000"


def collect_oi_data():
    """OI履歴データを収集"""
    print("=" * 60)
    print("OI履歴データ収集スクリプト（API経由）")
    print("=" * 60)

    # サーバーが起動しているか確認
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            print("❌ APIサーバーが起動していません")
            print("サーバーを起動してください:")
            print("  cd backend")
            print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return False
    except Exception as e:
        print(f"❌ API接続エラー: {e}")
        print("\nサーバーを起動してください:")
        print("  cd backend")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False

    print("\n✅ APIサーバー接続確認完了")
    print("\nOI履歴データ収集（全期間再取得）を実行中...")
    print("※ 既存のOIデータは削除されます")

    # OI履歴データ収集（全期間）
    try:
        response = requests.post(
            f"{BASE_URL}/api/data-collection/historical-oi",
            params={"symbol": "BTC/USDT:USDT", "interval": "1h"},
            timeout=300,
        )

        if response.status_code == 200:
            result = response.json()
            print("\n✅ 収集リクエスト成功:")
            print(f"  {result}")
            print("\nバックグラウンドで収集中です。サーバーログを確認してください。")
        else:
            print(f"\n❌ 収集リクエスト失敗: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ リクエストエラー: {e}")
        return False

    print("\n次のステップ:")
    print("  1. サーバーログで進捗確認")
    print("  2. ML評価: python -m scripts.ml_optimization.run_ml_pipeline")

    return True


if __name__ == "__main__":
    success = collect_oi_data()
    exit(0 if success else 1)
