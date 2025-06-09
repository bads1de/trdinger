#!/usr/bin/env python3
"""
サーバー起動テストスクリプト
"""

import sys
import traceback
import logging

# ログレベルを設定
logging.basicConfig(level=logging.DEBUG)

def test_server_startup():
    """サーバー起動をテスト"""
    
    print("=== サーバー起動テスト開始 ===")
    
    try:
        # アプリケーションのインポート
        print("1. アプリケーションをインポート中...")
        from app.main import app
        print("✅ アプリケーションインポート成功")
        
        # ルートの確認
        print("2. ルートを確認中...")
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(f"{route.methods if hasattr(route, 'methods') else 'N/A'} {route.path}")
        
        print(f"✅ 登録されたルート数: {len(routes)}")
        
        # auto-strategy関連のルートを確認
        auto_strategy_routes = [r for r in routes if 'auto-strategy' in r]
        print(f"✅ auto-strategy関連ルート数: {len(auto_strategy_routes)}")
        
        for route in auto_strategy_routes:
            print(f"   - {route}")
        
        # テストクライアントでの動作確認
        print("3. テストクライアントで動作確認中...")
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # ヘルスチェック
        response = client.get("/health")
        print(f"✅ ヘルスチェック: {response.status_code}")
        
        # デフォルト設定取得
        response = client.get("/api/auto-strategy/config/default")
        print(f"✅ デフォルト設定: {response.status_code}")
        if response.status_code == 200:
            print(f"   レスポンス: {response.json()}")
        else:
            print(f"   エラー: {response.text}")
        
        # プリセット設定取得
        response = client.get("/api/auto-strategy/config/presets")
        print(f"✅ プリセット設定: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   プリセット数: {len(data.get('presets', {}))}")
        else:
            print(f"   エラー: {response.text}")
        
        print("=== サーバー起動テスト完了 ===")
        return True
        
    except Exception as e:
        print(f"❌ サーバー起動テストエラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_server_startup()
    if not success:
        sys.exit(1)
    print("✅ サーバー起動テスト成功")
