#!/usr/bin/env python3
"""
FastAPI TestClient を使用したテスト
"""

from app.main import app
from fastapi.testclient import TestClient

def test_with_client():
    client = TestClient(app)
    
    # ヘルスチェック
    response = client.get('/health')
    print(f'Health check: {response.status_code}')
    
    # 自動戦略API
    response = client.get('/api/auto-strategy/config/default')
    print(f'Auto strategy API: {response.status_code}')
    if response.status_code == 200:
        print('✅ API working!')
        data = response.json()
        print(f'Success: {data.get("success", False)}')
        config = data.get("config", {})
        print(f'Population size: {config.get("population_size", "N/A")}')
    else:
        print(f'Error: {response.text}')
        
    # プリセット設定
    response = client.get('/api/auto-strategy/config/presets')
    print(f'Presets API: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Presets success: {data.get("success", False)}')
        presets = data.get("presets", {})
        print(f'Available presets: {list(presets.keys())}')

if __name__ == "__main__":
    test_with_client()
