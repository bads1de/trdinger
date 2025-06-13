#!/usr/bin/env python3
"""
利用可能なAPIエンドポイントを確認するスクリプト
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def check_endpoints():
    """利用可能なエンドポイントを確認"""
    print("🔍 APIエンドポイント確認")
    print("=" * 50)
    
    # 1. ヘルスチェック
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"Health Check Error: {e}")
    
    # 2. OpenAPI docs
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"Docs: {response.status_code}")
    except Exception as e:
        print(f"Docs Error: {e}")
    
    # 3. OpenAPI JSON
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        print(f"OpenAPI JSON: {response.status_code}")
        if response.status_code == 200:
            openapi_data = response.json()
            print("\n📋 利用可能なエンドポイント:")
            paths = openapi_data.get("paths", {})
            for path, methods in paths.items():
                for method in methods.keys():
                    print(f"  {method.upper()} {path}")
    except Exception as e:
        print(f"OpenAPI JSON Error: {e}")
    
    # 4. 自動戦略関連エンドポイントの直接テスト
    auto_strategy_endpoints = [
        "/api/auto-strategy/generate",
        "/api/auto-strategy/config/default",
        "/api/auto-strategy/config/presets",
        "/api/auto-strategy/experiments",
    ]
    
    print("\n🧬 自動戦略エンドポイント確認:")
    for endpoint in auto_strategy_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            print(f"  GET {endpoint}: {response.status_code}")
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"    Response: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}...")
                except:
                    print(f"    Response: {response.text[:100]}...")
        except Exception as e:
            print(f"  GET {endpoint}: Error - {e}")

if __name__ == "__main__":
    check_endpoints()
