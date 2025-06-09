#!/usr/bin/env python3
"""
クイックAPIテスト
"""

import requests
import json

def test_api():
    try:
        # デフォルト設定テスト
        response = requests.get('http://localhost:8000/api/auto-strategy/config/default', timeout=10)
        print(f'Default config: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'Success: {data.get("success", False)}')
            config = data.get("config", {})
            print(f'Config keys: {list(config.keys())}')
            print(f'Population size: {config.get("population_size", "N/A")}')
        else:
            print(f'Error: {response.text}')
            
        # プリセット設定テスト
        response = requests.get('http://localhost:8000/api/auto-strategy/config/presets', timeout=10)
        print(f'\nPresets: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'Success: {data.get("success", False)}')
            presets = data.get('presets', {})
            print(f'Presets: {list(presets.keys())}')
        else:
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'Exception: {e}')

if __name__ == "__main__":
    test_api()
