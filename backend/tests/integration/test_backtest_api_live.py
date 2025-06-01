#!/usr/bin/env python3
"""
バックテストAPIの動作テスト（統合テスト）

実際のAPIサーバーを起動してエンドポイントをテストします。
"""

import requests
import json
from datetime import datetime

# APIベースURL
BASE_URL = "http://localhost:8000"

def test_strategies_endpoint():
    """戦略一覧エンドポイントのテスト"""
    print("=== 戦略一覧取得テスト ===")
    response = requests.get(f"{BASE_URL}/api/backtest/strategies")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_health_endpoint():
    """ヘルスチェックエンドポイントのテスト"""
    print("\n=== ヘルスチェックテスト ===")
    response = requests.get(f"{BASE_URL}/api/backtest/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200

def test_backtest_run():
    """バックテスト実行テスト"""
    print("\n=== バックテスト実行テスト ===")
    
    # テスト用のバックテスト設定
    config = {
        "strategy_name": "SMA_CROSS",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-31T23:59:59Z",
        "initial_capital": 100000.0,
        "commission_rate": 0.001,
        "strategy_config": {
            "strategy_type": "SMA_CROSS",
            "parameters": {
                "n1": 20,
                "n2": 50
            }
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/backtest/run",
            json=config,
            timeout=60  # 60秒のタイムアウト
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("バックテスト成功!")
            print(f"戦略名: {result['result']['strategy_name']}")
            print(f"取引ペア: {result['result']['symbol']}")
            print(f"初期資金: {result['result']['initial_capital']}")
            
            metrics = result['result']['performance_metrics']
            print(f"総リターン: {metrics['total_return']:.2f}%")
            print(f"シャープレシオ: {metrics['sharpe_ratio']:.2f}")
            print(f"最大ドローダウン: {metrics['max_drawdown']:.2f}%")
            print(f"勝率: {metrics['win_rate']:.2f}%")
            print(f"総取引数: {metrics['total_trades']}")
            
            return True
        else:
            print(f"エラー: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("タイムアウトエラー: バックテストの実行に時間がかかりすぎています")
        return False
    except Exception as e:
        print(f"エラー: {e}")
        return False

def test_results_endpoint():
    """バックテスト結果一覧取得テスト"""
    print("\n=== バックテスト結果一覧取得テスト ===")
    response = requests.get(f"{BASE_URL}/api/backtest/results?limit=5")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"総件数: {result.get('total', 0)}")
        print(f"取得件数: {len(result.get('results', []))}")
        
        for i, backtest in enumerate(result.get('results', [])[:3]):
            print(f"\n結果 {i+1}:")
            print(f"  ID: {backtest['id']}")
            print(f"  戦略: {backtest['strategy_name']}")
            print(f"  シンボル: {backtest['symbol']}")
            print(f"  作成日時: {backtest['created_at']}")
        
        return True
    else:
        print(f"エラー: {response.text}")
        return False

def main():
    """メイン関数"""
    print("バックテストAPI動作テスト開始")
    print("=" * 50)
    
    tests = [
        ("ヘルスチェック", test_health_endpoint),
        ("戦略一覧取得", test_strategies_endpoint),
        ("バックテスト結果一覧", test_results_endpoint),
        ("バックテスト実行", test_backtest_run),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"{test_name}でエラー: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー:")
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    success_count = sum(1 for _, success in results if success)
    print(f"\n成功: {success_count}/{len(results)}")

if __name__ == "__main__":
    main()
