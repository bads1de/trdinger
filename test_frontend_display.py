"""
フロントエンド表示の動作確認スクリプト
"""

import requests
import json
import time

# API設定
FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://127.0.0.1:8000"

def test_frontend_api():
    """フロントエンドAPIの動作確認"""
    print("=== フロントエンドAPI動作確認 ===\n")
    
    try:
        # 1. 基本的な戦略取得
        print("1. 基本的な戦略取得テスト:")
        response = requests.get(f"{FRONTEND_URL}/api/strategies/unified", timeout=10)
        print(f"   ステータス: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   成功: {data.get('success')}")
            print(f"   戦略数: {len(data.get('strategies', []))}")
            print(f"   総数: {data.get('total_count')}")
            
            # 戦略詳細を表示
            strategies = data.get('strategies', [])
            for i, strategy in enumerate(strategies):
                print(f"   [{i+1}] {strategy.get('name', '不明')}")
                print(f"       ID: {strategy.get('id')}")
                print(f"       カテゴリ: {strategy.get('category')}")
                print(f"       期待リターン: {strategy.get('expected_return', 0):.3f}")
                print(f"       シャープレシオ: {strategy.get('sharpe_ratio', 0):.3f}")
                print(f"       フィットネス: {strategy.get('fitness_score', 'N/A')}")
                print(f"       実験ID: {strategy.get('experiment_id', 'N/A')}")
                print()
        else:
            print(f"   エラー: {response.text}")
            
    except Exception as e:
        print(f"   例外: {e}")
    
    print()
    
    # 2. フィルタリング機能テスト
    print("2. フィルタリング機能テスト:")
    
    test_filters = [
        {"experiment_id": 2, "name": "実験ID=2フィルター"},
        {"min_fitness": 0.8, "name": "最小フィットネス=0.8フィルター"},
        {"sort_by": "fitness_score", "sort_order": "desc", "name": "フィットネススコア降順ソート"},
        {"sort_by": "expected_return", "sort_order": "desc", "name": "期待リターン降順ソート"},
    ]
    
    for filter_config in test_filters:
        try:
            filter_name = filter_config.pop("name")
            print(f"   {filter_name}:")
            
            response = requests.get(
                f"{FRONTEND_URL}/api/strategies/unified", 
                params=filter_config, 
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                strategies = data.get('strategies', [])
                print(f"     取得戦略数: {len(strategies)}")
                
                for strategy in strategies:
                    print(f"     - {strategy.get('name', '不明')} "
                          f"(フィットネス: {strategy.get('fitness_score', 'N/A')}, "
                          f"実験ID: {strategy.get('experiment_id', 'N/A')})")
            else:
                print(f"     エラー: {response.status_code}")
                
        except Exception as e:
            print(f"     例外: {e}")
        
        print()

def test_backend_direct():
    """バックエンド直接呼び出しテスト"""
    print("3. バックエンド直接呼び出しテスト:")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/strategies/unified", timeout=10)
        print(f"   ステータス: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   成功: {data.get('success')}")
            print(f"   戦略数: {len(data.get('strategies', []))}")
            
            # 統計情報
            strategies = data.get('strategies', [])
            if strategies:
                fitness_scores = [s.get('fitness_score', 0) for s in strategies if s.get('fitness_score')]
                if fitness_scores:
                    print(f"   フィットネススコア範囲: {min(fitness_scores):.3f} - {max(fitness_scores):.3f}")
                
                experiment_ids = list(set(s.get('experiment_id') for s in strategies if s.get('experiment_id')))
                print(f"   実験ID: {experiment_ids}")
        else:
            print(f"   エラー: {response.text}")
            
    except Exception as e:
        print(f"   例外: {e}")
    
    print()

def test_strategy_stats():
    """戦略統計情報テスト"""
    print("4. 戦略統計情報テスト:")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/strategies/stats", timeout=10)
        print(f"   ステータス: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            
            print(f"   総戦略数: {stats.get('total_strategies', 0)}")
            print(f"   ショーケース戦略数: {stats.get('showcase_strategies', 0)}")
            print(f"   オートストラテジー数: {stats.get('auto_generated_strategies', 0)}")
            
            performance = stats.get('performance_summary', {})
            print(f"   平均リターン: {performance.get('avg_return', 0):.3f}")
            print(f"   平均シャープレシオ: {performance.get('avg_sharpe_ratio', 0):.3f}")
            print(f"   平均最大ドローダウン: {performance.get('avg_max_drawdown', 0):.3f}")
            print(f"   平均勝率: {performance.get('avg_win_rate', 0):.3f}")
            
        else:
            print(f"   エラー: {response.text}")
            
    except Exception as e:
        print(f"   例外: {e}")

def check_server_status():
    """サーバー状態確認"""
    print("=== サーバー状態確認 ===\n")
    
    # フロントエンドサーバー確認
    try:
        response = requests.get(f"{FRONTEND_URL}/api/strategies/unified", timeout=5)
        print(f"フロントエンドサーバー: OK (ステータス: {response.status_code})")
    except Exception as e:
        print(f"フロントエンドサーバー: NG ({e})")
    
    # バックエンドサーバー確認
    try:
        response = requests.get(f"{BACKEND_URL}/api/strategies/unified", timeout=5)
        print(f"バックエンドサーバー: OK (ステータス: {response.status_code})")
    except Exception as e:
        print(f"バックエンドサーバー: NG ({e})")
    
    print()

if __name__ == "__main__":
    check_server_status()
    test_frontend_api()
    test_backend_direct()
    test_strategy_stats()
    print("=== フロントエンド表示動作確認完了 ===")
