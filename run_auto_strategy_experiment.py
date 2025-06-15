"""
BTC/USDT:USDT シンボルでの新規オートストラテジー実験実行スクリプト
"""

import requests
import json
import time
from datetime import datetime

# API設定
BACKEND_URL = "http://127.0.0.1:8000"

def run_auto_strategy_experiment():
    """新しいオートストラテジー実験を実行"""
    
    print("=== BTC/USDT:USDT オートストラテジー実験開始 ===\n")
    
    # 実験設定
    experiment_config = {
        "experiment_name": f"BTC_USDT_USDT_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "base_config": {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2024-10-01",
            "end_date": "2024-12-01",
            "initial_capital": 100000,
            "commission_rate": 0.001
        },
        "ga_config": {
            "population_size": 20,
            "generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elite_size": 2,
            "fitness_function": "sharpe_ratio",
            "strategy_types": ["SMA_CROSS", "EMA_CROSS", "RSI_MEAN_REVERSION"],
            "max_indicators": 3
        }
    }
    
    try:
        # 1. 実験開始
        print("1. オートストラテジー実験を開始...")
        start_url = f"{BACKEND_URL}/api/auto-strategy/generate"
        
        response = requests.post(start_url, json=experiment_config, timeout=30)
        print(f"   ステータス: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   エラー: {response.text}")
            return None
            
        result = response.json()
        if not result.get("success"):
            print(f"   実験開始失敗: {result.get('message')}")
            return None
            
        experiment_id = result.get("experiment_id")
        print(f"   実験ID: {experiment_id}")
        print(f"   メッセージ: {result.get('message')}")
        
        # 2. 進捗監視
        print("\n2. 実験進捗を監視...")
        progress_url = f"{BACKEND_URL}/api/auto-strategy/experiments/{experiment_id}/progress"
        
        max_wait_time = 300  # 5分
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(progress_url, timeout=10)
                if response.status_code == 200:
                    progress_data = response.json()
                    
                    status = progress_data.get("status", "unknown")
                    progress = progress_data.get("progress", 0)
                    current_gen = progress_data.get("current_generation", 0)
                    total_gen = progress_data.get("total_generations", 0)
                    best_fitness = progress_data.get("best_fitness", 0)
                    
                    print(f"   進捗: {progress:.1%} | 世代: {current_gen}/{total_gen} | 最高フィットネス: {best_fitness:.3f} | ステータス: {status}")
                    
                    if status == "completed":
                        print("   実験完了！")
                        break
                    elif status == "failed":
                        print("   実験失敗")
                        return experiment_id
                        
                else:
                    print(f"   進捗取得エラー: {response.status_code}")
                    
            except Exception as e:
                print(f"   進捗監視エラー: {e}")
                
            time.sleep(10)  # 10秒待機
            
        # 3. 結果取得
        print("\n3. 実験結果を取得...")
        results_url = f"{BACKEND_URL}/api/auto-strategy/experiments/{experiment_id}/results"
        
        try:
            response = requests.get(results_url, timeout=30)
            if response.status_code == 200:
                results_data = response.json()
                
                if results_data.get("success"):
                    strategies = results_data.get("strategies", [])
                    print(f"   生成された戦略数: {len(strategies)}")
                    
                    # 上位3戦略を表示
                    for i, strategy in enumerate(strategies[:3]):
                        print(f"   [{i+1}] フィットネス: {strategy.get('fitness_score', 0):.3f}")
                        print(f"       戦略名: {strategy.get('name', '不明')}")
                        print(f"       指標: {', '.join(strategy.get('indicators', []))}")
                        print(f"       期待リターン: {strategy.get('expected_return', 0):.3f}")
                        print(f"       シャープレシオ: {strategy.get('sharpe_ratio', 0):.3f}")
                        print()
                else:
                    print(f"   結果取得失敗: {results_data.get('message')}")
            else:
                print(f"   結果取得エラー: {response.status_code}")
                
        except Exception as e:
            print(f"   結果取得例外: {e}")
            
        return experiment_id
        
    except Exception as e:
        print(f"実験実行エラー: {e}")
        return None

def verify_database_update(experiment_id):
    """データベースの更新を確認"""
    print("\n4. データベース更新確認...")
    
    try:
        # 統合APIで新しい戦略が取得できるかテスト
        api_url = f"{BACKEND_URL}/api/strategies/unified"
        params = {"experiment_id": experiment_id, "limit": 10}
        
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            strategies = data.get("strategies", [])
            print(f"   統合APIで取得された戦略数: {len(strategies)}")
            
            for strategy in strategies:
                print(f"   - {strategy.get('name', '不明')} (実験ID: {strategy.get('experiment_id')})")
        else:
            print(f"   統合API呼び出しエラー: {response.status_code}")
            
    except Exception as e:
        print(f"   データベース確認エラー: {e}")

if __name__ == "__main__":
    experiment_id = run_auto_strategy_experiment()
    
    if experiment_id:
        verify_database_update(experiment_id)
        print(f"\n=== 実験完了 (ID: {experiment_id}) ===")
    else:
        print("\n=== 実験失敗 ===")
