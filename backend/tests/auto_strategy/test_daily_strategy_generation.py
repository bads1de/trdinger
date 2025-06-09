#!/usr/bin/env python3
"""
日足データを使用した自動戦略生成のテストスクリプト
"""

import asyncio
import json
import time
import requests
from datetime import datetime, timedelta

# APIベースURL
BASE_URL = "http://localhost:8000"

def test_ga_strategy_generation():
    """日足データでGA戦略生成をテスト"""
    print("🚀 日足データでの自動戦略生成テスト開始")
    print("=" * 60)
    
    # 1. サーバーの動作確認
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ サーバー接続確認: OK")
        else:
            print("❌ サーバー接続失敗")
            return
    except Exception as e:
        print(f"❌ サーバー接続エラー: {e}")
        return
    
    # 2. GA設定（小規模テスト用）
    ga_request = {
        "experiment_name": "Daily_BTC_Strategy_Test",
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1d",  # 日足データを使用
            "start_date": "2024-01-01",
            "end_date": "2024-04-09",  # 利用可能なデータの範囲
            "initial_capital": 100000,
            "commission_rate": 0.001
        },
        "ga_config": {
            "population_size": 5,  # 小規模テスト
            "generations": 3,      # 短時間で完了
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "fitness_weights": {
                "total_return": 0.4,
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.2,
                "win_rate": 0.1
            }
        }
    }
    
    print("📊 テスト設定:")
    print(f"  シンボル: {ga_request['base_config']['symbol']}")
    print(f"  時間軸: {ga_request['base_config']['timeframe']}")
    print(f"  期間: {ga_request['base_config']['start_date']} ～ {ga_request['base_config']['end_date']}")
    print(f"  個体数: {ga_request['ga_config']['population_size']}")
    print(f"  世代数: {ga_request['ga_config']['generations']}")
    print()
    
    # 3. GA戦略生成開始
    try:
        print("🧬 GA戦略生成を開始...")
        response = requests.post(
            f"{BASE_URL}/api/auto-strategy/generate",
            json=ga_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ GA開始失敗: {response.status_code}")
            print(f"エラー詳細: {response.text}")
            return
        
        result = response.json()
        experiment_id = result.get("experiment_id")
        
        if not experiment_id:
            print("❌ 実験IDが取得できませんでした")
            return
        
        print(f"✅ GA実験開始成功")
        print(f"  実験ID: {experiment_id}")
        print()
        
    except Exception as e:
        print(f"❌ GA開始エラー: {e}")
        return
    
    # 4. 進捗監視
    print("📈 進捗監視中...")
    max_wait_time = 300  # 最大5分待機
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            # 進捗取得
            progress_response = requests.get(
                f"{BASE_URL}/api/auto-strategy/experiments/{experiment_id}/progress"
            )
            
            if progress_response.status_code == 200:
                progress = progress_response.json()
                
                if progress.get("status") == "completed":
                    print("🎉 GA実験完了!")
                    break
                elif progress.get("status") == "error":
                    print(f"❌ GA実験エラー: {progress.get('error_message', 'Unknown error')}")
                    return
                else:
                    # 進捗表示
                    current_gen = progress.get("current_generation", 0)
                    total_gen = progress.get("total_generations", 0)
                    best_fitness = progress.get("best_fitness", 0)
                    
                    print(f"  世代 {current_gen}/{total_gen}, 最高フィットネス: {best_fitness:.4f}")
            
            time.sleep(2)  # 2秒間隔で確認
            
        except Exception as e:
            print(f"進捗確認エラー: {e}")
            time.sleep(2)
    
    # 5. 結果取得
    try:
        print("\n📋 結果取得中...")
        results_response = requests.get(
            f"{BASE_URL}/api/auto-strategy/experiments/{experiment_id}/results"
        )
        
        if results_response.status_code == 200:
            results = results_response.json()
            
            print("✅ 結果取得成功")
            print("\n🏆 最優秀戦略:")
            
            best_strategy = results.get("best_strategy")
            if best_strategy:
                gene = best_strategy.get("gene", {})
                performance = best_strategy.get("performance", {})
                
                print(f"  フィットネス値: {best_strategy.get('fitness', 0):.4f}")
                print(f"  総リターン: {performance.get('total_return', 0):.2%}")
                print(f"  シャープレシオ: {performance.get('sharpe_ratio', 0):.4f}")
                print(f"  最大ドローダウン: {performance.get('max_drawdown', 0):.2%}")
                print(f"  勝率: {performance.get('win_rate', 0):.2%}")
                print(f"  取引回数: {performance.get('total_trades', 0)}")
                
                # 戦略の詳細
                indicators = gene.get("indicators", [])
                print(f"\n📊 使用指標 ({len(indicators)}個):")
                for i, indicator in enumerate(indicators, 1):
                    print(f"    {i}. {indicator.get('type')} - {indicator.get('parameters', {})}")
                
                entry_conditions = gene.get("entry_conditions", [])
                print(f"\n📈 エントリー条件 ({len(entry_conditions)}個):")
                for i, condition in enumerate(entry_conditions, 1):
                    print(f"    {i}. {condition}")
                
                exit_conditions = gene.get("exit_conditions", [])
                print(f"\n📉 エグジット条件 ({len(exit_conditions)}個):")
                for i, condition in enumerate(exit_conditions, 1):
                    print(f"    {i}. {condition}")
            
            # 実験統計
            stats = results.get("experiment_stats", {})
            print(f"\n📊 実験統計:")
            print(f"  実行時間: {stats.get('execution_time', 0):.2f}秒")
            print(f"  評価された戦略数: {stats.get('total_evaluations', 0)}")
            print(f"  有効戦略数: {stats.get('valid_strategies', 0)}")
            
        else:
            print(f"❌ 結果取得失敗: {results_response.status_code}")
            print(f"エラー詳細: {results_response.text}")
            
    except Exception as e:
        print(f"❌ 結果取得エラー: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 テスト完了")

if __name__ == "__main__":
    test_ga_strategy_generation()
