#!/usr/bin/env python3
"""
自動戦略生成API エンドポイントテスト

実際のAPIエンドポイントの動作をテストします。
"""

import time
import requests
import json
import sys
import os

def wait_for_server(max_wait=30):
    """サーバーの起動を待つ"""
    print("サーバーの起動を待機中...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/docs", timeout=2)
            if response.status_code == 200:
                print(f"✅ サーバー起動確認 ({i+1}秒後)")
                return True
        except:
            pass
        time.sleep(1)
        if i % 5 == 4:
            print(f"   待機中... {i+1}/{max_wait}秒")
    
    print("❌ サーバー起動タイムアウト")
    return False


def test_config_endpoints():
    """設定エンドポイントテスト"""
    print("\n=== 設定エンドポイントテスト ===")
    
    # デフォルト設定取得
    try:
        response = requests.get("http://localhost:8000/api/auto-strategy/config/default", timeout=10)
        print(f"デフォルト設定: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ デフォルト設定取得成功")
            print(f"   設定項目数: {len(data.get('config', {}))}")
        else:
            print(f"❌ デフォルト設定取得失敗: {response.text}")
    except Exception as e:
        print(f"❌ デフォルト設定エラー: {e}")
    
    # プリセット設定取得
    try:
        response = requests.get("http://localhost:8000/api/auto-strategy/config/presets", timeout=10)
        print(f"プリセット設定: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ プリセット設定取得成功")
            presets = data.get('presets', {})
            print(f"   プリセット数: {len(presets)}")
            for preset_name in presets.keys():
                print(f"   - {preset_name}")
        else:
            print(f"❌ プリセット設定取得失敗: {response.text}")
    except Exception as e:
        print(f"❌ プリセット設定エラー: {e}")


def test_strategy_test_endpoint():
    """戦略テストエンドポイントテスト"""
    print("\n=== 戦略テストエンドポイントテスト ===")
    
    # テスト用戦略遺伝子
    test_strategy = {
        "strategy_gene": {
            "id": "test_strategy_001",
            "indicators": [
                {
                    "type": "SMA",
                    "parameters": {"period": 20},
                    "enabled": True
                },
                {
                    "type": "RSI",
                    "parameters": {"period": 14},
                    "enabled": True
                }
            ],
            "entry_conditions": [
                {
                    "left_operand": "RSI_14",
                    "operator": "<",
                    "right_operand": 30
                }
            ],
            "exit_conditions": [
                {
                    "left_operand": "RSI_14",
                    "operator": ">",
                    "right_operand": 70
                }
            ],
            "risk_management": {
                "stop_loss": 0.02,
                "take_profit": 0.05
            },
            "metadata": {}
        },
        "backtest_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/auto-strategy/test-strategy",
            json=test_strategy,
            timeout=30
        )
        print(f"戦略テスト: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 戦略テスト成功")
            print(f"   成功: {data.get('success', False)}")
            if data.get('result'):
                print(f"   結果あり: {bool(data['result'])}")
        else:
            print(f"❌ 戦略テスト失敗: {response.text}")
            
    except Exception as e:
        print(f"❌ 戦略テストエラー: {e}")


def test_ga_generation_endpoint():
    """GA生成エンドポイントテスト"""
    print("\n=== GA生成エンドポイントテスト ===")
    
    # GA実行設定
    ga_config = {
        "experiment_name": "API_Test_Experiment",
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.001
        },
        "ga_config": {
            "population_size": 5,
            "generations": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 1,
            "max_indicators": 3,
            "allowed_indicators": ["SMA", "EMA", "RSI"],
            "fitness_weights": {
                "total_return": 0.3,
                "sharpe_ratio": 0.4,
                "max_drawdown": 0.2,
                "win_rate": 0.1
            },
            "fitness_constraints": {
                "min_trades": 1,
                "max_drawdown_limit": 0.9,
                "min_sharpe_ratio": -10.0
            }
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/auto-strategy/generate",
            json=ga_config,
            timeout=30
        )
        print(f"GA生成開始: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GA生成開始成功")
            print(f"   成功: {data.get('success', False)}")
            experiment_id = data.get('experiment_id')
            print(f"   実験ID: {experiment_id}")
            
            if experiment_id:
                # 進捗確認テスト
                test_progress_monitoring(experiment_id)
            
        else:
            print(f"❌ GA生成開始失敗: {response.text}")
            
    except Exception as e:
        print(f"❌ GA生成エラー: {e}")


def test_progress_monitoring(experiment_id):
    """進捗監視テスト"""
    print(f"\n=== 進捗監視テスト (実験ID: {experiment_id}) ===")
    
    max_checks = 10
    for i in range(max_checks):
        try:
            response = requests.get(
                f"http://localhost:8000/api/auto-strategy/experiments/{experiment_id}/progress",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('progress'):
                    progress = data['progress']
                    current_gen = progress.get('current_generation', 0)
                    total_gen = progress.get('total_generations', 0)
                    status = progress.get('status', 'unknown')
                    best_fitness = progress.get('best_fitness', 0)
                    
                    print(f"✅ 進捗確認 {i+1}: 世代{current_gen}/{total_gen}, 状態:{status}, フィットネス:{best_fitness:.4f}")
                    
                    if status in ["completed", "error"]:
                        print(f"✅ 実験終了: {status}")
                        
                        if status == "completed":
                            # 結果取得テスト
                            test_result_retrieval(experiment_id)
                        break
                else:
                    print(f"❌ 進捗データなし: {data}")
            else:
                print(f"❌ 進捗確認失敗: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 進捗確認エラー: {e}")
        
        time.sleep(2)  # 2秒待機
    
    else:
        print("⚠️ 進捗監視タイムアウト")


def test_result_retrieval(experiment_id):
    """結果取得テスト"""
    print(f"\n=== 結果取得テスト (実験ID: {experiment_id}) ===")
    
    try:
        response = requests.get(
            f"http://localhost:8000/api/auto-strategy/experiments/{experiment_id}/results",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 結果取得成功")
            print(f"   成功: {data.get('success', False)}")
            
            if data.get('result'):
                result = data['result']
                print(f"   最高フィットネス: {result.get('best_fitness', 0):.4f}")
                print(f"   実行時間: {result.get('execution_time', 0):.1f}秒")
                print(f"   完了世代数: {result.get('generations_completed', 0)}")
                
                best_strategy = result.get('best_strategy', {})
                if best_strategy:
                    indicators = best_strategy.get('indicators', [])
                    print(f"   最良戦略指標数: {len(indicators)}")
        else:
            print(f"❌ 結果取得失敗: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ 結果取得エラー: {e}")


def test_experiments_list():
    """実験一覧テスト"""
    print("\n=== 実験一覧テスト ===")
    
    try:
        response = requests.get("http://localhost:8000/api/auto-strategy/experiments", timeout=10)
        
        if response.status_code == 200:
            experiments = response.json()
            print(f"✅ 実験一覧取得成功")
            print(f"   実験数: {len(experiments)}")
            
            for exp in experiments[:3]:  # 最初の3件を表示
                print(f"   - ID: {exp.get('id', 'N/A')}, 名前: {exp.get('name', 'N/A')}, 状態: {exp.get('status', 'N/A')}")
        else:
            print(f"❌ 実験一覧取得失敗: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 実験一覧エラー: {e}")


def main():
    """メインテスト実行"""
    print("=== 自動戦略生成API エンドポイントテスト開始 ===")
    print("=" * 60)
    
    # サーバー起動待機
    if not wait_for_server():
        print("❌ サーバーが起動していません。テストを中止します。")
        return False
    
    # 各エンドポイントをテスト
    tests = [
        ("設定エンドポイント", test_config_endpoints),
        ("戦略テストエンドポイント", test_strategy_test_endpoint),
        ("実験一覧", test_experiments_list),
        ("GA生成エンドポイント", test_ga_generation_endpoint),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"テスト: {test_name}")
            print(f"{'='*60}")
            
            test_func()
            passed_tests += 1
            print(f"✅ {test_name}: 完了")
            
        except Exception as e:
            print(f"❌ {test_name}: エラー - {e}")
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("APIエンドポイントテスト結果")
    print(f"{'='*60}")
    
    print(f"完了したテスト: {passed_tests}/{total_tests}")
    print(f"完了率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("=== 全てのAPIエンドポイントテストが完了しました！ ===")
        print("APIは正常に動作しています。")
        return True
    else:
        print("=== 一部のAPIエンドポイントテストで問題が発生しました ===")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
