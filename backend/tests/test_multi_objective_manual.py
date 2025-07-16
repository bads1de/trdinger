"""
多目的最適化GA 手動テスト

実際のAPIエンドポイントを使用して多目的最適化GAをテストします。
"""

import requests
import json
import time
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# APIベースURL
BASE_URL = "http://localhost:8000"

def test_multi_objective_ga_api():
    """多目的最適化GA APIの手動テスト"""
    
    # 1. 多目的最適化設定
    request_data = {
        "experiment_name": "Manual_Multi_Objective_Test",
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "initial_capital": 10000,
            "commission_rate": 0.001
        },
        "ga_config": {
            "population_size": 6,
            "generations": 3,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 2,
            "max_indicators": 3,
            "allowed_indicators": ["SMA", "EMA", "RSI", "MACD", "BB", "ATR"],
            # 多目的最適化設定
            "enable_multi_objective": True,
            "objectives": ["total_return", "max_drawdown"],
            "objective_weights": [1.0, -1.0]
        }
    }
    
    logger.info("多目的最適化GA実験を開始します...")
    logger.info(f"設定: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
    
    try:
        # 2. GA実験開始
        response = requests.post(
            f"{BASE_URL}/api/auto-strategy/generate",
            json=request_data,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"実験開始エラー: {response.status_code}")
            logger.error(f"レスポンス: {response.text}")
            return False
        
        result = response.json()
        experiment_id = result.get("experiment_id")
        
        if not experiment_id:
            logger.error("実験IDが取得できませんでした")
            return False
        
        logger.info(f"実験開始成功: {experiment_id}")
        
        # 3. 実験完了まで待機
        max_wait_time = 300  # 5分
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # 実験結果を取得
                result_response = requests.get(
                    f"{BASE_URL}/api/auto-strategy/experiments/{experiment_id}/results",
                    timeout=10
                )
                
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    
                    if result_data.get("success"):
                        logger.info("実験完了！")
                        
                        # 4. 結果分析
                        analyze_multi_objective_results(result_data)
                        return True
                    else:
                        logger.info(f"実験進行中... ({time.time() - start_time:.1f}秒経過)")
                
                elif result_response.status_code == 404:
                    logger.info(f"実験進行中... ({time.time() - start_time:.1f}秒経過)")
                
                else:
                    logger.warning(f"結果取得エラー: {result_response.status_code}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"結果取得リクエストエラー: {e}")
            
            time.sleep(10)  # 10秒待機
        
        logger.error("実験がタイムアウトしました")
        return False
        
    except requests.exceptions.RequestException as e:
        logger.error(f"APIリクエストエラー: {e}")
        return False

def analyze_multi_objective_results(result_data):
    """多目的最適化結果の分析"""
    logger.info("=== 多目的最適化結果分析 ===")
    
    # 基本情報
    result = result_data.get("result", {})
    logger.info(f"実験成功: {result_data.get('success')}")
    logger.info(f"多目的最適化: {result_data.get('is_multi_objective')}")
    
    # パレート最適解の確認
    pareto_front = result_data.get("pareto_front", [])
    objectives = result_data.get("objectives", [])
    
    if pareto_front and objectives:
        logger.info(f"目的関数: {objectives}")
        logger.info(f"パレート最適解数: {len(pareto_front)}")
        
        # パレート最適解の詳細
        for i, solution in enumerate(pareto_front[:5]):  # 最初の5個を表示
            fitness_values = solution.get("fitness_values", [])
            logger.info(f"解 {i+1}: {dict(zip(objectives, fitness_values))}")
        
        # 最良戦略の情報
        best_strategy = result.get("best_strategy", {})
        if best_strategy:
            best_fitness = best_strategy.get("fitness_values", [])
            logger.info(f"最良戦略フィットネス: {dict(zip(objectives, best_fitness))}")
    
    else:
        logger.warning("パレート最適解が見つかりませんでした")
    
    # 実行時間
    execution_time = result.get("execution_time", 0)
    logger.info(f"実行時間: {execution_time:.2f}秒")

def test_single_vs_multi_objective():
    """単一目的と多目的最適化の比較テスト"""
    logger.info("=== 単一目的 vs 多目的最適化比較テスト ===")
    
    base_request = {
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001
        },
        "ga_config": {
            "population_size": 4,
            "generations": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 1,
            "max_indicators": 2,
            "allowed_indicators": ["SMA", "EMA", "RSI"]
        }
    }
    
    # 単一目的設定
    single_request = base_request.copy()
    single_request["experiment_name"] = "Comparison_Single_Objective"
    single_request["ga_config"]["enable_multi_objective"] = False
    
    # 多目的設定
    multi_request = base_request.copy()
    multi_request["experiment_name"] = "Comparison_Multi_Objective"
    multi_request["ga_config"]["enable_multi_objective"] = True
    multi_request["ga_config"]["objectives"] = ["total_return", "max_drawdown"]
    multi_request["ga_config"]["objective_weights"] = [1.0, -1.0]
    
    logger.info("単一目的最適化と多目的最適化の設定が準備できました")
    logger.info("実際のテストを実行するには、手動でAPIを呼び出してください")

if __name__ == "__main__":
    print("多目的最適化GA 手動テスト")
    print("=" * 50)
    
    # サーバーが起動しているかチェック
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ バックエンドサーバーが起動しています")
            
            # 実際のテスト実行
            print("\n1. 多目的最適化GAテストを実行中...")
            success = test_multi_objective_ga_api()
            
            if success:
                print("✅ 多目的最適化GAテスト成功！")
            else:
                print("❌ 多目的最適化GAテスト失敗")
            
            print("\n2. 比較テスト設定を準備中...")
            test_single_vs_multi_objective()
            
        else:
            print("❌ バックエンドサーバーにアクセスできません")
            
    except requests.exceptions.RequestException:
        print("❌ バックエンドサーバーが起動していません")
        print("先にバックエンドサーバーを起動してください: uvicorn app.main:app --reload")
