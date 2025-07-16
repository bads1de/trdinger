"""
多目的最適化GA API 手動テスト

実際のAPIエンドポイントを直接テストします。
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

def test_single_objective_api():
    """単一目的最適化APIテスト"""
    logger.info("=== 単一目的最適化APIテスト ===")
    
    # リクエストデータ
    request_data = {
        "experiment_name": "API_Test_Single_Objective",
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        },
        "ga_config": {
            "population_size": 4,
            "generations": 1,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 1,
            "max_indicators": 2,
            "allowed_indicators": ["SMA", "EMA", "RSI"],
            "enable_multi_objective": False,
            "objectives": ["total_return"],
            "objective_weights": [1.0],
        },
    }
    
    try:
        # API呼び出し
        response = requests.post(
            f"{BASE_URL}/api/auto-strategy/generate",
            json=request_data,
            timeout=30
        )
        
        logger.info(f"ステータスコード: {response.status_code}")
        logger.info(f"レスポンス: {response.json()}")
        
        if response.status_code == 200:
            logger.info("✅ 単一目的最適化API成功")
            return response.json().get("experiment_id")
        else:
            logger.error("❌ 単一目的最適化API失敗")
            return None
            
    except Exception as e:
        logger.error(f"❌ 単一目的最適化APIエラー: {e}")
        return None

def test_multi_objective_api():
    """多目的最適化APIテスト"""
    logger.info("=== 多目的最適化APIテスト ===")
    
    # リクエストデータ
    request_data = {
        "experiment_name": "API_Test_Multi_Objective",
        "base_config": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2023-01-15",
            "initial_capital": 10000,
            "commission_rate": 0.001,
        },
        "ga_config": {
            "population_size": 4,
            "generations": 1,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "elite_size": 1,
            "max_indicators": 2,
            "allowed_indicators": ["SMA", "EMA", "RSI"],
            "enable_multi_objective": True,
            "objectives": ["total_return", "max_drawdown"],
            "objective_weights": [1.0, -1.0],
        },
    }
    
    try:
        # API呼び出し
        response = requests.post(
            f"{BASE_URL}/api/auto-strategy/generate",
            json=request_data,
            timeout=30
        )
        
        logger.info(f"ステータスコード: {response.status_code}")
        logger.info(f"レスポンス: {response.json()}")
        
        if response.status_code == 200:
            logger.info("✅ 多目的最適化API成功")
            return response.json().get("experiment_id")
        else:
            logger.error("❌ 多目的最適化API失敗")
            return None
            
    except Exception as e:
        logger.error(f"❌ 多目的最適化APIエラー: {e}")
        return None

def test_experiment_result_api(experiment_id):
    """実験結果取得APIテスト"""
    if not experiment_id:
        logger.warning("実験IDがないため、結果取得テストをスキップ")
        return
        
    logger.info(f"=== 実験結果取得APIテスト (ID: {experiment_id}) ===")
    
    try:
        # API呼び出し
        response = requests.get(
            f"{BASE_URL}/api/auto-strategy/experiments/{experiment_id}/results",
            timeout=10
        )
        
        logger.info(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ 実験結果取得API成功")
            
            # 結果の構造確認
            if result.get("is_multi_objective"):
                logger.info("📊 多目的最適化結果:")
                logger.info(f"  - 目的: {result.get('objectives', [])}")
                logger.info(f"  - パレート最適解数: {len(result.get('pareto_front', []))}")
            else:
                logger.info("📈 単一目的最適化結果:")
                logger.info(f"  - 最良フィットネス: {result.get('result', {}).get('best_fitness')}")
                
        elif response.status_code == 404:
            logger.warning("⚠️ 実験が見つかりません（まだ完了していない可能性があります）")
        else:
            logger.error(f"❌ 実験結果取得API失敗: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ 実験結果取得APIエラー: {e}")

def test_api_validation():
    """APIバリデーションテスト"""
    logger.info("=== APIバリデーションテスト ===")
    
    # 無効なリクエスト
    invalid_requests = [
        {
            "name": "必須フィールド不足",
            "data": {
                "experiment_name": "Invalid_Test",
                # base_configとga_configが不足
            }
        },
        {
            "name": "無効な目的関数",
            "data": {
                "experiment_name": "Invalid_Objective_Test",
                "base_config": {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-15",
                    "initial_capital": 10000,
                    "commission_rate": 0.001,
                },
                "ga_config": {
                    "population_size": 4,
                    "generations": 1,
                    "enable_multi_objective": True,
                    "objectives": ["invalid_objective"],
                    "objective_weights": [1.0],
                },
            }
        }
    ]
    
    for test_case in invalid_requests:
        logger.info(f"テストケース: {test_case['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/auto-strategy/generate",
                json=test_case["data"],
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.info(f"✅ 適切にエラーが返されました: {response.status_code}")
            else:
                logger.warning(f"⚠️ エラーが期待されましたが成功しました: {response.status_code}")
                
        except Exception as e:
            logger.info(f"✅ 適切にエラーが発生しました: {e}")

def main():
    """メインテスト実行"""
    logger.info("🚀 多目的最適化GA API手動テスト開始")
    
    # サーバー接続確認
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            logger.info("✅ サーバー接続確認成功")
        else:
            logger.error("❌ サーバーに接続できません")
            return
    except Exception as e:
        logger.error(f"❌ サーバー接続エラー: {e}")
        logger.error("バックエンドサーバーが起動していることを確認してください")
        return
    
    # テスト実行
    single_experiment_id = test_single_objective_api()
    multi_experiment_id = test_multi_objective_api()
    
    # 少し待ってから結果取得テスト
    time.sleep(2)
    test_experiment_result_api(single_experiment_id)
    test_experiment_result_api(multi_experiment_id)
    
    # バリデーションテスト
    test_api_validation()
    
    logger.info("🎉 API手動テスト完了")

if __name__ == "__main__":
    main()
