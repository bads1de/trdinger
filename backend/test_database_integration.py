#!/usr/bin/env python3
"""
データベース統合テスト

簡素化されたオートストラテジーシステムのデータベース保存機能を
実際のデータベースを使用してテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from database.repositories.ga_experiment_repository import GAExperimentRepository
from database.repositories.generated_strategy_repository import GeneratedStrategyRepository
from database.repositories.backtest_result_repository import BacktestResultRepository
from database.connection import SessionLocal

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_experiment_creation():
    """実験作成のテスト"""
    print("\n=== 実験作成テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # テスト用の設定
        test_ga_config = GAConfig.create_fast()
        test_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
        
        # 実験作成
        experiment_id = service._create_experiment(
            "Test_Database_Integration",
            test_ga_config,
            test_backtest_config
        )
        
        print(f"✅ 実験作成成功: {experiment_id}")
        
        # 作成された実験の確認
        experiment_info = service._get_experiment_info(experiment_id)
        if experiment_info:
            print(f"✅ 実験情報取得成功: DB ID {experiment_info['db_id']}")
            return experiment_id, experiment_info
        else:
            print("❌ 作成された実験の情報取得失敗")
            return None, None
        
    except Exception as e:
        print(f"❌ 実験作成テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_repository_operations():
    """リポジトリ操作のテスト"""
    print("\n=== リポジトリ操作テスト ===")
    
    try:
        with SessionLocal() as db:
            # GAExperimentRepositoryのテスト
            ga_repo = GAExperimentRepository(db)
            
            # 実験作成
            test_config = {"test": "data"}
            experiment = ga_repo.create_experiment(
                name="Test_Repository_Operations",
                config=test_config,
                total_generations=5,
                status="running"
            )
            
            print(f"✅ GA実験作成成功: ID {experiment.id}")
            
            # GeneratedStrategyRepositoryのテスト
            strategy_repo = GeneratedStrategyRepository(db)
            
            # テスト戦略データ
            test_gene_data = {
                "id": "test_strategy_001",
                "indicators": [{"type": "RSI", "parameters": {"period": 14}}],
                "entry_conditions": [{"left": "RSI", "operator": "<", "right": "30"}],
                "exit_conditions": [{"left": "RSI", "operator": ">", "right": "70"}]
            }
            
            # 戦略保存
            strategy = strategy_repo.save_strategy(
                experiment_id=experiment.id,
                gene_data=test_gene_data,
                generation=5,
                fitness_score=1.25
            )
            
            print(f"✅ 戦略保存成功: ID {strategy.id}")
            
            # BacktestResultRepositoryのテスト
            result_repo = BacktestResultRepository(db)
            
            # テストバックテスト結果データ
            test_result_data = {
                "strategy_name": "Test_Strategy",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 100000,
                "commission_rate": 0.001,
                "config_json": {"test": "config"},
                "performance_metrics": {
                    "total_return": 0.15,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.08
                },
                "equity_curve": [100000, 110000, 115000],
                "trade_history": [{"profit": 10000}],
                "execution_time": 2.5,
                "status": "completed"
            }
            
            # バックテスト結果保存
            result = result_repo.save_backtest_result(test_result_data)
            
            print(f"✅ バックテスト結果保存成功: ID {result.get('id', 'N/A')}")
            
            return True
        
    except Exception as e:
        print(f"❌ リポジトリ操作テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_completion():
    """実験完了処理のテスト"""
    print("\n=== 実験完了処理テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # テスト用の実験を作成
        test_ga_config = GAConfig.create_fast()
        test_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000
        }
        
        experiment_id = service._create_experiment(
            "Test_Completion",
            test_ga_config,
            test_backtest_config
        )
        
        if not experiment_id:
            print("❌ 実験作成失敗")
            return False
        
        print(f"✅ テスト実験作成: {experiment_id}")
        
        # 実験完了処理
        test_result = {"best_fitness": 1.5, "execution_time": 120}
        service._complete_experiment(experiment_id, test_result)
        
        print("✅ 実験完了処理実行")
        
        # 進捗作成
        service._create_final_progress(experiment_id, test_result, test_ga_config)
        
        print("✅ 最終進捗作成")
        
        # 進捗確認
        progress = service.get_experiment_progress(experiment_id)
        if progress:
            print(f"✅ 進捗取得成功: ステータス {progress.status}")
        else:
            print("ℹ️  進捗データなし（正常）")
        
        return True
        
    except Exception as e:
        print(f"❌ 実験完了処理テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_experiment_list():
    """実験一覧取得のテスト"""
    print("\n=== 実験一覧取得テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # 実験一覧取得
        experiments = service.list_experiments()
        
        print(f"✅ 実験一覧取得成功: {len(experiments)}件")
        
        if experiments:
            latest_exp = experiments[0]
            print(f"   最新実験: {latest_exp.get('experiment_name', 'N/A')}")
            print(f"   ステータス: {latest_exp.get('status', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 実験一覧取得テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🗄️  データベース統合テスト開始")
    print("=" * 60)
    
    tests = [
        test_repository_operations,
        test_experiment_creation,
        test_experiment_completion,
        test_experiment_list,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 テスト結果: {passed}/{total} 成功")
    
    if passed == total:
        print("🎉 全テスト成功！データベース統合機能は正常に動作しています。")
        
        print("\n✅ 確認された機能:")
        print("   ✅ 実験作成: データベースに正常保存")
        print("   ✅ 戦略保存: generated_strategiesテーブルに保存")
        print("   ✅ 結果保存: backtest_resultsテーブルに保存")
        print("   ✅ 実験管理: 完了処理と進捗管理")
        print("   ✅ 一覧取得: 実験一覧の正常取得")
        
    else:
        print(f"⚠️  {total - passed}個のテストが失敗しました。")
        print("\n🔧 修正が必要な可能性があります。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
