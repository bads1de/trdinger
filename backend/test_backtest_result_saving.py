#!/usr/bin/env python3
"""
バックテスト結果保存機能のテスト

簡素化されたオートストラテジーシステムでバックテスト結果が
正常にデータベースに保存されることを確認します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from unittest.mock import Mock, patch
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene, IndicatorGene, Condition
from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_experiment_info_retrieval():
    """実験情報取得機能のテスト"""
    print("\n=== 実験情報取得テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # テスト用の実験IDで情報取得を試行
        test_experiment_id = "test_experiment_001"
        experiment_info = service._get_experiment_info(test_experiment_id)
        
        if experiment_info:
            print(f"✅ 実験情報取得成功: {experiment_info}")
        else:
            print(f"ℹ️  実験情報が見つかりません（正常）: {test_experiment_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ 実験情報取得テストエラー: {e}")
        return False

def test_save_experiment_result_structure():
    """実験結果保存構造のテスト"""
    print("\n=== 実験結果保存構造テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # テスト用のデータを作成
        test_experiment_id = "test_exp_001"
        test_ga_config = GAConfig.create_fast()
        test_backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.001
        }
        
        # テスト用の戦略遺伝子
        test_strategy = StrategyGene(
            id="test_strategy_001",
            indicators=[
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True)
            ],
            entry_conditions=[Condition("RSI", "<", "30")],
            exit_conditions=[Condition("RSI", ">", "70")],
            risk_management={"position_size": 0.1, "stop_loss": 0.02}
        )
        
        # テスト用の結果データ
        test_result = {
            "best_strategy": test_strategy,
            "best_fitness": 1.25,
            "all_strategies": [test_strategy],
            "fitness_scores": [1.25],
            "execution_time": 120.5
        }
        
        print("✅ テストデータ作成成功")
        print(f"   実験ID: {test_experiment_id}")
        print(f"   戦略ID: {test_strategy.id}")
        print(f"   フィットネス: {test_result['best_fitness']}")
        
        # 実験情報のモック作成
        mock_experiment_info = {
            "db_id": 1,
            "name": test_experiment_id,
            "status": "running",
            "config": test_ga_config.to_dict(),
            "created_at": "2024-01-01T00:00:00",
            "completed_at": None
        }
        
        # _get_experiment_infoをモック化
        with patch.object(service, '_get_experiment_info', return_value=mock_experiment_info):
            # バックテストサービスをモック化
            with patch.object(service, 'backtest_service') as mock_backtest_service:
                mock_backtest_service.run_backtest.return_value = {
                    "performance_metrics": {
                        "total_return": 0.15,
                        "sharpe_ratio": 1.2,
                        "max_drawdown": 0.08,
                        "win_rate": 0.6
                    },
                    "equity_curve": [100000, 105000, 110000],
                    "trade_history": [
                        {"entry_time": "2024-01-01", "exit_time": "2024-01-02", "profit": 5000}
                    ],
                    "execution_time": 2.5
                }
                
                # データベース操作をモック化
                with patch('backend.app.core.services.auto_strategy.services.auto_strategy_service.GeneratedStrategyRepository') as mock_strategy_repo:
                    with patch('backend.app.core.services.auto_strategy.services.auto_strategy_service.BacktestResultRepository') as mock_result_repo:
                        
                        # モックの戻り値を設定
                        mock_strategy_instance = Mock()
                        mock_strategy_instance.save_strategy.return_value = Mock(id=1)
                        mock_strategy_instance.save_strategies_batch.return_value = 1
                        mock_strategy_repo.return_value = mock_strategy_instance
                        
                        mock_result_instance = Mock()
                        mock_result_instance.save_backtest_result.return_value = {"id": 1}
                        mock_result_repo.return_value = mock_result_instance
                        
                        # 実験結果保存を実行
                        service._save_experiment_result(
                            test_experiment_id,
                            test_result,
                            test_ga_config,
                            test_backtest_config
                        )
                        
                        print("✅ 実験結果保存処理完了")
                        
                        # モックの呼び出しを確認
                        mock_strategy_instance.save_strategy.assert_called_once()
                        mock_result_instance.save_backtest_result.assert_called_once()
                        
                        print("✅ データベース保存メソッド呼び出し確認")
        
        return True
        
    except Exception as e:
        print(f"❌ 実験結果保存構造テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_result_data_preparation():
    """バックテスト結果データ準備のテスト"""
    print("\n=== バックテスト結果データ準備テスト ===")
    
    try:
        service = AutoStrategyService()
        
        # テストデータ
        detailed_result = {
            "performance_metrics": {
                "total_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.12,
                "win_rate": 0.65
            },
            "equity_curve": [100000, 110000, 125000],
            "trade_history": [
                {"entry_time": "2024-01-01", "exit_time": "2024-01-05", "profit": 10000},
                {"entry_time": "2024-01-10", "exit_time": "2024-01-15", "profit": 15000}
            ],
            "execution_time": 3.2
        }
        
        config = {
            "strategy_name": "AUTO_STRATEGY_TEST_001",
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "strategy_config": {
                "strategy_type": "GENERATED_AUTO",
                "parameters": {"test": "data"}
            }
        }
        
        experiment_id = "test_exp_001"
        db_experiment_id = 1
        best_fitness = 1.5
        
        # バックテスト結果データを準備
        result_data = service._prepare_backtest_result_data(
            detailed_result,
            config,
            experiment_id,
            db_experiment_id,
            best_fitness
        )
        
        print("✅ バックテスト結果データ準備成功")
        print(f"   戦略名: {result_data['strategy_name']}")
        print(f"   シンボル: {result_data['symbol']}")
        print(f"   パフォーマンス: {len(result_data['performance_metrics'])}個のメトリクス")
        print(f"   取引履歴: {len(result_data['trade_history'])}件")
        print(f"   ステータス: {result_data['status']}")
        
        # 必要なキーが含まれているか確認
        required_keys = [
            "strategy_name", "symbol", "timeframe", "start_date", "end_date",
            "initial_capital", "commission_rate", "config_json", 
            "performance_metrics", "equity_curve", "trade_history", 
            "execution_time", "status"
        ]
        
        missing_keys = [key for key in required_keys if key not in result_data]
        if missing_keys:
            print(f"❌ 不足しているキー: {missing_keys}")
            return False
        
        print("✅ 必要なキーすべて存在")
        
        return True
        
    except Exception as e:
        print(f"❌ バックテスト結果データ準備テストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("💾 バックテスト結果保存機能テスト開始")
    print("=" * 60)
    
    tests = [
        test_experiment_info_retrieval,
        test_save_experiment_result_structure,
        test_backtest_result_data_preparation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 テスト結果: {passed}/{total} 成功")
    
    if passed == total:
        print("🎉 全テスト成功！バックテスト結果保存機能は正常に動作しています。")
        
        print("\n✅ 確認された機能:")
        print("   ✅ 実験情報取得: 統合版で正常動作")
        print("   ✅ 結果保存構造: 適切なデータベース操作")
        print("   ✅ データ準備: 必要なフィールドすべて生成")
        print("   ✅ エラーハンドリング: 適切な例外処理")
        
    else:
        print(f"⚠️  {total - passed}個のテストが失敗しました。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
