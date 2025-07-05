#!/usr/bin/env python3
"""
実際のオートストラテジー機能で取引量0問題が修正されたかテストするスクリプト
"""

import sys
import os
import logging
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
from app.core.services.auto_strategy.utils.strategy_gene_utils import create_default_strategy_gene
from app.core.services.auto_strategy.models.strategy_gene import StrategyGene

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_strategy_execution():
    """単一戦略の実行テスト"""
    print("=== 単一戦略の実行テスト ===")
    
    try:
        # AutoStrategyServiceを初期化
        service = AutoStrategyService()
        
        # テスト用の戦略遺伝子を作成
        test_gene = create_default_strategy_gene(StrategyGene)
        
        # 取引量を明示的に設定
        test_gene.risk_management["position_size"] = 0.15  # 15%
        print(f"設定した取引量: {test_gene.risk_management['position_size']}")
        
        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-12-01",
            "end_date": "2024-12-02",  # 短期間でテスト
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        print("戦略テストを実行中...")
        result = service.test_strategy_generation(test_gene, backtest_config)
        
        if result.get("success"):
            print("✅ 戦略テスト成功")
            
            backtest_result = result.get("backtest_result", {})
            performance_metrics = backtest_result.get("performance_metrics", {})
            
            total_trades = performance_metrics.get("total_trades", 0)
            total_return = performance_metrics.get("total_return", 0.0)
            final_equity = performance_metrics.get("equity_final", 0.0)
            
            print(f"取引回数: {total_trades}")
            print(f"総リターン: {total_return:.2f}%")
            print(f"最終資産: {final_equity:,.2f}")
            
            if total_trades > 0:
                print("🎉 取引量0問題が解決されました！")
                return True
            else:
                print("⚠️ 取引回数が0です。条件が厳しすぎる可能性があります。")
                
                # 取引履歴を確認
                trade_history = backtest_result.get("trade_history", [])
                print(f"取引履歴の件数: {len(trade_history)}")
                
                return False
        else:
            print("❌ 戦略テスト失敗")
            errors = result.get("errors", [])
            if errors:
                print(f"エラー: {errors}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_small_ga_execution():
    """小規模なGA実行テスト"""
    print("\n=== 小規模なGA実行テスト ===")
    
    try:
        # AutoStrategyServiceを初期化
        service = AutoStrategyService()
        
        # 小規模なGA設定
        ga_config_dict = {
            "population_size": 3,
            "generations": 1,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "allowed_indicators": ["RSI", "SMA"],
            "fitness_weights": {
                "total_return": 0.4,
                "sharpe_ratio": 0.3,
                "max_drawdown": 0.2,
                "win_rate": 0.1
            }
        }
        
        # バックテスト設定
        backtest_config = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-12-01",
            "end_date": "2024-12-02",
            "initial_capital": 100000.0,
            "commission_rate": 0.001
        }
        
        print("小規模GA実行を開始...")
        
        # 実験を開始（同期的に実行するため、バックグラウンドタスクは使用しない）
        from fastapi import BackgroundTasks
        background_tasks = BackgroundTasks()
        
        experiment_id = service.start_strategy_generation(
            experiment_name="TEST_VOLUME_FIX",
            ga_config_dict=ga_config_dict,
            backtest_config_dict=backtest_config,
            background_tasks=background_tasks
        )
        
        print(f"実験ID: {experiment_id}")
        
        # 少し待ってから進捗を確認
        import time
        time.sleep(2)
        
        progress = service.get_progress(experiment_id)
        print(f"進捗: {progress}")
        
        if progress and progress.get("status") in ["running", "completed"]:
            print("✅ GA実行が開始されました")
            return True
        else:
            print("❌ GA実行に問題があります")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン関数"""
    print("実際のオートストラテジー機能での取引量0問題修正テストを開始します\n")
    
    results = []
    
    # テスト1: 単一戦略の実行
    results.append(test_single_strategy_execution())
    
    # テスト2: 小規模なGA実行（時間がかかるのでオプション）
    print("\n小規模GA実行テストを実行しますか？ (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response == 'y':
            results.append(test_small_ga_execution())
        else:
            print("GA実行テストをスキップしました")
    except KeyboardInterrupt:
        print("\nテストが中断されました")
        return False
    
    # 結果のまとめ
    print("\n" + "="*50)
    print("実際のオートストラテジー機能テスト結果:")
    print(f"成功: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 実際のオートストラテジー機能で取引量0問題が修正されています！")
    else:
        print("⚠️ 一部のテストが失敗しました。追加の調査が必要です。")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
