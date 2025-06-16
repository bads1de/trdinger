#!/usr/bin/env python3
"""
オートストラテジー機能のテストスクリプト
"""

from app.core.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.core.services.auto_strategy.models.ga_config import GAConfig
import time

def test_auto_strategy():
    try:
        service = AutoStrategyService()
        print('AutoStrategyService initialized successfully')
        
        ga_config = GAConfig(
            population_size=3,
            generations=1,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1,
            fitness_weights={
                'total_return': 0.4,
                'sharpe_ratio': 0.3,
                'max_drawdown': 0.2,
                'win_rate': 0.1
            }
        )
        print('GAConfig created successfully')
        
        # 利用可能なシンボルを使用
        backtest_config = {
            'symbol': 'BTC/USDT:USDT',  # 利用可能なシンボルを使用
            'timeframe': '1d',
            'start_date': '2024-01-01',
            'end_date': '2024-04-09',
            'initial_capital': 100000,
            'commission_rate': 0.001
        }
        print('Backtest config created successfully')
        
        experiment_id = service.start_strategy_generation(
            experiment_name='Test_Available_Symbol',
            ga_config=ga_config,
            backtest_config=backtest_config
        )
        print(f'Experiment started: {experiment_id}')
        
        # 少し待ってから進捗確認
        time.sleep(15)
        
        progress = service.get_experiment_progress(experiment_id)
        print(f'Progress: {progress}')
        
        if progress and progress.status == 'completed':
            print('実験完了！')
            
            # バックテスト結果を確認
            from database.connection import SessionLocal
            from database.repositories.backtest_result_repository import BacktestResultRepository
            
            db = SessionLocal()
            try:
                repo = BacktestResultRepository(db)
                results = repo.get_backtest_results(limit=5)
                print(f'最新のバックテスト結果数: {len(results)}')
                
                for result in results:
                    strategy_name = result.get('strategy_name', '')
                    if 'AUTO_STRATEGY' in strategy_name:
                        print(f'オートストラテジー結果見つかりました: {strategy_name}')
                        metrics = result.get('performance_metrics', {})
                        print(f'  リターン: {metrics.get("total_return", 0):.2f}%')
                        print(f'  シャープレシオ: {metrics.get("sharpe_ratio", 0):.3f}')
                        print(f'  取引数: {metrics.get("total_trades", 0)}')
                        print(f'  勝率: {metrics.get("win_rate", 0):.1f}%')
                        print(f'  最大ドローダウン: {metrics.get("max_drawdown", 0):.2f}%')
                        
                        # 取引履歴の確認
                        trade_history = result.get('trade_history', [])
                        print(f'  取引履歴数: {len(trade_history)}')
                        
                        # 資産曲線の確認
                        equity_curve = result.get('equity_curve', [])
                        print(f'  資産曲線データ数: {len(equity_curve)}')
                        
                        return True
                else:
                    print('オートストラテジー結果が見つかりませんでした')
                    return False
                    
            finally:
                db.close()
        else:
            print('実験が完了していません')
            return False
    
    except Exception as e:
        import traceback
        print(f'Error: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_auto_strategy()
    if success:
        print("✅ オートストラテジー機能のテスト成功")
    else:
        print("❌ オートストラテジー機能のテスト失敗")
