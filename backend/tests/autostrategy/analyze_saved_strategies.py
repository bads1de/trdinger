#!/usr/bin/env python3
"""
保存された戦略の詳細分析スクリプト
"""

import sys
import os
import json
from typing import Dict, Any, List

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_saved_strategies():
    """保存された戦略を分析"""
    print("=" * 60)
    print("Saved Strategies Analysis")
    print("=" * 60)

    try:
        from database.connection import SessionLocal
        from database.repositories.backtest_result_repository import BacktestResultRepository
        from database.repositories.generated_strategy_repository import GeneratedStrategyRepository

        db = SessionLocal()

        try:
            # バックテスト結果を取得
            backtest_repo = BacktestResultRepository(db)
            recent_results = backtest_repo.get_recent_backtest_results(limit=10)

            if not recent_results:
                print("No backtest results found.")
                return

            print(f"\nFound {len(recent_results)} backtest results:")

            # 戦略リポジトリ
            strategy_repo = GeneratedStrategyRepository(db)

            for i, result in enumerate(recent_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"ID: {result.get('id')}")
                print(f"Strategy Name: {result.get('strategy_name')}")
                print(f"Symbol: {result.get('symbol')}")
                print(f"Timeframe: {result.get('timeframe')}")
                print(f"Initial Capital: {result.get('initial_capital')}")
                print(f"Final Balance: {result.get('final_balance', 0):.2f}")
                print(f"Total Return: {result.get('total_return', 0):.4f}")
                print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.4f}")
                print(f"Max Drawdown: {result.get('max_drawdown', 0):.4f}")
                print(f"Win Rate: {result.get('win_rate', 0):.4f}")
                print(f"Total Trades: {result.get('total_trades', 0)}")
                print(f"Created At: {result.get('created_at')}")

                # 関連する生成戦略を取得
                backtest_result_id = result.get('id')
                if backtest_result_id:
                    from database.models import GeneratedStrategy
                    strategies = db.query(GeneratedStrategy).filter(
                        GeneratedStrategy.backtest_result_id == backtest_result_id
                    ).all()

                    if strategies:
                        print(f"\nRelated Generated Strategies: {len(strategies)}")
                        for j, strategy in enumerate(strategies, 1):
                            print(f"\n  Strategy {j}:")
                            print(f"    ID: {strategy.id}")
                            print(f"    Experiment ID: {strategy.experiment_id}")
                            print(f"    Generation: {strategy.generation}")
                            print(f"    Fitness Score: {strategy.fitness_score:.4f}")

                            # 遺伝子データを解析
                            gene_data = strategy.gene_data
                            if gene_data:
                                indicators = gene_data.get('indicators', [])
                                print(f"    Indicators: {[ind.get('type', 'Unknown') for ind in indicators]}")

                                # エントリー条件
                                entry_conditions = gene_data.get('entry_conditions', [])
                                if entry_conditions:
                                    print(f"    Entry Conditions: {len(entry_conditions)} conditions")

                                # イグジット条件
                                exit_conditions = gene_data.get('exit_conditions', [])
                                if exit_conditions:
                                    print(f"    Exit Conditions: {len(exit_conditions)} conditions")

                                # リスク管理
                                risk_management = gene_data.get('risk_management', {})
                                if risk_management:
                                    print(f"    Risk Management: {risk_management}")

                    else:
                        print("\nNo related generated strategies found.")

        finally:
            db.close()

    except Exception as e:
        print(f"Error analyzing strategies: {e}")
        import traceback
        traceback.print_exc()

def analyze_strategy_details():
    """戦略の詳細をより深く分析"""
    print("\n" + "=" * 60)
    print("Detailed Strategy Analysis")
    print("=" * 60)

    try:
        from database.connection import SessionLocal
        from database.repositories.generated_strategy_repository import GeneratedStrategyRepository

        db = SessionLocal()

        try:
            strategy_repo = GeneratedStrategyRepository(db)

            # バックテスト結果を持つ最新の戦略を取得
            strategies_with_backtest = strategy_repo.get_strategies_with_backtest_results(limit=5)

            if not strategies_with_backtest:
                print("No strategies with backtest results found.")
                return

            print(f"\nAnalyzing {len(strategies_with_backtest)} strategies with backtest results:")

            for i, strategy in enumerate(strategies_with_backtest, 1):
                print(f"\n--- Strategy {i} ---")
                print(f"Strategy ID: {strategy.id}")
                print(f"Experiment ID: {strategy.experiment_id}")
                print(f"Generation: {strategy.generation}")
                print(f"Fitness Score: {strategy.fitness_score:.4f}")

                # 遺伝子データの詳細分析
                gene_data = strategy.gene_data
                if gene_data:
                    print("\nGene Data Analysis:")

                    # インジケータ分析
                    indicators = gene_data.get('indicators', [])
                    print(f"  Indicators ({len(indicators)}):")
                    for j, ind in enumerate(indicators, 1):
                        ind_type = ind.get('type', 'Unknown')
                        ind_enabled = ind.get('enabled', True)
                        ind_params = ind.get('parameters', {})
                        print(f"    {j}. {ind_type} (enabled: {ind_enabled})")
                        if ind_params:
                            print(f"       Parameters: {ind_params}")

                    # 条件分析
                    entry_conditions = gene_data.get('entry_conditions', [])
                    print(f"  Entry Conditions ({len(entry_conditions)}):")
                    for j, cond in enumerate(entry_conditions, 1):
                        if isinstance(cond, dict):
                            cond_type = cond.get('type', 'Unknown')
                            print(f"    {j}. {cond_type}")
                        else:
                            print(f"    {j}. {type(cond).__name__}")

                    # リスク管理分析
                    risk_management = gene_data.get('risk_management', {})
                    if risk_management:
                        print(f"  Risk Management:")
                        for key, value in risk_management.items():
                            print(f"    {key}: {value}")

                    # メタデータ分析
                    metadata = gene_data.get('metadata', {})
                    if metadata:
                        print(f"  Metadata:")
                        for key, value in metadata.items():
                            print(f"    {key}: {value}")

                # バックテスト結果の分析
                if strategy.backtest_result:
                    bt_result = strategy.backtest_result
                    print(f"\nBacktest Results:")
                    print(f"  Strategy Name: {bt_result.strategy_name}")
                    print(f"  Symbol: {bt_result.symbol}")
                    print(f"  Total Return: {bt_result.performance_metrics.get('total_return', 0):.4f}")
                    print(f"  Sharpe Ratio: {bt_result.performance_metrics.get('sharpe_ratio', 0):.4f}")
                    print(f"  Max Drawdown: {bt_result.performance_metrics.get('max_drawdown', 0):.4f}")
                    print(f"  Win Rate: {bt_result.performance_metrics.get('win_rate', 0):.4f}")
                    print(f"  Total Trades: {bt_result.performance_metrics.get('total_trades', 0)}")

        finally:
            db.close()

    except Exception as e:
        print(f"Error in detailed analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_saved_strategies()
    analyze_strategy_details()