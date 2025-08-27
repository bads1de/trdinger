#!/usr/bin/env python3
"""
全戦略データの確認スクリプト
"""

import sys
import os

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_all_strategies():
    """データベース内の全戦略データを確認"""
    print("=" * 60)
    print("Database Strategy Analysis")
    print("=" * 60)

    try:
        from database.connection import SessionLocal
        from database.models import GeneratedStrategy, BacktestResult

        db = SessionLocal()

        try:
            # 生成戦略の総数を取得
            total_strategies = db.query(GeneratedStrategy).count()
            print(f"\nTotal Generated Strategies: {total_strategies}")

            if total_strategies > 0:
                # 最新の戦略を取得
                recent_strategies = db.query(GeneratedStrategy).order_by(
                    GeneratedStrategy.created_at.desc()
                ).limit(5).all()

                print("\nRecent Generated Strategies:")
                for i, strategy in enumerate(recent_strategies, 1):
                    print(f"\n--- Strategy {i} ---")
                    print(f"ID: {strategy.id}")
                    print(f"Experiment ID: {strategy.experiment_id}")
                    print(f"Generation: {strategy.generation}")
                    print(f"Fitness Score: {strategy.fitness_score}")
                    print(f"Backtest Result ID: {strategy.backtest_result_id}")
                    print(f"Created At: {strategy.created_at}")

                    # 遺伝子データの概要
                    gene_data = strategy.gene_data
                    if gene_data and isinstance(gene_data, dict):
                        indicators = gene_data.get('indicators', [])
                        print(f"Indicators: {[ind.get('type', 'Unknown') for ind in indicators]}")

                        entry_conditions = gene_data.get('entry_conditions', [])
                        print(f"Entry Conditions: {len(entry_conditions)}")

                        exit_conditions = gene_data.get('exit_conditions', [])
                        print(f"Exit Conditions: {len(exit_conditions)}")

                        risk_management = gene_data.get('risk_management', {})
                        if risk_management:
                            print(f"Risk Management: {risk_management}")

            # バックテスト結果の総数を取得
            total_backtests = db.query(BacktestResult).count()
            print(f"\nTotal Backtest Results: {total_backtests}")

            if total_backtests > 0:
                # 最新のバックテスト結果を取得
                recent_backtests = db.query(BacktestResult).order_by(
                    BacktestResult.created_at.desc()
                ).limit(5).all()

                print("\nRecent Backtest Results:")
                for i, result in enumerate(recent_backtests, 1):
                    print(f"\n--- Backtest Result {i} ---")
                    print(f"ID: {result.id}")
                    print(f"Strategy Name: {result.strategy_name}")
                    print(f"Symbol: {result.symbol}")
                    print(f"Timeframe: {result.timeframe}")
                    print(f"Final Balance: {result.final_balance}")
                    print(f"Total Return: {result.performance_metrics.get('total_return', 0) if result.performance_metrics else 0}")
                    print(f"Total Trades: {result.performance_metrics.get('total_trades', 0) if result.performance_metrics else 0}")
                    print(f"Created At: {result.created_at}")

            # 関連付けの確認
            print("\n" + "=" * 40)
            print("Strategy-Backtest Relationship Analysis")
            print("=" * 40)

            # backtest_result_idを持つ戦略
            strategies_with_backtest = db.query(GeneratedStrategy).filter(
                GeneratedStrategy.backtest_result_id.isnot(None)
            ).count()
            print(f"Strategies with Backtest Result ID: {strategies_with_backtest}")

            # 各バックテスト結果に関連する戦略数
            for result in recent_backtests:
                related_strategies = db.query(GeneratedStrategy).filter(
                    GeneratedStrategy.backtest_result_id == result.id
                ).count()
                print(f"Backtest Result {result.id} has {related_strategies} related strategies")

        finally:
            db.close()

    except Exception as e:
        print(f"Error analyzing database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all_strategies()