#!/usr/bin/env python3
"""
バックテスト結果をJSON形式でエクスポートするスクリプト
データベースから最新のバックテスト結果を取得してJSONファイルに保存
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def export_latest_backtest_to_json():
    """最新のバックテスト結果をJSON形式でエクスポート"""
    print("=== Backtest Result JSON Export ===")

    try:
        from database.repositories.backtest_result_repository import BacktestResultRepository
        from database.connection import SessionLocal

        db = SessionLocal()

        try:
            repo = BacktestResultRepository(db)

            # 最新のバックテスト結果を取得
            recent_results = repo.get_recent_backtest_results(limit=1)

            if not recent_results:
                print("WARNING: No backtest results found in database")
                return False

            latest_result = recent_results[0]
            print(f"SUCCESS: Retrieved latest backtest result: ID {latest_result.get('id')}")

            # JSONファイルに保存
            filename = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(project_root, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(latest_result, f, indent=2, ensure_ascii=False, default=str)

            print(f"SUCCESS: Backtest result saved to {filepath}")

            # 内容を確認
            print("\n=== Backtest Result Details ===")
            print(f"ID: {latest_result.get('id')}")
            print(f"Strategy Name: {latest_result.get('strategy_name')}")
            print(f"Symbol: {latest_result.get('symbol')}")
            print(f"Timeframe: {latest_result.get('timeframe')}")
            print(f"Initial Capital: {latest_result.get('initial_capital', 0):.2f}")
            print(f"Final Balance: {latest_result.get('final_balance', 0):.2f}")
            print(f"Total Return: {latest_result.get('total_return', 0):.4f}")
            print(f"Total Trades: {latest_result.get('total_trades', 0)}")
            print(f"Win Rate: {latest_result.get('win_rate', 0):.2f}%")
            print(f"Max Drawdown: {latest_result.get('max_drawdown', 0):.2f}%")
            print(f"Sharpe Ratio: {latest_result.get('sharpe_ratio', 0):.4f}")
            print(f"Created At: {latest_result.get('created_at')}")

            # トレード数やリターンが0の場合を確認
            issues_found = []
            if latest_result.get('total_trades', 0) == 0:
                issues_found.append("total_trades is 0")
            if latest_result.get('total_return', 0) == 0:
                issues_found.append("total_return is 0")
            if latest_result.get('final_balance', 0) == 0:
                issues_found.append("final_balance is 0")

            if issues_found:
                print(f"\nWARNING: Issues found: {', '.join(issues_found)}")
                return False, issues_found
            else:
                print("\nSUCCESS: All key metrics have valid values")
                return True, None

        finally:
            db.close()

    except Exception as e:
        print(f"FAILED: Export error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """メイン実行関数"""
    success, issues = export_latest_backtest_to_json()

    if success:
        print("\nSUCCESS: Backtest result exported successfully")
        return 0
    else:
        print(f"\nFAILED: Export failed - {issues}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)