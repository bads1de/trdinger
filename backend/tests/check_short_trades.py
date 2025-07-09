#!/usr/bin/env python3
"""
ショート取引確認スクリプト

データベースから最新のバックテスト結果を取得し、
取引履歴にショートポジションが含まれているかを確認します。
"""

import sys
import os
import json
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import SessionLocal
from database.repositories.backtest_result_repository import BacktestResultRepository

def analyze_trade_history(trade_history):
    """取引履歴を分析"""
    if not trade_history:
        return {
            "total_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "long_percentage": 0,
            "short_percentage": 0,
            "sample_trades": []
        }
    
    long_trades = []
    short_trades = []
    
    for trade in trade_history:
        size = trade.get("size", 0)
        if size > 0:
            long_trades.append(trade)
        elif size < 0:
            short_trades.append(trade)
    
    total = len(trade_history)
    long_count = len(long_trades)
    short_count = len(short_trades)
    
    return {
        "total_trades": total,
        "long_trades": long_count,
        "short_trades": short_count,
        "long_percentage": (long_count / total * 100) if total > 0 else 0,
        "short_percentage": (short_count / total * 100) if total > 0 else 0,
        "sample_trades": trade_history[:5]  # 最初の5取引をサンプルとして
    }

def check_recent_backtest_results():
    """最近のバックテスト結果を確認"""
    print("=== 最近のバックテスト結果確認 ===")
    
    db = SessionLocal()
    try:
        repo = BacktestResultRepository(db)
        
        # 最新の10件を取得
        results = repo.get_backtest_results(limit=10)
        
        if not results:
            print("❌ バックテスト結果が見つかりませんでした")
            return
        
        print(f"✅ {len(results)}件のバックテスト結果を取得")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- 結果 {i} ---")
            print(f"戦略名: {result.get('strategy_name', 'N/A')}")
            print(f"シンボル: {result.get('symbol', 'N/A')}")
            print(f"期間: {result.get('start_date', 'N/A')} - {result.get('end_date', 'N/A')}")
            print(f"作成日時: {result.get('created_at', 'N/A')}")
            
            # 取引履歴を分析
            trade_history = result.get('trade_history', [])
            analysis = analyze_trade_history(trade_history)
            
            print(f"取引分析:")
            print(f"  総取引数: {analysis['total_trades']}")
            print(f"  ロング取引: {analysis['long_trades']} ({analysis['long_percentage']:.1f}%)")
            print(f"  ショート取引: {analysis['short_trades']} ({analysis['short_percentage']:.1f}%)")
            
            if analysis['sample_trades']:
                print(f"  サンプル取引:")
                for j, trade in enumerate(analysis['sample_trades'], 1):
                    size = trade.get('size', 0)
                    direction = "LONG" if size > 0 else "SHORT" if size < 0 else "NEUTRAL"
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    pnl = trade.get('pnl', 0)
                    print(f"    {j}. {direction} - サイズ: {size:.4f}, エントリー: {entry_price:.2f}, エグジット: {exit_price:.2f}, P/L: {pnl:.2f}")
            
            # ショート取引が見つかった場合は詳細表示
            if analysis['short_trades'] > 0:
                print(f"🎯 ショート取引発見！")
                break
        
        # 全体統計
        print(f"\n=== 全体統計 ===")
        total_trades_all = sum(analyze_trade_history(r.get('trade_history', []))['total_trades'] for r in results)
        total_long_all = sum(analyze_trade_history(r.get('trade_history', []))['long_trades'] for r in results)
        total_short_all = sum(analyze_trade_history(r.get('trade_history', []))['short_trades'] for r in results)
        
        print(f"全結果の総取引数: {total_trades_all}")
        print(f"全結果のロング取引: {total_long_all} ({total_long_all/total_trades_all*100:.1f}%)" if total_trades_all > 0 else "全結果のロング取引: 0")
        print(f"全結果のショート取引: {total_short_all} ({total_short_all/total_trades_all*100:.1f}%)" if total_trades_all > 0 else "全結果のショート取引: 0")
        
        if total_short_all == 0:
            print("❌ 全ての結果でショート取引が0件です")
        else:
            print("✅ ショート取引が確認されました")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def check_auto_strategy_results():
    """AUTO_STRATEGY結果を特別に確認"""
    print("\n=== AUTO_STRATEGY結果確認 ===")
    
    db = SessionLocal()
    try:
        repo = BacktestResultRepository(db)
        
        # AUTO_STRATEGYの結果を検索
        results = repo.get_backtest_results_by_strategy("AUTO_STRATEGY")
        
        if not results:
            print("❌ AUTO_STRATEGY結果が見つかりませんでした")
            return
        
        print(f"✅ {len(results)}件のAUTO_STRATEGY結果を取得")
        
        for i, result in enumerate(results[:3], 1):  # 最新3件のみ
            print(f"\n--- AUTO_STRATEGY結果 {i} ---")
            print(f"作成日時: {result.get('created_at', 'N/A')}")
            
            # 取引履歴を詳細分析
            trade_history = result.get('trade_history', [])
            analysis = analyze_trade_history(trade_history)
            
            print(f"取引詳細:")
            print(f"  総取引数: {analysis['total_trades']}")
            print(f"  ロング取引: {analysis['long_trades']}")
            print(f"  ショート取引: {analysis['short_trades']}")
            
            if analysis['sample_trades']:
                print(f"  全取引詳細:")
                for j, trade in enumerate(trade_history, 1):
                    size = trade.get('size', 0)
                    direction = "LONG" if size > 0 else "SHORT" if size < 0 else "NEUTRAL"
                    entry_time = trade.get('entry_time', 'N/A')
                    print(f"    {j}. {direction} - サイズ: {size:.6f}, 時刻: {entry_time}")
                    
                    if j >= 10:  # 最大10取引まで表示
                        print(f"    ... (残り{len(trade_history)-10}取引)")
                        break
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def main():
    """メイン実行"""
    print("🔍 ショート取引確認開始\n")
    
    try:
        check_recent_backtest_results()
        check_auto_strategy_results()
        
        print("\n✅ 確認完了")
        print("\n📋 確認ポイント:")
        print("1. 最近のバックテスト結果にショート取引が含まれているか")
        print("2. AUTO_STRATEGY結果でショート取引が発生しているか")
        print("3. 取引履歴のsizeフィールドが負の値になっているか")
        
    except Exception as e:
        print(f"❌ 確認エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
