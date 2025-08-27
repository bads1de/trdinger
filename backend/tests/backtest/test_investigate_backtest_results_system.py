#!/usr/bin/env python3
"""
バックテスト結果の詳細調査スクリプト
"""

import sys
import os
import json

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def investigate_backtest_results():
    """バックテスト結果の詳細を調査"""
    print("=" * 60)
    print("Backtest Results Investigation")
    print("=" * 60)

    try:
        from database.connection import SessionLocal
        from database.models import BacktestResult, GeneratedStrategy

        db = SessionLocal()

        try:
            # バックテスト結果の詳細を取得
            backtest_results = db.query(BacktestResult).order_by(
                BacktestResult.created_at.desc()
            ).limit(5).all()

            if not backtest_results:
                print("No backtest results found.")
                return

            print(f"\nFound {len(backtest_results)} backtest results:")

            for i, result in enumerate(backtest_results, 1):
                print(f"\n--- Backtest Result {i} ---")
                print(f"ID: {result.id}")
                print(f"Strategy Name: {result.strategy_name}")
                print(f"Symbol: {result.symbol}")
                print(f"Timeframe: {result.timeframe}")
                print(f"Start Date: {result.start_date}")
                print(f"End Date: {result.end_date}")
                print(f"Initial Capital: {result.initial_capital}")
                print(f"Commission Rate: {result.commission_rate}")
                print(f"Status: {result.status}")
                print(f"Execution Time: {result.execution_time}")
                print(f"Error Message: {result.error_message}")

                # パフォーマンス指標の詳細
                if result.performance_metrics:
                    print("\nPerformance Metrics:")
                    if isinstance(result.performance_metrics, str):
                        try:
                            metrics = json.loads(result.performance_metrics)
                        except:
                            metrics = {"raw": result.performance_metrics}
                    else:
                        metrics = result.performance_metrics

                    for key, value in metrics.items():
                        print(f"  {key}: {value}")

                # 設定データの詳細
                if result.config_json:
                    print("\nConfig JSON:")
                    if isinstance(result.config_json, str):
                        try:
                            config = json.loads(result.config_json)
                        except:
                            config = {"raw": result.config_json}
                    else:
                        config = result.config_json

                    # 重要な設定のみ表示
                    important_keys = ['strategy_name', 'symbol', 'timeframe', 'initial_capital']
                    for key in important_keys:
                        if key in config:
                            print(f"  {key}: {config[key]}")

                # 取引履歴の確認
                if result.trade_history:
                    print("\nTrade History:")
                    if isinstance(result.trade_history, str):
                        try:
                            trades = json.loads(result.trade_history)
                        except:
                            trades = [{"raw": result.trade_history}]
                    else:
                        trades = result.trade_history

                    if isinstance(trades, list):
                        print(f"  Number of trades: {len(trades)}")
                        if len(trades) > 0:
                            print("  First trade sample:")
                            for key, value in list(trades[0].items())[:5]:
                                print(f"    {key}: {value}")
                    else:
                        print(f"  Trade history type: {type(trades)}")

                # 資産曲線の確認
                if result.equity_curve:
                    print("\nEquity Curve:")
                    if isinstance(result.equity_curve, str):
                        try:
                            equity = json.loads(result.equity_curve)
                        except:
                            equity = [{"raw": result.equity_curve}]
                    else:
                        equity = result.equity_curve

                    if isinstance(equity, list) and len(equity) > 0:
                        print(f"  Number of equity points: {len(equity)}")
                        print(f"  First equity value: {equity[0] if len(equity) > 0 else 'N/A'}")
                        print(f"  Last equity value: {equity[-1] if len(equity) > 0 else 'N/A'}")
                    else:
                        print(f"  Equity curve type: {type(equity)}")

        finally:
            db.close()

    except Exception as e:
        print(f"Error investigating backtest results: {e}")
        import traceback
        traceback.print_exc()

def check_ohlcv_data():
    """OHLCVデータの存在を確認"""
    print("\n" + "=" * 40)
    print("OHLCV Data Check")
    print("=" * 40)

    try:
        from database.connection import SessionLocal
        from database.models import OHLCVData

        db = SessionLocal()

        try:
            # BTC/USDTのデータを確認
            btc_data_count = db.query(OHLCVData).filter(
                OHLCVData.symbol.like('%BTC%')
            ).count()

            print(f"BTC-related OHLCV data: {btc_data_count} records")

            if btc_data_count > 0:
                # 最新のデータを確認
                latest_data = db.query(OHLCVData).filter(
                    OHLCVData.symbol.like('%BTC%')
                ).order_by(OHLCVData.timestamp.desc()).first()

                if latest_data:
                    print(f"Latest BTC data: {latest_data.symbol}, {latest_data.timestamp}, Close: {latest_data.close}")

                # 1時間足のデータを確認
                hourly_data = db.query(OHLCVData).filter(
                    OHLCVData.symbol.like('%BTC%'),
                    OHLCVData.timeframe == '1h'
                ).count()

                print(f"BTC hourly data: {hourly_data} records")

        finally:
            db.close()

    except Exception as e:
        print(f"Error checking OHLCV data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_backtest_results()
    check_ohlcv_data()