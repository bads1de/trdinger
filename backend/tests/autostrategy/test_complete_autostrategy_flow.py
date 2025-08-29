#!/usr/bin/env python3
"""
完全なオートストラテジーフロー実行テスト
新しく戦略を作成してバックテスト結果がDBに保存されるまでを確認
"""

import sys
import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mock_data():
    """モックデータを作成してデータベースに投入"""
    print("=== Creating Mock Data ===")

    try:
        from database.connection import SessionLocal
        from database.models import OHLCVData

        db = SessionLocal()

        try:
            # 既存のデータを確認
            existing_count = db.query(OHLCVData).filter(
                OHLCVData.symbol == 'BTC/USDT',
                OHLCVData.timeframe == '1d'
            ).count()

            if existing_count > 0:
                print(f"Mock data already exists: {existing_count} records")
                return True

            # モックデータの作成（2020年1月-12月）
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2020, 12, 31)
            current_date = start_date

            mock_data = []
            base_price = 10000.0

            while current_date <= end_date:
                # シンプルな価格変動をシミュレート
                import random
                price_change = random.uniform(-0.05, 0.05)  # -5% to +5%
                open_price = base_price
                close_price = base_price * (1 + price_change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
                volume = random.uniform(1000, 10000)

                mock_data.append(OHLCVData(
                    symbol='BTC/USDT',
                    timeframe='1d',
                    timestamp=current_date,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume
                ))

                # 次の日に進む
                current_date += timedelta(days=1)
                base_price = close_price

            # データベースに保存
            db.add_all(mock_data)
            db.commit()

            print(f"Created mock data: {len(mock_data)} records")
            return True

        except Exception as e:
            print(f"Error creating mock data: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    except Exception as e:
        print(f"Error in mock data creation: {e}")
        return False

def get_current_backtest_count():
    """現在のバックテスト結果数を確認"""
    print("=== Current Backtest Results Count ===")

    try:
        from database.connection import SessionLocal
        from database.models import BacktestResult

        db = SessionLocal()

        try:
            count = db.query(BacktestResult).count()
            print(f"Current backtest results: {count}")

            if count > 0:
                # 最新の結果を確認
                latest = db.query(BacktestResult).order_by(
                    BacktestResult.created_at.desc()
                ).first()

                if latest:
                    print(f"Latest result ID: {latest.id}")
                    print(f"Latest result created: {latest.created_at}")

            return count

        finally:
            db.close()

    except Exception as e:
        print(f"Error checking current count: {e}")
        return 0

def create_new_autostrategy_experiment():
    """新しくオートストラテジー実験を作成"""
    print("\n=== Creating New AutoStrategy Experiment ===")

    try:
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
        from database.connection import SessionLocal
        from database.repositories.backtest_result_repository import BacktestResultRepository

        # オートストラテジーサービス初期化
        print("Initializing AutoStrategy service...")
        auto_strategy_service = AutoStrategyService()

        # 実験設定
        experiment_name = "Complete_Test_Statistics_Fix"
        experiment_id = f"complete_test_{int(time.time())}"

        # GAConfig.create_fast()を使って正しい設定を取得
        from app.services.auto_strategy.config.auto_strategy_config import GAConfig
        ga_config_obj = GAConfig.create_fast()
        ga_config_obj.indicator_mode = 'technical_only'  # テクニカル指標のみに設定
        ga_config_obj.enable_multi_objective = False
        ga_config_obj.generations = 3  # テスト用に世代数を減らす
        ga_config_obj.population_size = 5  # テスト用に個体数を減らす

        # 辞書形式に変換
        ga_config = ga_config_obj.to_dict()

        backtest_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_date': '2020-01-01T00:00:00',
            'end_date': '2020-06-01T00:00:00',  # 期間を短くしてテストを速く
            'initial_capital': 100000.0,
            'commission_rate': 0.001
        }

        print(f"Starting NEW AutoStrategy experiment: {experiment_name}")
        print(f"New Experiment ID: {experiment_id}")
        print(f"Symbol: {backtest_config['symbol']}")
        print(f"Timeframe: {backtest_config['timeframe']}")
        print(f"Date range: {backtest_config['start_date']} to {backtest_config['end_date']}")
        print(f"Indicator Mode: {ga_config['indicator_mode']}")
        print(f"Population Size: {ga_config['population_size']}")
        print(f"Generations: {ga_config['generations']}")

        # 実験マネージャーで直接実行
        print("Running experiment with GA engine...")
        if auto_strategy_service.experiment_manager:
            try:
                # GAエンジンを初期化
                auto_strategy_service.experiment_manager.initialize_ga_engine(ga_config_obj)

                # 実験マネージャーで直接実行
                auto_strategy_service.experiment_manager.run_experiment(
                    experiment_id=experiment_id,
                    ga_config=ga_config_obj,
                    backtest_config=backtest_config,
                )
                print("[SUCCESS] New experiment completed!")
                return True, experiment_id
            except Exception as e:
                print(f"[ERROR] Experiment execution failed: {e}")
                import traceback
                traceback.print_exc()
                return False, str(e)
        else:
            print("[ERROR] Experiment manager not available")
            return False, "Experiment manager not available"

    except Exception as e:
        print(f"Error in experiment creation: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def check_new_backtest_results(initial_count):
    """新しいバックテスト結果が作成されたか確認"""
    print(f"\n=== Checking for New Backtest Results (Initial: {initial_count}) ===")

    try:
        from database.connection import SessionLocal
        from database.models import BacktestResult

        db = SessionLocal()

        try:
            new_count = db.query(BacktestResult).count()
            print(f"Current backtest results: {new_count}")

            if new_count > initial_count:
                print(f"[SUCCESS] New results created! ({new_count - initial_count} new)")

                # 新しい結果を取得
                new_results = db.query(BacktestResult).order_by(
                    BacktestResult.created_at.desc()
                ).limit(new_count - initial_count).all()

                for i, result in enumerate(new_results, 1):
                    print(f"\n--- New Result {i} ---")
                    print(f"ID: {result.id}")
                    print(f"Strategy Name: {result.strategy_name}")
                    print(f"Symbol: {result.symbol}")
                    print(f"Timeframe: {result.timeframe}")
                    print(f"Created At: {result.created_at}")

                    # パフォーマンス指標の確認
                    if result.performance_metrics:
                        metrics = result.performance_metrics
                        if isinstance(metrics, str):
                            try:
                                metrics = json.loads(metrics)
                            except:
                                metrics = {}

                        print("Performance Metrics:")
                        key_metrics = ['total_return', 'total_trades', 'sharpe_ratio', 'win_rate', 'max_drawdown']
                        for metric in key_metrics:
                            value = metrics.get(metric, 0)
                            status = "OK" if value != 0 and value != 0.0 else "ZERO"
                            print(f"  {metric}: {value} [{status}]")

                    # 取引履歴の確認
                    if result.trade_history:
                        trades = result.trade_history
                        if isinstance(trades, str):
                            try:
                                trades = json.loads(trades)
                            except:
                                trades = []

                        print(f"Trade History: {len(trades)} trades")

                    # 資産曲線の確認
                    if result.equity_curve:
                        equity = result.equity_curve
                        if isinstance(equity, str):
                            try:
                                equity = json.loads(equity)
                            except:
                                equity = []

                        print(f"Equity Curve: {len(equity)} points")
                        if len(equity) > 0:
                            print(f"  First equity: {equity[0].get('equity', 'N/A')}")
                            print(f"  Last equity: {equity[-1].get('equity', 'N/A')}")

                return True, new_count - initial_count
            else:
                print("[INFO] No new results found")
                return False, 0

        finally:
            db.close()

    except Exception as e:
        print(f"Error checking new results: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def main():
    """メイン実行関数"""
    print("=" * 70)
    print("Complete AutoStrategy Flow Test")
    print("Testing full cycle: Strategy Creation -> Backtest -> DB Save")
    print("=" * 70)

    # 0. モックデータを作成
    if not create_mock_data():
        print("\n[FAILED] Mock data creation failed")
        return 1

    # 1. 現在のバックテスト結果数を確認
    initial_count = get_current_backtest_count()

    # 2. 新しいオートストラテジー実験を作成・実行
    success, experiment_id = create_new_autostrategy_experiment()

    if not success:
        print(f"\n[FAILED] Experiment creation failed: {experiment_id}")
        return 1

    # 3. 新しいバックテスト結果が作成されたか確認
    results_created, new_count = check_new_backtest_results(initial_count)

    if results_created:
        print("\n[COMPLETE SUCCESS] Full AutoStrategy flow test passed!")
        print(f"New experiment ID: {experiment_id}")
        print(f"New backtest results: {new_count}")
        print("Strategy creation -> Backtest execution -> Database save: ALL SUCCESS")
        return 0
    else:
        print("\n[FAILED] No new backtest results found")
        print("Strategy may have been created but backtest results were not saved")
        return 1

if __name__ == "__main__":
    success_code = main()
    print(f"\nFinal result: {'PASS' if success_code == 0 else 'FAIL'}")
    sys.exit(success_code)