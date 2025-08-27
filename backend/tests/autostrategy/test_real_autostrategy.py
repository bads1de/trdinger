#!/usr/bin/env python3
"""
実際のオートストラテジー実行テスト
修正された統計情報抽出機能をテスト
"""

import sys
import os
import time

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_real_autostrategy():
    """実際のオートストラテジーを実行して統計情報をテスト"""
    print("=" * 60)
    print("Real AutoStrategy Test with Fixed Statistics")
    print("=" * 60)

    try:
        from app.services.auto_strategy.services.auto_strategy_service import (
            AutoStrategyService,
        )
        from database.connection import SessionLocal
        from database.repositories.backtest_result_repository import (
            BacktestResultRepository,
        )
        import json

        # オートストラテジーサービス初期化
        print("Initializing AutoStrategy service...")
        auto_strategy_service = AutoStrategyService()

        # 実験設定
        experiment_name = "Test_Statistics_Fix_Real"
        experiment_id = f"test_stats_fix_{int(time.time())}"

        # GAConfig.create_fast()を使って正しい設定を取得
        from app.services.auto_strategy.models.ga_config import GAConfig

        ga_config_obj = GAConfig.create_fast()
        ga_config_obj.indicator_mode = "technical_only"  # テクニカル指標のみに設定
        ga_config_obj.enable_multi_objective = False

        # 辞書形式に変換
        ga_config = ga_config_obj.to_dict()

        backtest_config = {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2020-12-31T00:00:00",
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
        }

        print(f"Starting AutoStrategy experiment: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Symbol: {backtest_config['symbol']}")
        print(f"Timeframe: {backtest_config['timeframe']}")
        print(f"Indicator Mode: {ga_config['indicator_mode']}")
        print(f"Population Size: {ga_config['population_size']}")
        print(f"Generations: {ga_config['generations']}")

        # 直接ExperimentManagerを使って同期実行
        print("Attempting direct experiment execution...")
        if auto_strategy_service.experiment_manager:
            try:
                # 実験マネージャーで直接実行
                auto_strategy_service.experiment_manager.run_experiment(
                    experiment_id=experiment_id,
                    ga_config=ga_config_obj,
                    backtest_config=backtest_config,
                )
                print("[SUCCESS] Experiment completed!")
                return True, {"experiment_id": experiment_id, "status": "completed"}
            except Exception as e:
                print(f"[ERROR] Direct execution failed: {e}")
                return False, str(e)
        else:
            print("[ERROR] Experiment manager not available")
            return False, "Experiment manager not available"

        print(f"Experiment started: {result}")

        # 実験の進行を監視
        max_attempts = 60  # 最大60回待機（約5分）
        attempt = 0

        while attempt < max_attempts:
            try:
                # 実験ステータスを確認
                status = auto_strategy_service.get_experiment_status(experiment_id)

                if status.get("status") == "completed":
                    print("\n[SUCCESS] Experiment completed!")
                    break
                elif status.get("status") == "running":
                    progress = status.get("progress", 0)
                    print(
                        f"[RUNNING] Progress: {progress:.1f}%, attempt {attempt + 1}/{max_attempts}"
                    )
                elif status.get("status") == "error":
                    print(
                        f"[ERROR] Experiment failed: {status.get('message', 'Unknown error')}"
                    )
                    return False

                time.sleep(5)  # 5秒待機
                attempt += 1

            except Exception as e:
                print(f"[ERROR] Failed to get experiment status: {e}")
                time.sleep(5)
                attempt += 1

        if attempt >= max_attempts:
            print("[TIMEOUT] Experiment did not complete within the time limit")
            return False

        # 実験結果を確認
        print("\n" + "=" * 40)
        print("Checking Experiment Results")
        print("=" * 40)

        db = SessionLocal()

        try:
            # 最新のバックテスト結果を取得
            backtest_repo = BacktestResultRepository(db)
            recent_results = backtest_repo.get_recent_backtest_results(limit=5)

            if not recent_results:
                print("No backtest results found")
                return False

            # 実験に関連する結果を探す
            experiment_results = []
            for result in recent_results:
                if experiment_id in str(result.get("strategy_name", "")):
                    experiment_results.append(result)

            if not experiment_results:
                print(f"No results found for experiment {experiment_id}")
                print("Available results:")
                for result in recent_results[:3]:
                    print(
                        f"  - {result.get('strategy_name')}: {result.get('created_at')}"
                    )
                return False

            print(
                f"Found {len(experiment_results)} results for experiment {experiment_id}"
            )

            # 最新の結果を分析
            latest_result = experiment_results[0]
            print("\nLatest Result Analysis:")
            print(f"Strategy Name: {latest_result.get('strategy_name')}")
            print(f"Symbol: {latest_result.get('symbol')}")
            print(f"Timeframe: {latest_result.get('timeframe')}")
            print(f"Initial Capital: {latest_result.get('initial_capital')}")
            print(f"Final Balance: {latest_result.get('final_balance', 0):.2f}")

            # 統計情報の確認
            performance_metrics = latest_result.get("performance_metrics", {})
            if isinstance(performance_metrics, str):
                try:
                    performance_metrics = json.loads(performance_metrics)
                except:
                    performance_metrics = {}

            print("\nPerformance Metrics:")
            critical_metrics = [
                "total_return",
                "total_trades",
                "sharpe_ratio",
                "win_rate",
                "max_drawdown",
            ]

            all_non_zero = True
            for metric in critical_metrics:
                value = performance_metrics.get(metric, 0)
                status = "OK" if value != 0 and value != 0.0 else "ZERO"
                print(f"  {metric}: {value} [{status}]")
                if value == 0 or value == 0.0:
                    all_non_zero = False

            # 取引履歴の確認
            trade_history = latest_result.get("trade_history", [])
            if isinstance(trade_history, str):
                try:
                    trade_history = json.loads(trade_history)
                except:
                    trade_history = []

            print(f"\nTrade History: {len(trade_history)} trades")

            if all_non_zero:
                print("\n[SUCCESS] All critical metrics have non-zero values!")
                print("The statistics extraction fix is working correctly.")
                return True
            else:
                print(
                    "\n[WARNING] Some metrics are still zero - fix may need adjustment"
                )
                return False

        finally:
            db.close()

    except Exception as e:
        print(f"Error in real autostrategy test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_autostrategy()
    print(f"\nFinal result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
